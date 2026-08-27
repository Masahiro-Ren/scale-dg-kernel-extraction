//------------------------------------------------------------------------------
// Batched DG element operations for p=7 (Nq=8): volume-flux divergence
// (three tensor contractions) + surface lift + combine, consuming
// precomputed volume-flux and boundary-flux arrays.
//
// This is the CUDA backend of the LAYERED tendency structure: the physics
// (boundary flux, volume flux) stays in OpenACC Fortran, and this kernel
// implements only the equation-independent element operations - the part
// that in FE-Project sits behind the element-operation library interface.
//
// Granularity: ONE WARP PER ELEMENT (4 elements per 128-thread block).
// The tensor contraction is a warp-collective device function
//   C(8x64) = D1D(8x8) x B(8x64)
// in two interchangeable flavors selected at launch:
//   - warp_contract_fma:  FP64 FMA, 16 output columns per lane
//   - warp_contract_dmma: inline-PTX m8n8k4 FP64 tensor cores (Tu et al.),
//     8 tiles per warp, A fragments (D1D) loaded once and reused
// All three directions use the same canonical GEMM on cyclically
// transposed layouts (summed index fastest). Per-direction results are
// folded into per-lane register accumulators, so C tiles round-trip
// through one reused shared buffer only.
//------------------------------------------------------------------------------
#include <cuda_runtime.h>

#define NQ  8
#define NP  512
#define NFP 64
#define NFT 384
#define EPB 4     // elements per block (one warp each)
#define NT  128   // threads per block

#define LDB 12    // bank-conflict-free ld for fragment loads
#define LDC 10    // bank-conflict-free ld for fragment stores

__device__ __forceinline__ void dmma_m8n8k4(double &c0, double &c1,
                                            const double a, const double b) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
      "{%0,%1}, {%2}, {%3}, {%0,%1};"
      : "+d"(c0), "+d"(c1)
      : "d"(a), "d"(b));
}

// Warp-collective contraction, FP64 FMA flavor.
// Contract: all 32 lanes call together with uniform arguments.
__device__ __forceinline__ void warp_contract_fma(const double *sD,
                                                  const double (*sB)[LDB],
                                                  double (*sC)[LDC],
                                                  const int lane) {
  for (int m = lane; m < NP; m += 32) {
    const int row = m & 7, col = m >> 3;
    double s = 0.0;
#pragma unroll
    for (int l = 0; l < 8; ++l) s += sD[row + 8 * l] * sB[col][l];
    sC[col][row] = s;
  }
}

// Warp-collective contraction, FP64 tensor-core (DMMA) flavor.
// Lane maps per Tu et al. Tables III-V; A fragments loaded once.
__device__ __forceinline__ void warp_contract_dmma(const double *sD,
                                                   const double (*sB)[LDB],
                                                   double (*sC)[LDC],
                                                   const int lane) {
  const int rowA = lane >> 2, colA = lane & 3;
  const int rowB = lane & 3, colB = lane >> 2;
  const int rowC = lane >> 2, colC = 2 * (lane & 3);
  const double a0 = sD[rowA + 8 * colA];
  const double a1 = sD[rowA + 8 * (colA + 4)];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    double c0 = 0.0, c1 = 0.0;
    dmma_m8n8k4(c0, c1, a0, sB[8 * nt + colB][rowB]);
    dmma_m8n8k4(c0, c1, a1, sB[8 * nt + colB][rowB + 4]);
    sC[8 * nt + colC][rowC] = c0;
    sC[8 * nt + colC + 1][rowC] = c1;
  }
}

// node n -> (row, col) of the canonical GEMM for direction d
__device__ __forceinline__ void node_to_rc(const int n, const int d, int &row,
                                           int &col) {
  const int i = n & 7, jk = n >> 3, j = jk & 7, k = jk >> 3;
  if (d == 0) {
    row = i; col = jk;          // x: summed index i fastest
  } else if (d == 1) {
    row = j; col = i + 8 * k;   // y: summed index j fastest
  } else {
    row = k; col = i + 8 * j;   // z: summed index k fastest
  }
}

extern "C" __global__ void __launch_bounds__(NT)
cal_eop_p7_kernel(double *__restrict__ dqdt,
                  const double *__restrict__ fx,
                  const double *__restrict__ fy,
                  const double *__restrict__ fz,
                  const double *__restrict__ ebnd,    // (384, Ne)
                  const double *__restrict__ D1D,     // (8,8) col-major
                  const double *__restrict__ Lift,    // (8,8,8,6)
                  const double *__restrict__ escale,  // (512, Ne, 3)
                  const int Ne, const int NeA, const int use_tc) {
  __shared__ double sD[64];
  __shared__ double sB[EPB][NFP][LDB];
  __shared__ double sC[EPB][NFP][LDC];

  const int t = threadIdx.x;
  const int warp = t >> 5, lane = t & 31;
  const int ke = blockIdx.x * EPB + warp;

  if (t < 64) sD[t] = D1D[t];
  __syncthreads();
  if (ke >= Ne) return;

  const long ebase = (long)ke * NP;
  const double *flux[3] = {fx, fy, fz};
  double acc[16];
  int row, col;

  for (int d = 0; d < 3; ++d) {
    // stage this element's volume flux in the cyclic layout
    for (int m = lane; m < NP; m += 32) {
      node_to_rc(m, d, row, col);
      sB[warp][col][row] = flux[d][ebase + m];
    }
    __syncwarp();
    if (use_tc) {
      warp_contract_dmma(sD, sB[warp], sC[warp], lane);
    } else {
      warp_contract_fma(sD, sB[warp], sC[warp], lane);
    }
    __syncwarp();
    // fold Escale_d * C into per-lane accumulators (node-major ownership)
    for (int m2 = 0; m2 < 16; ++m2) {
      const int n = lane + 32 * m2;
      node_to_rc(n, d, row, col);
      const double v = sC[warp][col][row];
      const double e = escale[(long)n + ebase + (long)d * Ne * NP];
      if (d == 0) acc[m2] = e * v;
      else        acc[m2] += e * v;
    }
    __syncwarp();  // sB/sC are reused by the next direction
  }

  // surface lift + combine + write
  for (int m2 = 0; m2 < 16; ++m2) {
    const int n = lane + 32 * m2;
    const int i = n & 7, jk = n >> 3, j = jk & 7, k = jk >> 3;
    const long fb = (long)ke * NFT;
    const double lift =
        Lift[n           ] * ebnd[fb +       i + 8 * k]
      + Lift[n +      NP ] * ebnd[fb +  64 + j + 8 * k]
      + Lift[n + 2L * NP ] * ebnd[fb + 128 + i + 8 * k]
      + Lift[n + 3L * NP ] * ebnd[fb + 192 + j + 8 * k]
      + Lift[n + 4L * NP ] * ebnd[fb + 256 + i + 8 * j]
      + Lift[n + 5L * NP ] * ebnd[fb + 320 + i + 8 * j];
    dqdt[ebase + n] = -(acc[m2] + lift);
  }
}

extern "C" int cal_eop_p7_launch(double *dqdt, const double *fx,
                                 const double *fy, const double *fz,
                                 const double *ebnd, const double *D1D,
                                 const double *Lift, const double *escale,
                                 int Ne, int NeA, int use_tc) {
  const int nblk = (Ne + EPB - 1) / EPB;
  cal_eop_p7_kernel<<<nblk, NT>>>(dqdt, fx, fy, fz, ebnd, D1D, Lift, escale,
                                  Ne, NeA, use_tc);
  return (int)cudaGetLastError();
}
