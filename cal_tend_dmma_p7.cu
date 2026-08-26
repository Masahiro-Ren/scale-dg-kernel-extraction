//------------------------------------------------------------------------------
// Fused tendency kernel for p=7 with the tensor contractions executed on
// FP64 tensor cores via inline-PTX DMMA (mma.m8n8k4), following
// Tu et al., "Accelerating High-Order Finite Element Simulations at
// Extreme Scale with FP64 Tensor Cores" (IEEE, 2026):
//   - direct PTX DMMA with explicit lane<->element maps (paper Tables III-V)
//   - cyclic index reordering so the summed index is always the
//     fastest-changing index in shared memory (paper Sec. III-D): all three
//     contractions become the same canonical GEMM  C(8x64) = D1D(8x8) * B(8x64)
//     on transposed input/output layouts
//   - leading-dimension padding chosen for bank-conflict-free fragment
//     loads (ld=12 for inputs) and stores (ld=10 for outputs)
//   - A fragments (D1D) are loaded once per warp and reused for all tiles
//
// One 128-thread block per element (4 warps); each warp owns 2 of the 8
// column tiles per direction. Called from CUDA Fortran via the extern "C"
// launcher; array pointers are OpenACC device pointers (host_data).
// Fortran arrays are column-major; all indexing here is explicit and 0-based.
//------------------------------------------------------------------------------
#include <cuda_runtime.h>

#define NQ  8
#define NP  512   // NQ^3
#define NFP 64    // NQ^2
#define NFT 384   // 6*NFP
#define NT  128   // threads per block

// Padded leading dimensions (8-byte words). 16 lanes per access phase must
// hit 16 distinct (addr mod 16) double-banks:
//  - B loads  (lane -> [row + 4*colphase]): ld 12 -> conflict-free
//  - C stores (lane -> [row][2*(lane&3)+{0,1}]): ld 10 -> conflict-free
#define LDB 12
#define LDC 10

__device__ __forceinline__ void dmma_m8n8k4(double &c0, double &c1,
                                            const double a, const double b) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
      "{%0,%1}, {%2}, {%3}, {%0,%1};"
      : "+d"(c0), "+d"(c1)
      : "d"(a), "d"(b));
}

extern "C" __global__ void __launch_bounds__(NT)
cal_tend_dmma_p7_kernel(double *__restrict__ dqdt,
                        const double *__restrict__ q,
                        const double *__restrict__ u,
                        const double *__restrict__ v,
                        const double *__restrict__ w,
                        const double *__restrict__ D1D,      // (8,8) col-major
                        const double *__restrict__ Lift,     // (8,8,8,6)
                        const int *__restrict__ vmapM,       // (384,Ne)
                        const int *__restrict__ vmapP,       // (384,Ne)
                        const double *__restrict__ normal,   // (384,Ne,3)
                        const double *__restrict__ escale,   // (512,Ne,3)
                        const double *__restrict__ fscale,   // (384,Ne)
                        const int Ne, const int NeA) {
  __shared__ double sB[3][NFP][LDB];  // contraction inputs, cyclic layouts
  __shared__ double sC[3][NFP][LDC];  // contraction outputs
  __shared__ double sfe[6][NQ][NQ];   // face flux (fp2, fp1) per face
  __shared__ double sD[NQ * NQ];      // D1D staged verbatim (col-major)

  const int ke = blockIdx.x;          // 0-based element id
  const int t = threadIdx.x;
  const long nbase = (long)ke * NP;

  if (t < NQ * NQ) sD[t] = D1D[t];

  // Volume flux components in cyclic layouts:
  //   x: B0[col=jk][row=i]      (summed index i fastest)
  //   y: B1[col=i+8k][row=j]    (summed index j fastest)
  //   z: B2[col=i+8j][row=k]    (summed index k fastest)
  for (int n = t; n < NP; n += NT) {
    const int i = n & 7, jk = n >> 3, j = jk & 7, k = jk >> 3;
    const double qn = q[nbase + n];
    sB[0][jk][i] = qn * u[nbase + n];
    sB[1][i + 8 * k][j] = qn * v[nbase + n];
    sB[2][i + 8 * j][k] = qn * w[nbase + n];
  }

  // Element boundary flux (upwind), gathered through vmapM/vmapP
  for (int fp = t; fp < NFT; fp += NT) {
    const long iM = (long)vmapM[fp + (long)ke * NFT] - 1;  // Fortran 1-based
    const long iP = (long)vmapP[fp + (long)ke * NFT] - 1;

    const double qM = q[iM], qP = q[iP];
    const long gn = fp + (long)ke * NFT;
    const double nx = normal[gn], ny = normal[gn + (long)Ne * NFT],
                 nz = normal[gn + 2L * Ne * NFT];

    const double velM = u[iM] * nx + v[iM] * ny + w[iM] * nz;
    const double velP = u[iP] * nx + v[iP] * ny + w[iP] * nz;
    const double alpha = 0.5 * fabs(velP + velM);

    const int f = fp / NFP, r = fp & 63, a = r & 7, b = r >> 3;
    sfe[f][b][a] =
        0.5 * fscale[gn] * (qP * velP - qM * velM - alpha * (qP - qM));
  }
  __syncthreads();

  // Contractions on FP64 tensor cores: for each direction d,
  //   sC[d](row, col) = sum_l D1D(row, l) * sB[d](l, col),  8x64 output
  // m8n8k4 lane maps (paper Tables III-V), 0-based lanes:
  //   A: lane = 4*row + col   -> element D1D(rowA, colA + 4*kstep)
  //   B: lane = row + 4*col   -> element sB(rowB + 4*kstep, colB + tile*8)
  //   C: lane holds c0,c1 at (rowC, 2*(lane&3) + {0,1})
  {
    const int lane = t & 31, warp = t >> 5;
    const int rowA = lane >> 2, colA = lane & 3;
    const int rowB = lane & 3, colB = lane >> 2;
    const int rowC = lane >> 2, colC = 2 * (lane & 3);

    // A fragments: D1D(rowA, colA) col-major, reused for every tile
    const double a0 = sD[rowA + 8 * colA];
    const double a1 = sD[rowA + 8 * (colA + 4)];

    for (int d = 0; d < 3; ++d) {
      for (int nt = 2 * warp; nt < 2 * warp + 2; ++nt) {  // 2 tiles per warp
        double c0 = 0.0, c1 = 0.0;
        dmma_m8n8k4(c0, c1, a0, sB[d][8 * nt + colB][rowB]);
        dmma_m8n8k4(c0, c1, a1, sB[d][8 * nt + colB][rowB + 4]);
        sC[d][8 * nt + colC][rowC] = c0;
        sC[d][8 * nt + colC + 1][rowC] = c1;
      }
    }
  }
  __syncthreads();

  // Lift + combine
  for (int n = t; n < NP; n += NT) {
    const int i = n & 7, jk = n >> 3, j = jk & 7, k = jk >> 3;
    const long ln = i + 8L * j + 64L * k;

    const double lift = Lift[ln] * sfe[0][k][i]
                      + Lift[ln + NP] * sfe[1][k][j]
                      + Lift[ln + 2L * NP] * sfe[2][k][i]
                      + Lift[ln + 3L * NP] * sfe[3][k][j]
                      + Lift[ln + 4L * NP] * sfe[4][j][i]
                      + Lift[ln + 5L * NP] * sfe[5][j][i];

    const long gn = n + (long)ke * NP;
    dqdt[nbase + n] = -(escale[gn] * sC[0][jk][i]
                      + escale[gn + (long)Ne * NP] * sC[1][i + 8 * k][j]
                      + escale[gn + 2L * Ne * NP] * sC[2][i + 8 * j][k]
                      + lift);
  }
}

extern "C" int cal_tend_dmma_p7_launch(double *dqdt, const double *q,
                                       const double *u, const double *v,
                                       const double *w, const double *D1D,
                                       const double *Lift, const int *vmapM,
                                       const int *vmapP, const double *normal,
                                       const double *escale,
                                       const double *fscale, int Ne, int NeA) {
  cal_tend_dmma_p7_kernel<<<Ne, NT>>>(dqdt, q, u, v, w, D1D, Lift, vmapM,
                                      vmapP, normal, escale, fscale, Ne, NeA);
  return (int)cudaGetLastError();
}
