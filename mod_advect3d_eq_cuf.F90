!-------------------------------------------------------------------------------
!> module advect3d_eq_cuf
!!
!! @par Description
!!      CUDA Fortran implementation of the fused tendency kernel
!!      (element boundary flux + volume derivative + surface lift).
!!      Requires a CUDA Fortran build (nvfortran -cuda); compiled to an
!!      empty module otherwise. Kernels are specialized for PolyOrder = 7
!!      (Nq = 8, Np = 512, Nfp = 64, NfpTot = 384).
!!
!!      One thread block (128 threads) processes one element. The element's
!!      field values, volume fluxes, and face fluxes are staged in shared
!!      memory. Two kernel variants:
!!        - cal_tend_p7_kernel:    tensor contractions with FP64 FMA
!!        - cal_tend_p7_tc_kernel: tensor contractions with FP64 tensor
!!          cores (WMMA m8n8k4 DMMA), one warp per pair of 8x8x8-tiles
!!
!! @author Xuanzhengbo Ren, Team SCALE
!<
#ifdef _CUDA
module mod_advect3d_eq_cuf
  use mod_common, only: RP
  use cudafor
  implicit none
  private

  public :: advect3d_eq_cal_tend_cuf

  integer, parameter :: NTHREADS = 128

contains

  !> Launch the fused CUDA Fortran tendency kernel.
  !  Field arrays are device-resident via OpenACC data management; the
  !  device pointers are obtained with host_data use_device.
  subroutine advect3d_eq_cal_tend_cuf( dqdt, & ! (inout)
    q, u, v, w,                              & ! (in)
    D1D, D1D_tr, Lift_mat,                   & ! (in)
    VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
    Nq, Np, NfpTot, Ne, NeA, use_tc )          ! (in)
    implicit none
    integer, intent(in) :: Nq
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(inout) :: dqdt(Np,NeA)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA)
    real(RP), intent(in) :: v(Np,NeA)
    real(RP), intent(in) :: w(Np,NeA)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(in) :: Fscale(NfpTot,Ne)
    logical, intent(in) :: use_tc

    integer :: istat
    !------------------------------------------------------------

    if ( Nq /= 8 ) then
      write(*,*) "CUF tendency kernels are implemented for PolyOrder = 7 only"
      error stop
    end if

    !$acc host_data use_device(dqdt, q, u, v, w, D1D, D1D_tr, Lift_mat, &
    !$acc                      VMapM, VMapP, normal_fn, Escale, Fscale)
    if ( use_tc ) then
      call cal_tend_p7_tc_kernel<<<Ne, NTHREADS>>>( dqdt, q, u, v, w, &
        D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, Ne, NeA )
    else
      call cal_tend_p7_kernel<<<Ne, NTHREADS>>>( dqdt, q, u, v, w, &
        D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, Ne, NeA )
    end if
    !$acc end host_data

    ! CUF kernel launches are asynchronous; synchronize so host timers
    ! are valid and subsequent OpenACC kernels are ordered correctly.
    istat = cudaDeviceSynchronize()
    if ( istat /= 0 ) then
      write(*,*) "CUF tendency kernel failed: ", cudaGetErrorString(istat)
      error stop
    end if
    return
  end subroutine advect3d_eq_cal_tend_cuf

  !> Fused tendency kernel for p=7, FP64 FMA contractions.
  !  Block = one element, 128 threads (4 nodes/thread).
  attributes(global) subroutine cal_tend_p7_kernel( dqdt, q_, u_, v_, w_, &
    D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, Ne, NeA )
    implicit none
    integer, value :: Ne, NeA
    real(RP) :: dqdt(512*NeA)
    real(RP) :: q_(512*NeA), u_(512*NeA), v_(512*NeA), w_(512*NeA)
    real(RP) :: D1D(8,8), D1D_tr(8,8)
    real(RP) :: Lift_mat(8,8,8,6)
    integer :: VMapM(384,Ne), VMapP(384,Ne)
    real(RP) :: normal_fn(384,Ne,3)
    real(RP) :: Escale(512,Ne,3)
    real(RP) :: Fscale(384,Ne)

    real(RP), shared :: s_q(512), s_u(512), s_v(512), s_w(512)
    real(RP), shared :: s_fx(8,64)   ! flux_x as (i, jk)
    real(RP), shared :: s_fy(8,8,8)  ! flux_y as (i, j, k)
    real(RP), shared :: s_fz(64,8)   ! flux_z as (ij, k)
    real(RP), shared :: s_fe(8,8,6)  ! face flux as (fp1, fp2, face)
    real(RP), shared :: s_D(8,8), s_Dt(8,8)

    integer :: ke, t, nbase
    integer :: n, fp, i, j, k, ij, jk, l, f, a, b, r
    integer :: iM, iP
    real(RP) :: qM, qP, VelM, VelP, alpha
    real(RP) :: Dx, Dy, Dz, lift
    !------------------------------------------------------------

    ke = blockIdx%x
    t  = threadIdx%x
    nbase = (ke-1)*512

    if ( t <= 64 ) then
      i = mod(t-1,8) + 1
      j = (t-1)/8 + 1
      s_D(i,j)  = D1D(i,j)
      s_Dt(i,j) = D1D_tr(i,j)
    end if

    ! Stage the element's field values
    do n = t, 512, 128
      s_q(n) = q_(nbase+n)
      s_u(n) = u_(nbase+n)
      s_v(n) = v_(nbase+n)
      s_w(n) = w_(nbase+n)
    end do
    call syncthreads()

    ! Volume flux components
    do n = t, 512, 128
      i  = mod(n-1,8) + 1
      jk = (n-1)/8 + 1
      j  = mod(jk-1,8) + 1
      k  = (jk-1)/8 + 1
      ij = i + (j-1)*8
      s_fx(i,jk)  = s_q(n)*s_u(n)
      s_fy(i,j,k) = s_q(n)*s_v(n)
      s_fz(ij,k)  = s_q(n)*s_w(n)
    end do

    ! Element boundary flux (own side from shared, neighbor side gathered)
    do fp = t, 384, 128
      iM = VMapM(fp,ke)
      iP = VMapP(fp,ke)

      qM = s_q(iM-nbase)
      qP = q_(iP)

      VelM = s_u(iM-nbase)*normal_fn(fp,ke,1) &
           + s_v(iM-nbase)*normal_fn(fp,ke,2) &
           + s_w(iM-nbase)*normal_fn(fp,ke,3)
      VelP = u_(iP)*normal_fn(fp,ke,1) &
           + v_(iP)*normal_fn(fp,ke,2) &
           + w_(iP)*normal_fn(fp,ke,3)

      alpha = 0.5_RP * abs( VelP + VelM )

      f = (fp-1)/64 + 1
      r = mod(fp-1,64)
      a = mod(r,8) + 1
      b = r/8 + 1
      s_fe(a,b,f) = 0.5_RP * Fscale(fp,ke) * ( &
           qP * VelP - qM * VelM - alpha * ( qP - qM ) )
    end do
    call syncthreads()

    ! Sum-factorized contraction + lift + combine
    do n = t, 512, 128
      i  = mod(n-1,8) + 1
      jk = (n-1)/8 + 1
      j  = mod(jk-1,8) + 1
      k  = (jk-1)/8 + 1
      ij = i + (j-1)*8

      Dx = 0.0_RP; Dy = 0.0_RP; Dz = 0.0_RP
      do l = 1, 8
        Dx = Dx + s_D(i,l)    * s_fx(l,jk)
        Dy = Dy + s_fy(i,l,k) * s_Dt(l,j)
        Dz = Dz + s_fz(ij,l)  * s_Dt(l,k)
      end do

      lift = Lift_mat(i,j,k,1)*s_fe(i,k,1) &
           + Lift_mat(i,j,k,2)*s_fe(j,k,2) &
           + Lift_mat(i,j,k,3)*s_fe(i,k,3) &
           + Lift_mat(i,j,k,4)*s_fe(j,k,4) &
           + Lift_mat(i,j,k,5)*s_fe(i,j,5) &
           + Lift_mat(i,j,k,6)*s_fe(i,j,6)

      dqdt(nbase+n) = -( Escale(n,ke,1)*Dx &
                       + Escale(n,ke,2)*Dy &
                       + Escale(n,ke,3)*Dz + lift )
    end do
    return
  end subroutine cal_tend_p7_kernel

  !> Fused tendency kernel for p=7 with the three tensor contractions
  !  executed on FP64 tensor cores (WMMA m8n8k4). Each contraction is a
  !  batch of 8x8 GEMM tiles with K=8 (two accumulating k4 steps); the
  !  4 warps of the block each own 2 of the 8 tiles per direction.
  attributes(global) subroutine cal_tend_p7_tc_kernel( dqdt, q_, u_, v_, w_, &
    D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, Ne, NeA )
    use wmma
    implicit none
    integer, value :: Ne, NeA
    real(RP) :: dqdt(512*NeA)
    real(RP) :: q_(512*NeA), u_(512*NeA), v_(512*NeA), w_(512*NeA)
    real(RP) :: D1D(8,8), D1D_tr(8,8)
    real(RP) :: Lift_mat(8,8,8,6)
    integer :: VMapM(384,Ne), VMapP(384,Ne)
    real(RP) :: normal_fn(384,Ne,3)
    real(RP) :: Escale(512,Ne,3)
    real(RP) :: Fscale(384,Ne)

    real(RP), shared :: s_q(512), s_u(512), s_v(512), s_w(512)
    real(RP), shared :: s_fx(8,64), s_fy(8,8,8), s_fz(64,8)
    real(RP), shared :: s_Dx(8,64), s_Dy(8,8,8), s_Dz(64,8)
    real(RP), shared :: s_fe(8,8,6)
    real(RP), shared :: s_D(8,8), s_Dt(8,8)

    ! WMMA fragment types for FP64 m8n8k4
    ! (= WMMASubMatrix(WMMAMatrix{A,B}, 8,8,4, Real, WMMAColMajorKind8)
    !  and WMMASubMatrix(WMMAMatrixC, 8,8,4, Real, WMMAKind8))
    type(subMatrixA_m8n8k4_Real8_Cmajor) :: saa
    type(subMatrixB_m8n8k4_Real8_Cmajor) :: sbb
    type(subMatrixC_m8n8k4_Real8)        :: scc

    integer :: ke, t, nbase, warp, mt
    integer :: n, fp, i, j, k, ij, jk, f, a, b, r
    integer :: iM, iP
    real(RP) :: qM, qP, VelM, VelP, alpha
    real(RP) :: lift
    !------------------------------------------------------------

    ke = blockIdx%x
    t  = threadIdx%x
    nbase = (ke-1)*512

    if ( t <= 64 ) then
      i = mod(t-1,8) + 1
      j = (t-1)/8 + 1
      s_D(i,j)  = D1D(i,j)
      s_Dt(i,j) = D1D_tr(i,j)
    end if

    do n = t, 512, 128
      s_q(n) = q_(nbase+n)
      s_u(n) = u_(nbase+n)
      s_v(n) = v_(nbase+n)
      s_w(n) = w_(nbase+n)
    end do
    call syncthreads()

    do n = t, 512, 128
      i  = mod(n-1,8) + 1
      jk = (n-1)/8 + 1
      j  = mod(jk-1,8) + 1
      k  = (jk-1)/8 + 1
      ij = i + (j-1)*8
      s_fx(i,jk)  = s_q(n)*s_u(n)
      s_fy(i,j,k) = s_q(n)*s_v(n)
      s_fz(ij,k)  = s_q(n)*s_w(n)
    end do

    do fp = t, 384, 128
      iM = VMapM(fp,ke)
      iP = VMapP(fp,ke)

      qM = s_q(iM-nbase)
      qP = q_(iP)

      VelM = s_u(iM-nbase)*normal_fn(fp,ke,1) &
           + s_v(iM-nbase)*normal_fn(fp,ke,2) &
           + s_w(iM-nbase)*normal_fn(fp,ke,3)
      VelP = u_(iP)*normal_fn(fp,ke,1) &
           + v_(iP)*normal_fn(fp,ke,2) &
           + w_(iP)*normal_fn(fp,ke,3)

      alpha = 0.5_RP * abs( VelP + VelM )

      f = (fp-1)/64 + 1
      r = mod(fp-1,64)
      a = mod(r,8) + 1
      b = r/8 + 1
      s_fe(a,b,f) = 0.5_RP * Fscale(fp,ke) * ( &
           qP * VelP - qM * VelM - alpha * ( qP - qM ) )
    end do
    call syncthreads()

    ! Tensor-core contractions: warp w handles tiles/slices 2w+1, 2w+2
    warp = (t-1)/32
    do mt = 2*warp+1, 2*warp+2
      ! x direction: Dx(:, cols) = D1D x fx(:, cols), 8 columns per tile
      scc = 0.0_RP
      call wmmaLoadMatrix(saa, s_D(1,1), 8)
      call wmmaLoadMatrix(sbb, s_fx(1,8*(mt-1)+1), 8)
      call wmmaMatmul(scc, saa, sbb, scc)
      call wmmaLoadMatrix(saa, s_D(1,5), 8)
      call wmmaLoadMatrix(sbb, s_fx(5,8*(mt-1)+1), 8)
      call wmmaMatmul(scc, saa, sbb, scc)
      call wmmaStoreMatrix(s_Dx(1,8*(mt-1)+1), scc, 8)

      ! y direction: Dy(:,:,k) = fy(:,:,k) x D1D_tr, one k-slice per tile
      scc = 0.0_RP
      call wmmaLoadMatrix(saa, s_fy(1,1,mt), 8)
      call wmmaLoadMatrix(sbb, s_Dt(1,1), 8)
      call wmmaMatmul(scc, saa, sbb, scc)
      call wmmaLoadMatrix(saa, s_fy(1,5,mt), 8)
      call wmmaLoadMatrix(sbb, s_Dt(5,1), 8)
      call wmmaMatmul(scc, saa, sbb, scc)
      call wmmaStoreMatrix(s_Dy(1,1,mt), scc, 8)

      ! z direction: Dz(rows,:) = fz(rows,:) x D1D_tr, 8 rows per tile
      scc = 0.0_RP
      call wmmaLoadMatrix(saa, s_fz(8*(mt-1)+1,1), 64)
      call wmmaLoadMatrix(sbb, s_Dt(1,1), 8)
      call wmmaMatmul(scc, saa, sbb, scc)
      call wmmaLoadMatrix(saa, s_fz(8*(mt-1)+1,5), 64)
      call wmmaLoadMatrix(sbb, s_Dt(5,1), 8)
      call wmmaMatmul(scc, saa, sbb, scc)
      call wmmaStoreMatrix(s_Dz(8*(mt-1)+1,1), scc, 64)
    end do
    call syncthreads()

    ! Lift + combine
    do n = t, 512, 128
      i  = mod(n-1,8) + 1
      jk = (n-1)/8 + 1
      j  = mod(jk-1,8) + 1
      k  = (jk-1)/8 + 1
      ij = i + (j-1)*8

      lift = Lift_mat(i,j,k,1)*s_fe(i,k,1) &
           + Lift_mat(i,j,k,2)*s_fe(j,k,2) &
           + Lift_mat(i,j,k,3)*s_fe(i,k,3) &
           + Lift_mat(i,j,k,4)*s_fe(j,k,4) &
           + Lift_mat(i,j,k,5)*s_fe(i,j,5) &
           + Lift_mat(i,j,k,6)*s_fe(i,j,6)

      dqdt(nbase+n) = -( Escale(n,ke,1)*s_Dx(i,jk) &
                       + Escale(n,ke,2)*s_Dy(i,j,k) &
                       + Escale(n,ke,3)*s_Dz(ij,k) + lift )
    end do
    return
  end subroutine cal_tend_p7_tc_kernel

end module mod_advect3d_eq_cuf
#else
!- Empty placeholder when not built with CUDA Fortran (nvfortran -cuda)
module mod_advect3d_eq_cuf
end module mod_advect3d_eq_cuf
#endif
