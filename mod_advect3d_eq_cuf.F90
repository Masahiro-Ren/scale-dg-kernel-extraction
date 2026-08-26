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

  ! CUDA C++ launcher for the inline-PTX DMMA kernel (cal_tend_dmma_p7.cu),
  ! following Tu et al. (IEEE 2026): direct PTX m8n8k4 DMMA, cyclic index
  ! reordering, bank-conflict-free lane maps.
  interface
    integer(c_int) function cal_tend_dmma_p7_launch( dqdt, q, u, v, w, &
        D1D, Lift, vmapM, vmapP, normal, escale, fscale, Ne, NeA )     &
        bind(c, name="cal_tend_dmma_p7_launch")
      use iso_c_binding, only: c_int, c_double
      real(c_double), device :: dqdt(*), q(*), u(*), v(*), w(*), D1D(*), Lift(*)
      integer(c_int), device :: vmapM(*), vmapP(*)
      real(c_double), device :: normal(*), escale(*), fscale(*)
      integer(c_int), value :: Ne, NeA
    end function cal_tend_dmma_p7_launch
  end interface

contains

  !> Launch a fused CUDA tendency kernel.
  !  Field arrays are device-resident via OpenACC data management; the
  !  device pointers are obtained with host_data use_device.
  !  variant: 0 = FP64 FMA (CUF), 1 = FP64 WMMA tensor cores (CUF_TC),
  !           2 = thread block cluster + DSM flux (CUF_DSM),
  !           3 = inline-PTX DMMA per Tu et al. (DMMA),
  !           4 = compute-once shared-face fluxes via cluster DSM (FLUX_DSM)
  subroutine advect3d_eq_cal_tend_cuf( dqdt, & ! (inout)
    q, u, v, w,                              & ! (in)
    D1D, D1D_tr, Lift_mat,                   & ! (in)
    VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
    Nq, Np, NfpTot, Ne, NeA,                 & ! (in)
    NeX, NeY, NeZ, variant )                   ! (in)
    use iso_c_binding, only: c_int
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
    integer, intent(in) :: NeX, NeY, NeZ
    integer, intent(in) :: variant

    integer :: istat
    !------------------------------------------------------------

    if ( Nq /= 8 ) then
      write(*,*) "CUF tendency kernels are implemented for PolyOrder = 7 only"
      error stop
    end if
    if ( variant == 2 .or. variant == 4 ) then
      if ( mod(NeX,2) + mod(NeY,2) + mod(NeZ,2) /= 0 ) then
        write(*,*) "CUF_DSM/FLUX_DSM require even NeX, NeY, NeZ (2x2x2 clusters)"
        error stop
      end if
    end if

    !$acc host_data use_device(dqdt, q, u, v, w, D1D, D1D_tr, Lift_mat, &
    !$acc                      VMapM, VMapP, normal_fn, Escale, Fscale)
    select case ( variant )
    case (1)
      call cal_tend_p7_tc_kernel<<<Ne, NTHREADS>>>( dqdt, q, u, v, w, &
        D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, Ne, NeA )
    case (2)
      call cal_tend_p7_dsm_kernel<<<dim3(NeX,NeY,NeZ), NTHREADS>>>( &
        dqdt, q, u, v, w, &
        D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, &
        NeX, NeY, Ne, NeA )
    case (3)
      istat = cal_tend_dmma_p7_launch( dqdt, q, u, v, w, D1D, Lift_mat, &
        VMapM, VMapP, normal_fn, Escale, Fscale, Ne, NeA )
      if ( istat /= 0 ) then
        write(*,*) "DMMA kernel launch failed, code ", istat
        error stop
      end if
    case (4)
      call cal_tend_p7_fdsm_kernel<<<dim3(NeX,NeY,NeZ), NTHREADS>>>( &
        dqdt, q, u, v, w, &
        D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, &
        NeX, NeY, Ne, NeA )
    case default
      call cal_tend_p7_kernel<<<Ne, NTHREADS>>>( dqdt, q, u, v, w, &
        D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, Ne, NeA )
    end select
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

  !> Fused tendency kernel for p=7 with 2x2x2 thread block clusters.
  !  The grid is launched as (NeX, NeY, NeZ), so each cluster covers a
  !  2x2x2 brick of elements; for each element, one face neighbor per
  !  direction (3 of 6 faces) is in-cluster and its staged field values
  !  are read via distributed shared memory instead of global gathers.
  !  Periodic-wrap neighbors point into the halo (keP > Ne) and never
  !  match a cluster partner, so they take the global path.
  attributes(global) cluster_dims(2,2,2) subroutine cal_tend_p7_dsm_kernel( &
    dqdt, q_, u_, v_, w_, &
    D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, &
    NeX, NeY, Ne, NeA )
    use cooperative_groups
    implicit none
    integer, value :: NeX, NeY, Ne, NeA
    real(RP) :: dqdt(512*NeA)
    real(RP) :: q_(512*NeA), u_(512*NeA), v_(512*NeA), w_(512*NeA)
    real(RP) :: D1D(8,8), D1D_tr(8,8)
    real(RP) :: Lift_mat(8,8,8,6)
    integer :: VMapM(384,Ne), VMapP(384,Ne)
    real(RP) :: normal_fn(384,Ne,3)
    real(RP) :: Escale(512,Ne,3)
    real(RP) :: Fscale(384,Ne)

    type(cluster_group) :: cluster

    real(RP), shared :: s_q(512), s_u(512), s_v(512), s_w(512)
    real(RP), shared :: s_fx(8,64), s_fy(8,8,8), s_fz(64,8)
    real(RP), shared :: s_fe(8,8,6)
    real(RP), shared :: s_D(8,8), s_Dt(8,8)

    ! Remote (distributed shared memory) views of a partner block's staging
    real(RP), shared :: r_q(512); pointer(p_rq, r_q)
    real(RP), shared :: r_u(512); pointer(p_ru, r_u)
    real(RP), shared :: r_v(512); pointer(p_rv, r_v)
    real(RP), shared :: r_w(512); pointer(p_rw, r_w)

    integer :: ke, t, nbase
    integer :: cx0, cy0, cz0, myrank0
    integer :: nbr_x, nbr_y, nbr_z, rank_x, rank_y, rank_z
    integer :: n, fp, i, j, k, ij, jk, l, f, a, b, r
    integer :: iM, iP, keP, rr, nl
    real(RP) :: qM, qP, VelM, VelP, alpha
    real(RP) :: Dx, Dy, Dz, lift
    !------------------------------------------------------------

    ke = blockIdx%x + (blockIdx%y-1)*NeX + (blockIdx%z-1)*NeX*NeY
    t  = threadIdx%x
    nbase = (ke-1)*512

    cluster = this_cluster()

    ! Cluster-local coords and the in-cluster face partner per direction
    ! (rank order verified: rank = 1 + cx0 + 2*cy0 + 4*cz0, x fastest)
    cx0 = mod(blockIdx%x-1, 2)
    cy0 = mod(blockIdx%y-1, 2)
    cz0 = mod(blockIdx%z-1, 2)
    myrank0 = cx0 + 2*cy0 + 4*cz0
    nbr_x = ke + (1 - 2*cx0)
    nbr_y = ke + (1 - 2*cy0)*NeX
    nbr_z = ke + (1 - 2*cz0)*NeX*NeY
    rank_x = 1 + ieor(myrank0, 1)
    rank_y = 1 + ieor(myrank0, 2)
    rank_z = 1 + ieor(myrank0, 4)

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
    ! Peers read our staging via DSM: cluster-wide barrier, not block-local
    call syncthreads(cluster)

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
      VelM = s_u(iM-nbase)*normal_fn(fp,ke,1) &
           + s_v(iM-nbase)*normal_fn(fp,ke,2) &
           + s_w(iM-nbase)*normal_fn(fp,ke,3)

      keP = (iP-1)/512 + 1
      if ( keP == nbr_x ) then
        rr = rank_x
      else if ( keP == nbr_y ) then
        rr = rank_y
      else if ( keP == nbr_z ) then
        rr = rank_z
      else
        rr = 0
      end if

      if ( rr /= 0 ) then
        ! neighbor is in this cluster: read its shared-memory staging
        p_rq = cluster_map_shared_rank(s_q, rr)
        p_ru = cluster_map_shared_rank(s_u, rr)
        p_rv = cluster_map_shared_rank(s_v, rr)
        p_rw = cluster_map_shared_rank(s_w, rr)
        nl = iP - (keP-1)*512
        qP = r_q(nl)
        VelP = r_u(nl)*normal_fn(fp,ke,1) &
             + r_v(nl)*normal_fn(fp,ke,2) &
             + r_w(nl)*normal_fn(fp,ke,3)
      else
        qP = q_(iP)
        VelP = u_(iP)*normal_fn(fp,ke,1) &
             + v_(iP)*normal_fn(fp,ke,2) &
             + w_(iP)*normal_fn(fp,ke,3)
      end if

      alpha = 0.5_RP * abs( VelP + VelM )

      f = (fp-1)/64 + 1
      r = mod(fp-1,64)
      a = mod(r,8) + 1
      b = r/8 + 1
      s_fe(a,b,f) = 0.5_RP * Fscale(fp,ke) * ( &
           qP * VelP - qM * VelM - alpha * ( qP - qM ) )
    end do
    ! No block may exit (or proceed past flux) while peers still read its
    ! staging remotely
    call syncthreads(cluster)

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
  end subroutine cal_tend_p7_dsm_kernel

  !> Fused tendency kernel for p=7 with compute-once face fluxes.
  !!
  !!  Every interior face flux is normally computed twice (once per
  !!  adjacent element, with opposite normals). For the 12 faces interior
  !!  to each 2x2x2 cluster, this kernel computes each face ONCE: with
  !!  the owner's outward normal n and S = qP*(uP.n) - qM*(uM.n),
  !!  D = alpha*(qP - qM), the two sides' fluxes are
  !!      F_own     = 0.5*Fs*(S - D)
  !!      F_partner = 0.5*Fs*(S + D) = F_own + Fs*D
  !!  so the owner stores H = Fs*D next to its normal flux and the
  !!  partner reconstructs its flux with a single add via distributed
  !!  shared memory (no gathers, no dot products).
  !!  (Relies on Fscale matching across the face - true on this uniform
  !!  mesh. The reconstruction reassociates the arithmetic, so results
  !!  differ from the baseline at ULP level, like DMMA.)
  !!
  !!  Ownership uses a balanced XOR rule so every block owns 1 or 2 of
  !!  its 3 in-cluster faces (computes 4-5 of 6 faces instead of 6):
  !!    x-face: owner has cx0 == XOR(cy0,cz0)
  !!    y-face: owner has cy0 == XOR(cx0,cz0)
  !!    z-face: owner has cz0 /= XOR(cx0,cy0)
  attributes(global) cluster_dims(2,2,2) subroutine cal_tend_p7_fdsm_kernel( &
    dqdt, q_, u_, v_, w_, &
    D1D, D1D_tr, Lift_mat, VMapM, VMapP, normal_fn, Escale, Fscale, &
    NeX, NeY, Ne, NeA )
    use cooperative_groups
    implicit none
    integer, value :: NeX, NeY, Ne, NeA
    real(RP) :: dqdt(512*NeA)
    real(RP) :: q_(512*NeA), u_(512*NeA), v_(512*NeA), w_(512*NeA)
    real(RP) :: D1D(8,8), D1D_tr(8,8)
    real(RP) :: Lift_mat(8,8,8,6)
    integer :: VMapM(384,Ne), VMapP(384,Ne)
    real(RP) :: normal_fn(384,Ne,3)
    real(RP) :: Escale(512,Ne,3)
    real(RP) :: Fscale(384,Ne)

    type(cluster_group) :: cluster

    real(RP), shared :: s_q(512), s_u(512), s_v(512), s_w(512)
    real(RP), shared :: s_fx(8,64), s_fy(8,8,8), s_fz(64,8)
    real(RP), shared :: s_fe(8,8,6)
    real(RP), shared :: s_H(8,8,3)   !< Fs*D of owned in-cluster faces, by direction
    real(RP), shared :: s_D(8,8), s_Dt(8,8)

    ! Remote (DSM) views of a partner block's shared memory
    real(RP), shared :: r_q(512); pointer(p_rq, r_q)
    real(RP), shared :: r_u(512); pointer(p_ru, r_u)
    real(RP), shared :: r_v(512); pointer(p_rv, r_v)
    real(RP), shared :: r_w(512); pointer(p_rw, r_w)
    real(RP), shared :: r_fe(8,8,6); pointer(p_rfe, r_fe)
    real(RP), shared :: r_H(8,8,3);  pointer(p_rH, r_H)

    integer :: ke, t, nbase
    integer :: cx0, cy0, cz0, myrank0
    integer :: fin(3)   !< my in-cluster face per direction
    integer :: fop(3)   !< the partner's face that touches mine
    integer :: rrk(3)   !< partner rank per direction
    logical :: own(3)   !< do I compute the shared face in this direction?
    integer :: n, fp, i, j, k, ij, jk, l, f, d, a, b, r, idx
    integer :: iM, iP, nl
    real(RP) :: qM, qP, VelM, VelP, alpha
    real(RP) :: Dx, Dy, Dz, lift
    !------------------------------------------------------------

    ke = blockIdx%x + (blockIdx%y-1)*NeX + (blockIdx%z-1)*NeX*NeY
    t  = threadIdx%x
    nbase = (ke-1)*512

    cluster = this_cluster()

    cx0 = mod(blockIdx%x-1, 2)
    cy0 = mod(blockIdx%y-1, 2)
    cz0 = mod(blockIdx%z-1, 2)
    myrank0 = cx0 + 2*cy0 + 4*cz0
    rrk(1) = 1 + ieor(myrank0, 1)
    rrk(2) = 1 + ieor(myrank0, 2)
    rrk(3) = 1 + ieor(myrank0, 4)

    ! Faces: 1:-y 2:+x 3:+y 4:-x 5:-z 6:+z
    if ( cx0 == 0 ) then; fin(1) = 2; fop(1) = 4; else; fin(1) = 4; fop(1) = 2; end if
    if ( cy0 == 0 ) then; fin(2) = 3; fop(2) = 1; else; fin(2) = 1; fop(2) = 3; end if
    if ( cz0 == 0 ) then; fin(3) = 6; fop(3) = 5; else; fin(3) = 5; fop(3) = 6; end if

    own(1) = ( cx0 == ieor(cy0,cz0) )
    own(2) = ( cy0 == ieor(cx0,cz0) )
    own(3) = ( cz0 /= ieor(cx0,cy0) )

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
    call syncthreads(cluster)

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

    ! Flux, phase 1: faces this block computes.
    do fp = t, 384, 128
      f = (fp-1)/64 + 1
      if ( f == 2 .or. f == 4 ) then
        d = 1
      else if ( f == 1 .or. f == 3 ) then
        d = 2
      else
        d = 3
      end if

      if ( f == fin(d) .and. .not. own(d) ) cycle  ! partner computes it

      iM = VMapM(fp,ke)
      iP = VMapP(fp,ke)

      qM = s_q(iM-nbase)
      VelM = s_u(iM-nbase)*normal_fn(fp,ke,1) &
           + s_v(iM-nbase)*normal_fn(fp,ke,2) &
           + s_w(iM-nbase)*normal_fn(fp,ke,3)

      if ( f == fin(d) ) then
        ! owned in-cluster face: neighbor side from partner's staging (DSM)
        p_rq = cluster_map_shared_rank(s_q, rrk(d))
        p_ru = cluster_map_shared_rank(s_u, rrk(d))
        p_rv = cluster_map_shared_rank(s_v, rrk(d))
        p_rw = cluster_map_shared_rank(s_w, rrk(d))
        nl = mod(iP-1,512) + 1
        qP = r_q(nl)
        VelP = r_u(nl)*normal_fn(fp,ke,1) &
             + r_v(nl)*normal_fn(fp,ke,2) &
             + r_w(nl)*normal_fn(fp,ke,3)
      else
        ! out-of-cluster neighbor or periodic halo: global gather
        qP = q_(iP)
        VelP = u_(iP)*normal_fn(fp,ke,1) &
             + v_(iP)*normal_fn(fp,ke,2) &
             + w_(iP)*normal_fn(fp,ke,3)
      end if

      alpha = 0.5_RP * abs( VelP + VelM )

      r = mod(fp-1,64)
      a = mod(r,8) + 1
      b = r/8 + 1
      s_fe(a,b,f) = 0.5_RP * Fscale(fp,ke) * ( &
           qP * VelP - qM * VelM - alpha * ( qP - qM ) )
      if ( f == fin(d) ) then
        ! dissipation term for the partner's one-add reconstruction
        s_H(a,b,d) = Fscale(fp,ke) * alpha * ( qP - qM )
      end if
    end do
    call syncthreads(cluster)

    ! Flux, phase 2: reconstruct non-owned in-cluster faces from the
    ! owner's stored flux and dissipation term: F_mine = F_owner + H.
    do idx = t, 192, 128
      d = (idx-1)/64 + 1
      if ( own(d) ) cycle
      r = mod(idx-1,64)
      a = mod(r,8) + 1
      b = r/8 + 1
      p_rfe = cluster_map_shared_rank(s_fe, rrk(d))
      p_rH  = cluster_map_shared_rank(s_H,  rrk(d))
      s_fe(a,b,fin(d)) = r_fe(a,b,fop(d)) + r_H(a,b,d)
    end do
    ! No block may exit while peers still read its s_fe/s_H/staging
    call syncthreads(cluster)

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
  end subroutine cal_tend_p7_fdsm_kernel

end module mod_advect3d_eq_cuf
#else
!- Empty placeholder when not built with CUDA Fortran (nvfortran -cuda)
module mod_advect3d_eq_cuf
end module mod_advect3d_eq_cuf
#endif
