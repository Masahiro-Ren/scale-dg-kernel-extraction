!-------------------------------------------------------------------------------
!> module mesh
!!
!! @par Description
!!      A module to calculate the tendency 3D advection equation
!!
!! @author Yuta Kawai, Xuanzhengbo Ren, Team SCALE
!<
module mod_advect3d_eq
  use mod_common, only: RP, &
  Timer, Timer_start, Timer_stop, Timer_elapsed
  implicit none
  private

  public :: setup_advect3d_eq_setup
  public :: setup_advect3d_eq_finalize
  public :: advect3d_eq_cal_tend

  type(Timer) :: timer_ebnd_flux
  type(Timer) :: timer_dqdt
  type(Timer) :: timer_fused
  type(Timer) :: timer_cuf
  type(Timer) :: timer_vflux  !< LAYERED: volume-flux (physics) kernel
  type(Timer) :: timer_eop    !< LAYERED: element-operations CUDA kernel

  !> Tendency kernel type: separate flux / volume+lift kernels (original)
  integer, parameter :: TEND_KERNEL_TYPEID_SPLIT = 1
  !> Tendency kernel type: single element loop computing the boundary flux
  !! and the volume derivative + surface lift per element
  integer, parameter :: TEND_KERNEL_TYPEID_FUSED = 2
  !> Tendency kernel type: fused CUDA Fortran kernel (p=7 only, needs -cuda)
  integer, parameter :: TEND_KERNEL_TYPEID_CUF = 3
  !> Same as CUF with the tensor contractions on FP64 tensor cores
  integer, parameter :: TEND_KERNEL_TYPEID_CUF_TC = 4
  !> Same as CUF with 2x2x2 thread block clusters: in-cluster flux
  !! neighbors are read via distributed shared memory
  integer, parameter :: TEND_KERNEL_TYPEID_CUF_DSM = 5
  !> Fused CUDA C++ kernel with inline-PTX FP64 DMMA contractions
  !! (Tu et al., IEEE 2026)
  integer, parameter :: TEND_KERNEL_TYPEID_DMMA = 6
  !> Register-blocked columns: 64 threads/element, z-column per thread
  !! in registers, x/y contractions via 8x8 shared slices
  integer, parameter :: TEND_KERNEL_TYPEID_CUF_COL = 7
  !> One warp per element: shuffle-based x/y contractions, register z
  integer, parameter :: TEND_KERNEL_TYPEID_CUF_WARP = 8
  !> LAYERED structure (per the SCALE-DG developer's proposed split):
  !! boundary flux = OpenACC kernel, volume flux (physics) = OpenACC
  !! kernel writing global flux arrays, element operations
  !! (contraction + lift + combine) = batched CUDA kernel behind the
  !! library-style interface. LAYER_TC runs the contractions on FP64
  !! tensor cores (DMMA).
  integer, parameter :: TEND_KERNEL_TYPEID_LAYERED  = 9
  integer, parameter :: TEND_KERNEL_TYPEID_LAYER_TC = 10

  integer :: mesh_NeX, mesh_NeY, mesh_NeZ  !< mesh dims (for cluster launch)

  integer :: tend_kernel_typeid = TEND_KERNEL_TYPEID_SPLIT

  ! Work array for the element boundary flux (SPLIT and LAYERED types).
  ! Allocated on the heap (not as an automatic array) so that large meshes
  ! do not overflow the stack, and kept resident on the device.
  real(RP), allocatable :: ebnd_flux(:,:)

  ! Volume-flux arrays crossing the LAYERED physics/element-operations
  ! interface (the price of the layering: a global-memory round trip).
  real(RP), allocatable :: vflux_x(:,:), vflux_y(:,:), vflux_z(:,:)
  logical :: eop_use_tc = .false.  !< LAYER_TC: tensor-core contraction

  ! The volume-derivative + lift implementation is swappable behind one
  ! call site (SPLIT structure stays unchanged): cal_dqdt (original
  ! OpenACC) or cal_dqdt_cuda (LAYERED: physics kernel + CUDA element
  ! operations). Selected once at setup - no branch in the time loop.
  abstract interface
    subroutine dqdt_kernel_iface( dqdt, q, u, v, w, flux_bnd, &
      D1D, D1D_tr, Lift_mat, Escale, Nq, Np, NfpTot, Ne, NeA )
      import RP
      integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
      real(RP), intent(out) :: dqdt(Np,NeA)
      real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
      real(RP), intent(in) :: flux_bnd(NfpTot,Ne)
      real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
      real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
      real(RP), intent(in) :: Escale(Np,Ne,3)
    end subroutine dqdt_kernel_iface
  end interface
  procedure(dqdt_kernel_iface), pointer :: cal_dqdt_ptr => null()
contains
  !> Setup
!OCL SERIAL
  subroutine setup_advect3d_eq_setup( tend_kernel_type, NeX, NeY, NeZ )
    implicit none
    character(len=*), intent(in) :: tend_kernel_type
    integer, intent(in) :: NeX, NeY, NeZ
    !------------------------------------------------------------------------------
    mesh_NeX = NeX; mesh_NeY = NeY; mesh_NeZ = NeZ
    cal_dqdt_ptr => cal_dqdt
    select case( trim(tend_kernel_type) )
    case ("SPLIT")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_SPLIT
    case ("FUSED")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_FUSED
    case ("CUF")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_CUF
    case ("CUF_TC")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_CUF_TC
    case ("CUF_DSM")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_CUF_DSM
    case ("DMMA")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_DMMA
    case ("CUF_COL")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_CUF_COL
    case ("CUF_WARP")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_CUF_WARP
    case ("LAYERED", "LAYER_TC")
#ifdef _CUDA
      if ( trim(tend_kernel_type) == "LAYER_TC" ) then
        tend_kernel_typeid = TEND_KERNEL_TYPEID_LAYER_TC
        eop_use_tc = .true.
      else
        tend_kernel_typeid = TEND_KERNEL_TYPEID_LAYERED
      end if
      cal_dqdt_ptr => cal_dqdt_cuda
#else
      write(*,*) "TendencyKernel_Type LAYERED/LAYER_TC requires a CUDA build (nvfortran -cuda)"
      error stop
#endif
    case default
      write(*,*) "Unsupported tend_kernel_type: ", tend_kernel_type
    end select
    return
  end subroutine setup_advect3d_eq_setup
  !> Finalize
!OCL SERIAL
  subroutine setup_advect3d_eq_finalize()
    implicit none
    !------------------------------------------------------------------------------
    if ( tend_kernel_typeid >= TEND_KERNEL_TYPEID_CUF .and. &
         tend_kernel_typeid <= TEND_KERNEL_TYPEID_CUF_WARP ) then
      write(*,'(A30,ES24.5)') "CUF fused tendency:", Timer_elapsed(timer_cuf)
    else if ( tend_kernel_typeid == TEND_KERNEL_TYPEID_FUSED ) then
      ! Both phases run in one kernel; only the combined time is
      ! measurable from the host.
      write(*,'(A30,ES24.5)') "Fused flux+volume+lift:", Timer_elapsed(timer_fused)
    else
      write(*,'(A30,ES24.5)') "Element boundary flux:", Timer_elapsed(timer_ebnd_flux)
      write(*,'(A30,ES24.5)') "Volume derivate + surface lift:", Timer_elapsed(timer_dqdt)
      if ( tend_kernel_typeid >= TEND_KERNEL_TYPEID_LAYERED ) then
        write(*,'(A30,ES24.5)') "  Volume flux (physics):", Timer_elapsed(timer_vflux)
        write(*,'(A30,ES24.5)') "  Element ops (CUDA):", Timer_elapsed(timer_eop)
      end if
    end if
    return
  end subroutine setup_advect3d_eq_finalize

  !> Calculate the tendency of 3D advection equation
!OCL SERIAL
  subroutine advect3d_eq_cal_tend( dqdt, & ! (out)
    q, u, v, w,                              & ! (in)
    D1D, D1D_tr, Lift_mat,                   & ! (in)
    VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
    Nq, Np, NfpTot, Ne, NeA )                  ! (in)

#ifdef _CUDA
    use mod_advect3d_eq_cuf, only: advect3d_eq_cal_tend_cuf
#endif
     implicit none
    integer, intent(in) :: Nq
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(out) :: dqdt(Np,NeA)
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

    integer :: ke

    ! Per-element work arrays for the FUSED kernel type
    ! (thread-private on CPU, gang-private on GPU)
    real(RP) :: flux_e(NfpTot)
    real(RP) :: flux_x(Np), flux_y(Np), flux_z(Np)
    real(RP) :: DxFlux(Np), DyFlux(Np), DzFlux(Np)
    real(RP) :: LiftBndFlux(Np)
    !------------------------------------------------------------

    if ( tend_kernel_typeid >= TEND_KERNEL_TYPEID_CUF .and. &
         tend_kernel_typeid <= TEND_KERNEL_TYPEID_CUF_WARP ) then

#ifdef _CUDA
      call Timer_start(timer_cuf)
      call advect3d_eq_cal_tend_cuf( dqdt,    & ! (inout)
        q, u, v, w,                              & ! (in)
        D1D, D1D_tr, Lift_mat,                   & ! (in)
        VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
        Nq, Np, NfpTot, Ne, NeA,                 & ! (in)
        mesh_NeX, mesh_NeY, mesh_NeZ,            & ! (in)
        tend_kernel_typeid - TEND_KERNEL_TYPEID_CUF ) ! 0=CUF,1=TC,2=DSM,3=DMMA
      call Timer_stop(timer_cuf)
#else
      write(*,*) "TendencyKernel_Type CUF*/DMMA requires a CUDA Fortran build (nvfortran -cuda)"
      error stop
#endif

    else if ( tend_kernel_typeid == TEND_KERNEL_TYPEID_FUSED ) then

      call Timer_start(timer_fused)
      !$omp parallel do private( flux_e, flux_x, flux_y, flux_z, &
      !$omp                      DxFlux, DyFlux, DzFlux, LiftBndFlux )
      !$acc parallel loop gang default(present) &
      !$acc private( flux_e, flux_x, flux_y, flux_z, &
      !$acc          DxFlux, DyFlux, DzFlux, LiftBndFlux )
      do ke = 1, Ne
        call cal_elembnd_flux_elem( flux_e, & ! (out)
          ke, q, u, v, w,                   & ! (in)
          VMapM, VMapP, normal_fn, Fscale,  & ! (in)
          Np, NfpTot, Ne, NeA )
        call cal_dqdt_elem( dqdt,             & ! (inout)
          ke, q, u, v, w, flux_e,             & ! (in)
          D1D, D1D_tr, Lift_mat, Escale,      & ! (in)
          flux_x, flux_y, flux_z,             & ! (work)
          DxFlux, DyFlux, DzFlux, LiftBndFlux,& ! (work)
          Nq, Np, NfpTot, Ne, NeA )
      end do
      call Timer_stop(timer_fused)

    else ! SPLIT (original)

      if ( .not. allocated(ebnd_flux) ) then
        allocate( ebnd_flux(NfpTot,Ne) )
        !$acc enter data create(ebnd_flux)
      end if

      call Timer_start(timer_ebnd_flux)
      call cal_elembnd_flux( ebnd_flux,   & ! (out)
         q, u, v, w,                      & ! (in)
         VMapM, VMapP, normal_fn, Fscale, & ! (in)
         Np, NfpTot, Ne, NeA )
      call Timer_stop(timer_ebnd_flux)

      call Timer_start(timer_dqdt)
      call cal_dqdt_ptr( dqdt,           & ! (out)
         q, u, v, w,  ebnd_flux,         & ! (in)
         D1D, D1D_tr, Lift_mat,          & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)
      call Timer_stop(timer_dqdt)

    end if

     return
  end subroutine advect3d_eq_cal_tend

#ifdef _CUDA
  !> LAYERED implementation of the volume derivative + surface lifting:
  !! same interface and role as cal_dqdt, but internally split into the
  !! per-equation physics (volume flux, OpenACC) and the batched
  !! equation-independent element operations (CUDA), joined by global
  !! flux arrays - the structure proposed by the SCALE-DG developer.
!OCL SERIAL
  subroutine cal_dqdt_cuda( dqdt,  & ! (out)
    q, u, v, w, flux_bnd,          & ! (in)
    D1D, D1D_tr, Lift_mat, Escale, & ! (in)
    Nq, Np, NfpTot, Ne, NeA        ) ! (in)

    use mod_advect3d_eq_cuf, only: advect3d_eq_eop_cuda
    implicit none
    integer, intent(in) :: Nq, Np, NfpTot, Ne, NeA
    real(RP), intent(out) :: dqdt(Np,NeA)
    real(RP), intent(in) :: q(Np,NeA), u(Np,NeA), v(Np,NeA), w(Np,NeA)
    real(RP), intent(in) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: D1D(Nq,Nq), D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    !------------------------------------------------------------

    if ( .not. allocated(vflux_x) ) then
      allocate( vflux_x(Np,Ne), vflux_y(Np,Ne), vflux_z(Np,Ne) )
      !$acc enter data create(vflux_x, vflux_y, vflux_z)
    end if

    call Timer_start(timer_vflux)
    call cal_volflux( vflux_x, vflux_y, vflux_z, & ! (out)
       q, u, v, w, Np, Ne, NeA )                   ! (in)
    call Timer_stop(timer_vflux)

    call Timer_start(timer_eop)
    call advect3d_eq_eop_cuda( dqdt,       & ! (inout)
      vflux_x, vflux_y, vflux_z, flux_bnd, & ! (in)
      D1D, Lift_mat, Escale,               & ! (in)
      Nq, Np, NfpTot, Ne, NeA, eop_use_tc )
    call Timer_stop(timer_eop)
    return
  end subroutine cal_dqdt_cuda
#endif

  !> Compute the volume flux components as global arrays (LAYERED
  !! structure). In the full model this kernel is the per-equation
  !! physics; here it is the advective flux q*(u,v,w).
!OCL SERIAL
  subroutine cal_volflux( fx, fy, fz, & ! (out)
    q, u, v, w, Np, Ne, NeA )           ! (in)
    implicit none
    integer, intent(in) :: Np, Ne, NeA
    real(RP), intent(out) :: fx(Np,Ne), fy(Np,Ne), fz(Np,Ne)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA)
    real(RP), intent(in) :: v(Np,NeA)
    real(RP), intent(in) :: w(Np,NeA)

    integer :: ke, n
    !------------------------------------------

    !$omp parallel do private(n)
    !$acc parallel loop collapse(2) gang vector default(present)
    do ke = 1, Ne
      do n = 1, Np
        fx(n,ke) = q(n,ke) * u(n,ke)
        fy(n,ke) = q(n,ke) * v(n,ke)
        fz(n,ke) = q(n,ke) * w(n,ke)
      end do
    end do
    return
  end subroutine cal_volflux

  !> Calculate the element boundary flux
!OCL SERIAL
  subroutine cal_elembnd_flux( flux, & ! (out)
    q_, u_, v_, w_,                  & ! (in)
    VMapM, VMapP, normal_fn, Fscale, & ! (in)
    Np, NfpTot, Ne, NeA              ) ! (in)
    implicit none
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(out) :: flux(NfpTot,Ne)
    real(RP), intent(in) :: q_(Np*NeA)
    real(RP), intent(in) :: u_(Np*NeA)
    real(RP), intent(in) :: v_(Np*NeA)
    real(RP), intent(in) :: w_(Np*NeA)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3)
    real(RP), intent(in) :: Fscale(NfpTot,Ne)

    integer :: ke, fp
    integer :: iM, iP
    real(RP) :: qM, qP
    real(RP) :: VelM, VelP
    real(RP) :: alpha

    !------------------------------------------

    !$omp parallel do private(fp,iM,iP,qM,qP,VelM,VelP,alpha)
    !$acc parallel loop collapse(2) gang vector default(present) &
    !$acc private(iM,iP,qM,qP,VelM,VelP,alpha)
    do ke = 1, Ne
    do fp = 1, NfpTot
      iM = VMapM(fp,ke)
      iP = VMapP(fp,ke)

      qM = q_(iM)
      qP = q_(iP)

      VelM = &
           u_(iM)*normal_fn(fp,ke,1) &
         + v_(iM)*normal_fn(fp,ke,2) &
         + w_(iM)*normal_fn(fp,ke,3)

      VelP = &
           u_(iP)*normal_fn(fp,ke,1) &
         + v_(iP)*normal_fn(fp,ke,2) &
         + w_(iP)*normal_fn(fp,ke,3)

      alpha = 0.5_RP * abs( VelP + VelM )

      flux(fp,ke) = 0.5_RP * Fscale(fp,ke) * ( &
           qP * VelP                           &
         - qM * VelM                           &
         - alpha * ( qP - qM ) )
    end do
    end do
    return
  end subroutine cal_elembnd_flux

  !> Calculate the volume derivative and apply surface lifting
!OCL SERIAL
  subroutine cal_dqdt( dqdt,       & ! (out)
    q, u, v, w, flux_bnd,          & ! (in)
    D1D, D1D_tr, Lift_mat, Escale, & ! (in)
    Nq, Np, NfpTot, Ne, NeA        ) ! (in)

    use mod_dg_optr_kernel, only: &
      tensorprod_divlike_dirXYZ, &
      tensorprod_Lift_hexahedral
    implicit none
    integer, intent(in) :: Nq
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(out) :: dqdt(Np,NeA)
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA)
    real(RP), intent(in) :: v(Np,NeA)
    real(RP), intent(in) :: w(Np,NeA)
    real(RP), intent(in) :: flux_bnd(NfpTot,Ne)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)

    integer :: ke, n

    real(RP) :: flux_x(Np), flux_y(Np), flux_z(Np)
    real(RP) :: DxFlux(Np), DyFlux(Np), DzFlux(Np)

    real(RP) :: LiftBndFlux(Np)
    !------------------------------------------------------------

    !$omp parallel do private( n, flux_x, flux_y, flux_z, DxFlux, DyFlux, DzFlux, LiftBndFlux )
    !$acc parallel loop gang default(present) &
    !$acc private( flux_x, flux_y, flux_z, DxFlux, DyFlux, DzFlux, LiftBndFlux )
    do ke = 1, Ne
      !$acc loop vector
      do n = 1, Np
        flux_x(n) = q(n,ke) * u(n,ke)
        flux_y(n) = q(n,ke) * v(n,ke)
        flux_z(n) = q(n,ke) * w(n,ke)
      end do

      call tensorprod_divlike_dirXYZ( &
        DxFlux, DyFlux, DzFlux,       & ! (out)
        D1D, D1D_tr,                  & ! (in)
        flux_x, flux_y, flux_z, Nq    ) ! (in)

      call tensorprod_Lift_hexahedral( &
        LiftBndFlux,                 & ! (out)
        Lift_mat, flux_bnd(:,ke), Nq ) ! (in)

      !$acc loop vector
      do n = 1, Np
        dqdt(n,ke) = -( &
             Escale(n,ke,1)*DxFlux(n) &
           + Escale(n,ke,2)*DyFlux(n) &
           + Escale(n,ke,3)*DzFlux(n) &
           + LiftBndFlux(n) )
      end do
    end do
    return
  end subroutine cal_dqdt

  !> Calculate the element boundary flux of a single element
  !! (device-callable; used by the FUSED tendency kernel type)
!OCL SERIAL
  subroutine cal_elembnd_flux_elem( flux_e, & ! (out)
    ke, q_, u_, v_, w_,              & ! (in)
    VMapM, VMapP, normal_fn, Fscale, & ! (in)
    Np, NfpTot, Ne, NeA              ) ! (in)
!$acc routine vector
    implicit none
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(out) :: flux_e(NfpTot)
    integer, intent(in) :: ke
    real(RP), intent(in) :: q_(Np*NeA)
    real(RP), intent(in) :: u_(Np*NeA)
    real(RP), intent(in) :: v_(Np*NeA)
    real(RP), intent(in) :: w_(Np*NeA)
    integer, intent(in) :: VMapM(NfpTot,Ne)
    integer, intent(in) :: VMapP(NfpTot,Ne)
    real(RP), intent(in) :: normal_fn(NfpTot,Ne,3)
    real(RP), intent(in) :: Fscale(NfpTot,Ne)

    integer :: fp
    integer :: iM, iP
    real(RP) :: qM, qP
    real(RP) :: VelM, VelP
    real(RP) :: alpha
    !------------------------------------------

    !$acc loop vector private(iM,iP,qM,qP,VelM,VelP,alpha)
    do fp = 1, NfpTot
      iM = VMapM(fp,ke)
      iP = VMapP(fp,ke)

      qM = q_(iM)
      qP = q_(iP)

      VelM = &
           u_(iM)*normal_fn(fp,ke,1) &
         + v_(iM)*normal_fn(fp,ke,2) &
         + w_(iM)*normal_fn(fp,ke,3)

      VelP = &
           u_(iP)*normal_fn(fp,ke,1) &
         + v_(iP)*normal_fn(fp,ke,2) &
         + w_(iP)*normal_fn(fp,ke,3)

      alpha = 0.5_RP * abs( VelP + VelM )

      flux_e(fp) = 0.5_RP * Fscale(fp,ke) * ( &
           qP * VelP                           &
         - qM * VelM                           &
         - alpha * ( qP - qM ) )
    end do
    return
  end subroutine cal_elembnd_flux_elem

  !> Calculate the volume derivative and apply surface lifting of a single
  !! element (device-callable; used by the FUSED tendency kernel type)
  !! The work arrays are passed from the caller because automatic arrays
  !! cannot be used inside device routines.
!OCL SERIAL
  subroutine cal_dqdt_elem( dqdt,  & ! (inout)
    ke, q, u, v, w, flux_e,        & ! (in)
    D1D, D1D_tr, Lift_mat, Escale, & ! (in)
    flux_x, flux_y, flux_z,        & ! (work)
    DxFlux, DyFlux, DzFlux,        & ! (work)
    LiftBndFlux,                   & ! (work)
    Nq, Np, NfpTot, Ne, NeA        ) ! (in)

    use mod_dg_optr_kernel, only: &
      tensorprod_divlike_dirXYZ, &
      tensorprod_Lift_hexahedral
!$acc routine vector
    implicit none
    integer, intent(in) :: Nq
    integer, intent(in) :: Np
    integer, intent(in) :: NfpTot
    integer, intent(in) :: Ne
    integer, intent(in) :: NeA
    real(RP), intent(inout) :: dqdt(Np,NeA)
    integer, intent(in) :: ke
    real(RP), intent(in) :: q(Np,NeA)
    real(RP), intent(in) :: u(Np,NeA)
    real(RP), intent(in) :: v(Np,NeA)
    real(RP), intent(in) :: w(Np,NeA)
    real(RP), intent(in) :: flux_e(NfpTot)
    real(RP), intent(in) :: D1D(Nq,Nq)
    real(RP), intent(in) :: D1D_tr(Nq,Nq)
    real(RP), intent(in) :: Lift_mat(Nq,Nq,Nq,6)
    real(RP), intent(in) :: Escale(Np,Ne,3)
    real(RP), intent(out) :: flux_x(Np), flux_y(Np), flux_z(Np)
    real(RP), intent(out) :: DxFlux(Np), DyFlux(Np), DzFlux(Np)
    real(RP), intent(out) :: LiftBndFlux(Np)

    integer :: n
    !------------------------------------------------------------

    !$acc loop vector
    do n = 1, Np
      flux_x(n) = q(n,ke) * u(n,ke)
      flux_y(n) = q(n,ke) * v(n,ke)
      flux_z(n) = q(n,ke) * w(n,ke)
    end do

    call tensorprod_divlike_dirXYZ( &
      DxFlux, DyFlux, DzFlux,       & ! (out)
      D1D, D1D_tr,                  & ! (in)
      flux_x, flux_y, flux_z, Nq    ) ! (in)

    call tensorprod_Lift_hexahedral( &
      LiftBndFlux,          & ! (out)
      Lift_mat, flux_e, Nq  ) ! (in)

    !$acc loop vector
    do n = 1, Np
      dqdt(n,ke) = -( &
           Escale(n,ke,1)*DxFlux(n) &
         + Escale(n,ke,2)*DyFlux(n) &
         + Escale(n,ke,3)*DzFlux(n) &
         + LiftBndFlux(n) )
    end do
    return
  end subroutine cal_dqdt_elem
end module mod_advect3d_eq
