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

  integer :: mesh_NeX, mesh_NeY, mesh_NeZ  !< mesh dims (for cluster launch)

  integer :: tend_kernel_typeid = TEND_KERNEL_TYPEID_SPLIT

  ! Work array for the element boundary flux (SPLIT kernel type only).
  ! Allocated on the heap (not as an automatic array) so that large meshes
  ! do not overflow the stack, and kept resident on the device.
  real(RP), allocatable :: ebnd_flux(:,:)
contains
  !> Setup
!OCL SERIAL
  subroutine setup_advect3d_eq_setup( tend_kernel_type, NeX, NeY, NeZ )
    implicit none
    character(len=*), intent(in) :: tend_kernel_type
    integer, intent(in) :: NeX, NeY, NeZ
    !------------------------------------------------------------------------------
    mesh_NeX = NeX; mesh_NeY = NeY; mesh_NeZ = NeZ
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
    if ( tend_kernel_typeid >= TEND_KERNEL_TYPEID_CUF ) then
      write(*,'(A30,ES24.5)') "CUF fused tendency:", Timer_elapsed(timer_cuf)
    else if ( tend_kernel_typeid == TEND_KERNEL_TYPEID_FUSED ) then
      ! Both phases run in one kernel; only the combined time is
      ! measurable from the host.
      write(*,'(A30,ES24.5)') "Fused flux+volume+lift:", Timer_elapsed(timer_fused)
    else
      write(*,'(A30,ES24.5)') "Element boundary flux:", Timer_elapsed(timer_ebnd_flux)
      write(*,'(A30,ES24.5)') "Volume derivate + surface lift:", Timer_elapsed(timer_dqdt)
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

    if ( tend_kernel_typeid >= TEND_KERNEL_TYPEID_CUF ) then

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
      call cal_dqdt( dqdt,               & ! (out)
         q, u, v, w,  ebnd_flux,         & ! (in)
         D1D, D1D_tr, Lift_mat,          & ! (in)
         Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)
      call Timer_stop(timer_dqdt)

    end if

     return
  end subroutine advect3d_eq_cal_tend

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
