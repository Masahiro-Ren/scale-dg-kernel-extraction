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

  !> Tendency kernel type: every element computes all 6 face fluxes
  !! (interior faces are evaluated twice, once per side) - original
  integer, parameter :: TEND_KERNEL_TYPEID_SPLIT = 1
  !> Tendency kernel type: each interior face flux is computed once.
  !! Three direction passes; the owner element computes its plus face
  !! and writes both sides: with the owner's outward normal,
  !!   F_own  = 0.5*Fs*(S - D),  F_twin = F_own + Fs*D,
  !! so the twin's entry costs one extra fused multiply-add instead of
  !! 8 gathers and two dot products. Halves the flux computations and
  !! the flux-phase gather reads; every flux slot has exactly one
  !! writer, so no synchronization is needed beyond the loop barriers.
  integer, parameter :: TEND_KERNEL_TYPEID_FLUXONCE = 2
  !> Like FLUXONCE but as a single sweep over elements with per-thread
  !! ring buffers, sharing a face only when its reuse distance fits in
  !! cache: -x faces reuse the previous element's +x result (register/L1
  !! distance), -y faces reuse a pencil ring buffer (NeX faces, ~L2),
  !! and z faces are recomputed on both sides because their reuse
  !! distance (NeX*NeY elements) exceeds any cache level.
  integer, parameter :: TEND_KERNEL_TYPEID_SWEEP = 3

  integer :: tend_kernel_typeid = TEND_KERNEL_TYPEID_SPLIT
  integer :: mesh_NeX, mesh_NeY, mesh_NeZ  !< mesh dims (for twin-element ids)

  ! Work array for the element boundary flux. Allocated on the heap (not
  ! as an automatic array) so that large meshes do not overflow the stack.
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
    case ("FLUXONCE")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_FLUXONCE
    case ("SWEEP")
      tend_kernel_typeid = TEND_KERNEL_TYPEID_SWEEP
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
    write(*,'(A30,ES24.5)') "Element boundary flux:", Timer_elapsed(timer_ebnd_flux)
    write(*,'(A30,ES24.5)') "Volume derivate + surface lift:", Timer_elapsed(timer_dqdt)
    return
  end subroutine setup_advect3d_eq_finalize

  !> Calculate the tendency of 3D advection equation
!OCL SERIAL
  subroutine advect3d_eq_cal_tend( dqdt, & ! (out)
    q, u, v, w,                              & ! (in)
    D1D, D1D_tr, Lift_mat,                   & ! (in)
    VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
    Nq, Np, NfpTot, Ne, NeA )                  ! (in)

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

    !------------------------------------------------------------

    if ( .not. allocated(ebnd_flux) ) then
      allocate( ebnd_flux(NfpTot,Ne) )
    end if

    call Timer_start(timer_ebnd_flux)
    if ( tend_kernel_typeid == TEND_KERNEL_TYPEID_FLUXONCE ) then
      call cal_elembnd_flux_once( ebnd_flux, & ! (out)
         q, u, v, w,                      & ! (in)
         VMapM, VMapP, normal_fn, Fscale, & ! (in)
         mesh_NeX, mesh_NeY, mesh_NeZ,    & ! (in)
         Np, NfpTot, Ne, NeA )
    else if ( tend_kernel_typeid == TEND_KERNEL_TYPEID_SWEEP ) then
      call cal_elembnd_flux_sweep( ebnd_flux, & ! (out)
         q, u, v, w,                      & ! (in)
         VMapM, VMapP, normal_fn, Fscale, & ! (in)
         mesh_NeX, mesh_NeY, mesh_NeZ,    & ! (in)
         Np, NfpTot, Ne, NeA )
    else
      call cal_elembnd_flux( ebnd_flux,   & ! (out)
         q, u, v, w,                      & ! (in)
         VMapM, VMapP, normal_fn, Fscale, & ! (in)
         Np, NfpTot, Ne, NeA )
    end if
    call Timer_stop(timer_ebnd_flux)

    call Timer_start(timer_dqdt)
    call cal_dqdt( dqdt,               & ! (out)
       q, u, v, w,  ebnd_flux,         & ! (in)
       D1D, D1D_tr, Lift_mat,          & ! (in)
       Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)
    call Timer_stop(timer_dqdt)

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

    integer :: ke
    integer :: iM(NfpTot), iP(NfpTot)
    real(RP) :: qM(NfpTot), qP(NfpTot)
    real(RP) :: VelM(NfpTot), VelP(NfpTot)
    real(RP) :: alpha(NfpTot)

    !------------------------------------------

    !$omp parallel do private(iM,iP,qM,qP,VelM,VelP,alpha)
    do ke = 1, Ne
      iM(:) = VMapM(:,ke)
      iP(:) = VMapP(:,ke)

      qM(:) = q_(iM(:))
      qP(:) = q_(iP(:))

      VelM(:) = &
           u_(iM(:))*normal_fn(:,ke,1) &
         + v_(iM(:))*normal_fn(:,ke,2) &
         + w_(iM(:))*normal_fn(:,ke,3)
     
      VelP(:) = &
           u_(iP(:))*normal_fn(:,ke,1) &
         + v_(iP(:))*normal_fn(:,ke,2) &
         + w_(iP(:))*normal_fn(:,ke,3)

      alpha(:) = 0.5_RP * abs( VelP(:) + VelM(:) )

      flux(:,ke) = 0.5_RP * Fscale(:,ke) * ( &
           qP(:) * VelP(:)                   &
         - qM(:) * VelM(:)                   &
         - alpha(:) * ( qP(:) - qM(:) ) )
    end do
    return
  end subroutine cal_elembnd_flux

  !> Calculate the element boundary flux, computing each interior face
  !! flux ONCE (TendencyKernel_Type = "FLUXONCE").
  !!
  !! Three passes, one per direction. In pass d every element evaluates
  !! only its plus face (+x, +y, +z); the minus-face entry of the twin
  !! element across that face is reconstructed from the same evaluation:
  !! with the owner's outward normal n, S = qP*(uP.n) - qM*(uM.n) and
  !! D = alpha*(qP - qM),
  !!    flux(own +face)   = 0.5*Fs*(S - D)
  !!    flux(twin -face)  = 0.5*Fs*(S + D) = flux(own) + Fs*D
  !! (Fscale matches across a face on this uniform mesh; face-node
  !! ordering matches by construction of Fmask/VMapP.) Each flux slot
  !! has exactly one writer, so the passes need no synchronization.
  !! The reconstruction reassociates the arithmetic: results can differ
  !! from SPLIT at ULP level (bitwise identical for uniform velocity).
!OCL SERIAL
  subroutine cal_elembnd_flux_once( flux, & ! (out)
    q_, u_, v_, w_,                  & ! (in)
    VMapM, VMapP, normal_fn, Fscale, & ! (in)
    NeX, NeY, NeZ,                   & ! (in)
    Np, NfpTot, Ne, NeA              ) ! (in)
    implicit none
    integer, intent(in) :: NeX, NeY, NeZ
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

    integer :: dir, f_p, f_m, sp0, sm0
    integer :: ke, keT
    integer :: i, j, k
    integer :: iM(NfpTot/6), iP(NfpTot/6)
    real(RP) :: qM(NfpTot/6), qP(NfpTot/6)
    real(RP) :: VelM(NfpTot/6), VelP(NfpTot/6)
    real(RP) :: alpha(NfpTot/6)
    real(RP) :: Fown(NfpTot/6)

    integer :: nfp
    !------------------------------------------

    nfp = NfpTot/6

    ! Faces: 1:-y 2:+x 3:+y 4:-x 5:-z 6:+z
    do dir = 1, 3
      select case (dir)
      case (1)
        f_p = 2; f_m = 4
      case (2)
        f_p = 3; f_m = 1
      case (3)
        f_p = 6; f_m = 5
      end select
      sp0 = (f_p-1)*nfp
      sm0 = (f_m-1)*nfp

      !$omp parallel do private(i,j,k,keT,iM,iP,qM,qP,VelM,VelP,alpha,Fown)
      do ke = 1, Ne
        ! twin element across this element's plus face (periodic wrap)
        i = mod(ke-1, NeX) + 1
        j = mod((ke-1)/NeX, NeY) + 1
        k = (ke-1)/(NeX*NeY) + 1
        select case (dir)
        case (1)
          if ( i < NeX ) then; keT = ke + 1
          else;                keT = ke - (NeX-1); end if
        case (2)
          if ( j < NeY ) then; keT = ke + NeX
          else;                keT = ke - (NeY-1)*NeX; end if
        case (3)
          if ( k < NeZ ) then; keT = ke + NeX*NeY
          else;                keT = ke - (NeZ-1)*NeX*NeY; end if
        end select

        iM(:) = VMapM(sp0+1:sp0+nfp, ke)
        iP(:) = VMapP(sp0+1:sp0+nfp, ke)

        qM(:) = q_(iM(:))
        qP(:) = q_(iP(:))

        VelM(:) = &
             u_(iM(:))*normal_fn(sp0+1:sp0+nfp,ke,1) &
           + v_(iM(:))*normal_fn(sp0+1:sp0+nfp,ke,2) &
           + w_(iM(:))*normal_fn(sp0+1:sp0+nfp,ke,3)

        VelP(:) = &
             u_(iP(:))*normal_fn(sp0+1:sp0+nfp,ke,1) &
           + v_(iP(:))*normal_fn(sp0+1:sp0+nfp,ke,2) &
           + w_(iP(:))*normal_fn(sp0+1:sp0+nfp,ke,3)

        alpha(:) = 0.5_RP * abs( VelP(:) + VelM(:) )

        Fown(:) = 0.5_RP * Fscale(sp0+1:sp0+nfp,ke) * ( &
             qP(:) * VelP(:)                             &
           - qM(:) * VelM(:)                             &
           - alpha(:) * ( qP(:) - qM(:) ) )

        flux(sp0+1:sp0+nfp, ke) = Fown(:)

        ! twin's minus face: one FMA instead of a full re-evaluation
        flux(sm0+1:sm0+nfp, keT) = Fown(:) &
           + Fscale(sp0+1:sp0+nfp,ke) * alpha(:) * ( qP(:) - qM(:) )
      end do
    end do
    return
  end subroutine cal_elembnd_flux_once

  !> Calculate the element boundary flux with a single sweep over
  !! elements and cache-resident sharing (TendencyKernel_Type = "SWEEP").
  !!
  !! Each thread sweeps a contiguous range of elements in memory order.
  !! When an element's minus-face twin was computed earlier in the same
  !! thread's sweep, its flux is reconstructed with one FMA
  !! (F_minus = F_plus + Fs*D) from a per-thread buffer instead of
  !! being re-evaluated:
  !!   -x: the previous element's +x result (one-face buffer)
  !!   -y: a pencil ring buffer of NeX faces (~L2-resident)
  !!   -z: recomputed on both sides - the reuse distance (NeX*NeY
  !!       elements) exceeds any cache, so recompute beats a DRAM
  !!       round trip of (F,H)
  !! Periodic-wrap faces and thread-range boundaries fall back to direct
  !! evaluation, so no synchronization is needed anywhere.
!OCL SERIAL
  subroutine cal_elembnd_flux_sweep( flux, & ! (out)
    q_, u_, v_, w_,                  & ! (in)
    VMapM, VMapP, normal_fn, Fscale, & ! (in)
    NeX, NeY, NeZ,                   & ! (in)
    Np, NfpTot, Ne, NeA              ) ! (in)
!$  use omp_lib, only: omp_get_thread_num, omp_get_num_threads
    implicit none
    integer, intent(in) :: NeX, NeY, NeZ
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

    integer :: nfp
    integer :: sxp, sxm, syp, sym, szp, szm
    integer :: it, nthr, chunk, ks, kend
    integer :: ke, i, j

    ! per-thread reuse buffers (private in the parallel region)
    real(RP) :: bufFx(NfpTot/6),     bufHx(NfpTot/6)      ! previous +x face
    real(RP) :: bufFy(NfpTot/6,NeX), bufHy(NfpTot/6,NeX)  ! +y pencil ring
    real(RP) :: Ftmp(NfpTot/6), Htmp(NfpTot/6)
    !------------------------------------------

    nfp = NfpTot/6
    ! Faces: 1:-y 2:+x 3:+y 4:-x 5:-z 6:+z
    sxp = 1*nfp; sxm = 3*nfp
    syp = 2*nfp; sym = 0
    szp = 5*nfp; szm = 4*nfp

    !$omp parallel private(it, nthr, chunk, ks, kend, ke, i, j, &
    !$omp                  bufFx, bufHx, bufFy, bufHy, Ftmp, Htmp)
    it = 0; nthr = 1
    !$ it   = omp_get_thread_num()
    !$ nthr = omp_get_num_threads()
    chunk = (Ne + nthr - 1) / nthr
    ks    = it*chunk + 1
    kend  = min(Ne, (it+1)*chunk)

    do ke = ks, kend
      i = mod(ke-1, NeX) + 1
      j = mod((ke-1)/NeX, NeY) + 1

      ! -x face: previous element's +x result, one FMA
      if ( i > 1 .and. ke-1 >= ks ) then
        flux(sxm+1:sxm+nfp, ke) = bufFx(:) + bufHx(:)
      else
        call compute_face( ke, sxm, Ftmp, Htmp )
        flux(sxm+1:sxm+nfp, ke) = Ftmp(:)
      end if

      ! -y face: pencil ring buffer, one FMA
      if ( j > 1 .and. ke-NeX >= ks ) then
        flux(sym+1:sym+nfp, ke) = bufFy(:,i) + bufHy(:,i)
      else
        call compute_face( ke, sym, Ftmp, Htmp )
        flux(sym+1:sym+nfp, ke) = Ftmp(:)
      end if

      ! -z face: reuse distance exceeds cache; recompute
      call compute_face( ke, szm, Ftmp, Htmp )
      flux(szm+1:szm+nfp, ke) = Ftmp(:)

      ! +x face: compute once, keep (F,H) for the next element
      call compute_face( ke, sxp, bufFx, bufHx )
      flux(sxp+1:sxp+nfp, ke) = bufFx(:)

      ! +y face: compute once, keep (F,H) in the pencil ring
      call compute_face( ke, syp, Ftmp, Htmp )
      flux(syp+1:syp+nfp, ke) = Ftmp(:)
      bufFy(:,i) = Ftmp(:)
      bufHy(:,i) = Htmp(:)

      ! +z face
      call compute_face( ke, szp, Ftmp, Htmp )
      flux(szp+1:szp+nfp, ke) = Ftmp(:)
    end do
    !$omp end parallel
    return
  contains
    !> Evaluate the upwind flux F and dissipation term H = Fs*D for the
    !! nfp nodes of one face (slots s0+1..s0+nfp) of element ke0.
    subroutine compute_face( ke0, s0, F, H )
      implicit none
      integer, intent(in) :: ke0, s0
      real(RP), intent(out) :: F(NfpTot/6), H(NfpTot/6)

      integer :: iM(NfpTot/6), iP(NfpTot/6)
      real(RP) :: qM(NfpTot/6), qP(NfpTot/6)
      real(RP) :: VelM(NfpTot/6), VelP(NfpTot/6)
      real(RP) :: alpha(NfpTot/6)
      !----------------------------------------

      iM(:) = VMapM(s0+1:s0+nfp, ke0)
      iP(:) = VMapP(s0+1:s0+nfp, ke0)

      qM(:) = q_(iM(:))
      qP(:) = q_(iP(:))

      VelM(:) = &
           u_(iM(:))*normal_fn(s0+1:s0+nfp,ke0,1) &
         + v_(iM(:))*normal_fn(s0+1:s0+nfp,ke0,2) &
         + w_(iM(:))*normal_fn(s0+1:s0+nfp,ke0,3)

      VelP(:) = &
           u_(iP(:))*normal_fn(s0+1:s0+nfp,ke0,1) &
         + v_(iP(:))*normal_fn(s0+1:s0+nfp,ke0,2) &
         + w_(iP(:))*normal_fn(s0+1:s0+nfp,ke0,3)

      alpha(:) = 0.5_RP * abs( VelP(:) + VelM(:) )

      F(:) = 0.5_RP * Fscale(s0+1:s0+nfp,ke0) * ( &
           qP(:) * VelP(:)                         &
         - qM(:) * VelM(:)                         &
         - alpha(:) * ( qP(:) - qM(:) ) )

      H(:) = Fscale(s0+1:s0+nfp,ke0) * alpha(:) * ( qP(:) - qM(:) )
      return
    end subroutine compute_face
  end subroutine cal_elembnd_flux_sweep

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

    integer :: ke

    real(RP) :: flux_x(Np), flux_y(Np), flux_z(Np)
    real(RP) :: DxFlux(Np), DyFlux(Np), DzFlux(Np)

    real(RP) :: LiftBndFlux(Np)
    !------------------------------------------------------------

    !$omp parallel do private( flux_x, flux_y, flux_z, DxFlux, DyFlux, DzFlux, LiftBndFlux )
    do ke = 1, Ne
      flux_x(:) = q(:,ke) * u(:,ke)
      flux_y(:) = q(:,ke) * v(:,ke)
      flux_z(:) = q(:,ke) * w(:,ke)

      call tensorprod_divlike_dirXYZ( &
        DxFlux, DyFlux, DzFlux,       & ! (out)
        D1D, D1D_tr,                  & ! (in)
        flux_x, flux_y, flux_z, Nq    ) ! (in)

      call tensorprod_Lift_hexahedral( &
        LiftBndFlux,                 & ! (out)
        Lift_mat, flux_bnd(:,ke), Nq ) ! (in)

      dqdt(:,ke) = -( &
           Escale(:,ke,1)*DxFlux(:) &
         + Escale(:,ke,2)*DyFlux(:) &
         + Escale(:,ke,3)*DzFlux(:) &
         + LiftBndFlux(:) )
    end do
    return
  end subroutine cal_dqdt
end module mod_advect3d_eq