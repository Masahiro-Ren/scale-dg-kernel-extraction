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
contains
  !> Setup
!OCL SERIAL
  subroutine setup_advect3d_eq_setup()
    implicit none
    !------------------------------------------------------------------------------
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
    real(RP) :: ebnd_flux(NfpTot,Ne)

    !------------------------------------------------------------

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