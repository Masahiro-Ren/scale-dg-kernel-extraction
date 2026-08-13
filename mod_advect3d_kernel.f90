module mod_advect3d_kernel
  use mod_common, only: RP
  implicit none
  private

  public :: advect3d_kernel_cal_tend

contains
  !> Advection DG kernel
!OCL SERIAL
  subroutine advect3d_kernel_cal_tend( dqdt, & ! (out)
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

    call cal_elembnd_flux( ebnd_flux,   & ! (out)
       q, u, v, w,                      & ! (in)
       VMapM, VMapP, normal_fn, Fscale, & ! (in)
       Np, NfpTot, Ne, NeA )

    call cal_dqdt( dqdt,               & ! (out)
       q, u, v, w,  ebnd_flux,         & ! (in)
       D1D, D1D_tr, Lift_mat,          & ! (in)
       Escale, Nq, Np, NfpTot, Ne, NeA ) ! (in)

     return
  end subroutine advect3d_kernel_cal_tend

  !> Calculate the element boundary flux
!OCL SERIAL
  subroutine cal_elembnd_flux( flux, & ! (out)
    q_, u_, v_, w_,                  & ! (in)
    VMapM, VMapP, normal_fn, Fscale, & ! (in)
    Np, NfpTot, Ne, NeA              ) ! (in)

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

  !> Tensor-product differentiation
!OCL SERIAL
  subroutine tensorprod_divlike_dirXYZ( &
    vec_out_x, vec_out_y, vec_out_z, &
    Mat, Mat_tr,                     &
    vec_in_x, vec_in_y, vec_in_z,    &
    Nq )
    implicit none
    integer, intent(in) :: Nq
    real(RP), intent(in) :: Mat(Nq,Nq)
    real(RP), intent(in) :: Mat_tr(Nq,Nq)
    real(RP), intent(in) :: vec_in_x(Nq,Nq**2)
    real(RP), intent(in) :: vec_in_y(Nq,Nq,Nq)
    real(RP), intent(in) :: vec_in_z(Nq,Nq,Nq)
    real(RP), intent(out) :: vec_out_x(Nq,Nq**2)
    real(RP), intent(out) :: vec_out_y(Nq,Nq**2)
    real(RP), intent(out) :: vec_out_z(Nq,Nq**2)

    integer :: i,j,k,l,jk
    !----------------------------------------------------------

    vec_out_x(:,:) = 0.0_RP
    vec_out_y(:,:) = 0.0_RP
    vec_out_z(:,:) = 0.0_RP

    !- x-direction
    do jk = 1, Nq**2
      do i = 1, Nq
        do l = 1, Nq
          vec_out_x(i,jk) = &
               vec_out_x(i,jk) &
             + Mat(i,l)*vec_in_x(l,jk)
        end do
      end do
    end do

    !- y-direction
    do k = 1, Nq
      do j = 1, Nq
        jk = j + (k-1)*Nq
        do i = 1, Nq
          do l = 1, Nq
            vec_out_y(i,jk) = vec_out_y(i,jk) &
               + vec_in_y(i,l,k)*Mat_tr(l,j)
          end do
        end do
      end do
    end do

    !- z-direction
    do k = 1, Nq
      do j = 1, Nq
        jk = j + (k-1)*Nq
        do i = 1, Nq
          do l = 1, Nq
            vec_out_z(i,jk) = vec_out_z(i,jk) &
               + vec_in_z(i,j,l)*Mat_tr(l,k)

          end do
        end do
      end do
    end do
    return
  end subroutine tensorprod_divlike_dirXYZ

  !> Tensor-product lifting for a hexahedral element
!OCL SERIAL
  subroutine tensorprod_Lift_hexahedral( &
    vec_out,         &
    Lift, vec_in, Nq )
    implicit none
    integer, intent(in) :: Nq
    real(RP), intent(out) :: vec_out(Nq,Nq,Nq)
    real(RP), intent(in) :: Lift(Nq,Nq,Nq,6)
    real(RP), intent(in) :: vec_in(Nq,Nq,6)

    integer :: i,j,k
    !----------------------------------------------------------

    do k = 1, Nq
    do j = 1, Nq
    do i = 1, Nq
      vec_out(i,j,k) = &
            Lift(i,j,k,1)*vec_in(i,k,1) &
          + Lift(i,j,k,2)*vec_in(j,k,2) &
          + Lift(i,j,k,3)*vec_in(i,k,3) &
          + Lift(i,j,k,4)*vec_in(j,k,4) &
          + Lift(i,j,k,5)*vec_in(i,j,5) &
          + Lift(i,j,k,6)*vec_in(i,j,6)
    end do
    end do
    end do
    return
  end subroutine tensorprod_Lift_hexahedral
end module mod_advect3d_kernel