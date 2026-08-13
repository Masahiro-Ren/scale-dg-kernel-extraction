program main
  use mod_common, only: RP, PI
  use mod_mesh, only: &
    PolyOrder, Nq, Np, NfpTot, Ne, NeA,  &
    D1D, D1D_tr, Lift_mat, VMapM, VMapP, &
    normal_fn, Escale, Fscale, pos_en, update_halo
  use mod_advect3d_kernel, only: &
    advect3d_kernel_cal_tend
  
  implicit none

  integer :: nstep, output_interval
  integer :: istep, stage
  real(RP) :: dt

  real(RP), allocatable :: q(:,:)
  real(RP), allocatable :: q0(:,:)
  real(RP), allocatable :: dqdt(:,:)
  real(RP), allocatable :: u(:,:)
  real(RP), allocatable :: v(:,:)
  real(RP), allocatable :: w(:,:)

  ! 3-stage third-order SSP Runge-Kutta method
  integer, parameter :: RK_nstage = 3
  real(RP), parameter :: rk_a(RK_nstage) = [ 0.0_RP, 0.75_RP, 1.0_RP/3.0_RP ]
  real(RP), parameter :: rk_b(RK_nstage) = [ 1.0_RP, 0.25_RP, 2.0_RP/3.0_RP ]

  integer :: kelem
  !------------------------------------------------------------

  call init()

  !============================================================
  ! SSPRK(3,3)
  !============================================================

  do istep = 1, nstep

    !$omp parallel do
    do kelem=1, Ne 
      q0(:,kelem) = q(:,kelem)
    end do

    do stage = 1, RK_nstage
      call update_halo(q)

      call advect3d_kernel_cal_tend( dqdt,       & ! (out)
        q, u, v, w,                              & ! (in)
        D1D, D1D_tr, Lift_mat,                   & ! (in)
        VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
        Nq, Np, NfpTot, Ne, NeA )

      !$omp parallel do
      do kelem=1, Ne
        q(:,kelem) = rk_a(stage) * q0(:,kelem) &
                   + rk_b(stage) * (q(:,kelem) + dt * dqdt(:,kelem))
      end do
    end do

    if (mod(istep,output_interval) == 0) then
      write(*,'(I8,2ES24.15)') &
           istep, minval(q(:,1:Ne)), maxval(q(:,1:Ne))
    end if
  end do

contains
  subroutine init()
    use mod_mesh, only: mesh_init
    implicit none

    character(len=256) :: conf_file
    integer :: NeX = 4
    integer :: NeY = 4
    integer :: NeZ = 4
    integer :: PolyOrder = 3
    real(RP) :: vel_x = 1.0_RP
    real(RP) :: vel_y = 1.0_RP
    real(RP) :: vel_z = 1.0_RP

    namelist /PARAM_ADVECT3D/ &
      NeX, NeY, NeZ, PolyOrder,   &
      dt, nstep, output_interval, &
      vel_x, vel_y, vel_z

    integer :: fid
    integer :: ke, p
    !------------------------------------------------------------

    dt    = 1.0e-3_RP
    nstep = 100
    output_interval = 10

    call get_command_argument(1,conf_file)

    fid = 10
    open(fid,file=trim(conf_file),status='old',action='read')
    read(fid,nml=PARAM_ADVECT3D)
    close(fid)

    !-
    call mesh_init( NeX, NeY, NeZ, PolyOrder, &
      1.0_RP, 1.0_RP, 1.0_RP )

    allocate( q(Np,NeA), q0(Np,NeA), dqdt(Np,NeA))
    allocate( u(Np,NeA), v(Np,NeA), w(Np,NeA))

    !- Initial condition

    do ke = 1, Ne
    do p = 1, Np
      q(p,ke) = sin(2.0_RP*PI*pos_en(p,ke,1)) &
              * sin(2.0_RP*PI*pos_en(p,ke,2)) &
              * sin(2.0_RP*PI*pos_en(p,ke,3))

      u(p,ke) = vel_x
      v(p,ke) = vel_y
      w(p,ke) = vel_z
    end do
    end do

    call update_halo(q)
    call update_halo(u)
    call update_halo(v)
    call update_halo(w)
    return
  end subroutine init

end program main