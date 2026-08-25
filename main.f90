!> Main program for SCALE-DG kernel extraction
!!
!! @author Yuta Kawai, Team SCALE
!!
program main
  !-----------------------------------------------------------------------------
  !
  !++ used modules
  !  
  use mod_common, only: &
    RP,                                     &
    Timer,                                  &
    Timer_start, Timer_stop, Timer_elapsed, &
    RK_nstage => RK3s3oSSP_nstage,          &
    rk_a => RK3s3oSSP_rk_a,                 &
    rk_b => RK3s3oSSP_rk_b
  use mod_mesh, only: &
    PolyOrder, Nq, Np, NfpTot, Ne, NeA,  &
    D1D, D1D_tr, Lift_mat, VMapM, VMapP, &
    normal_fn, Escale, Fscale, pos_en, update_halo
  use mod_advect3d_eq, only: &
    advect3d_eq_cal_tend
  implicit none

  !-----------------------------------------------------------------------------
  !
  !++ Public parameters & variables
  !

  integer :: nstep, output_interval
  integer :: istep, stage
  real(RP) :: dt

  real(RP), allocatable :: q(:,:)
  real(RP), allocatable :: q0(:,:)
  real(RP), allocatable :: dqdt(:,:)
  real(RP), allocatable :: u(:,:)
  real(RP), allocatable :: v(:,:)
  real(RP), allocatable :: w(:,:)

  integer :: kelem
  integer :: pn

  type(Timer) :: timer_main
  type(Timer) :: timer_cal_tend

  !- Main program ----------------------------------------------------------

  call init()

  !- Loop for time integration
  do istep = 1, nstep

    !$omp parallel do
    !$acc parallel loop collapse(2) default(present)
    do kelem=1, Ne
      do pn=1, Np
        q0(pn,kelem) = q(pn,kelem)
      end do
    end do

    do stage = 1, RK_nstage
      call update_halo(q)

      call Timer_start(timer_cal_tend)
      call advect3d_eq_cal_tend( dqdt,       & ! (out)
        q, u, v, w,                              & ! (in)
        D1D, D1D_tr, Lift_mat,                   & ! (in)
        VMapM, VMapP, normal_fn, Escale, Fscale, & ! (in)
        Nq, Np, NfpTot, Ne, NeA )
      call Timer_stop(timer_cal_tend)

      !$omp parallel do
      !$acc parallel loop collapse(2) default(present)
      do kelem=1, Ne
        do pn=1, Np
          q(pn,kelem) = rk_a(stage) * q0(pn,kelem) &
                      + rk_b(stage) * ( q(pn,kelem) + dt * dqdt(pn,kelem) )
        end do
      end do
    end do

    if (mod(istep,output_interval) == 0) then
      !$acc update host(q)
      write(*,'(I8,2ES24.15)') &
           istep, minval(q(:,1:Ne)), maxval(q(:,1:Ne))
    end if
  end do

  call final()
contains
  !> Initialize modules with DG mesh and DG operator kernel
  subroutine init()
    use, intrinsic :: iso_fortran_env, only: error_unit
    use mod_common, only: PI
    use mod_mesh, only: mesh_setup
    use mod_advect3d_eq, only: setup_advect3d_eq_setup
    use mod_dg_optr_kernel, only: dg_optr_kernel_setup    
    implicit none

    character(len=256) :: conf_file
    integer :: NeX = 4
    integer :: NeY = 4
    integer :: NeZ = 4
    integer :: PolyOrder = 3
    real(RP) :: vel_x = 1.0_RP
    real(RP) :: vel_y = 1.0_RP
    real(RP) :: vel_z = 1.0_RP
    character(len=8) :: DGOptrKernel_OptType = "OPT1" ! GENERAL or OPT1

    namelist /PARAM_ADVECT3D/ &
      NeX, NeY, NeZ, PolyOrder,   &
      dt, nstep, output_interval, &
      vel_x, vel_y, vel_z,        &
      DGOptrKernel_OptType

    integer :: fid
    integer :: ke, p
    !------------------------------------------------------------

    dt    = 1.0e-3_RP
    nstep = 100
    output_interval = 10

    if (command_argument_count() < 1) then
      write(error_unit,'(A)') 'Error: configuration file argument is required.'
      write(error_unit,'(A)') 'Usage: scale-dg_extraction input.conf'
      flush(error_unit)
      error stop 1
    end if

    call get_command_argument(1,conf_file)

    fid = 10
    open(fid,file=trim(conf_file),status='old',action='read')
    read(fid,nml=PARAM_ADVECT3D)
    close(fid)

    !- Initialize a mesh module
    call mesh_setup( NeX, NeY, NeZ, PolyOrder, &
      1.0_RP, 1.0_RP, 1.0_RP )

    allocate( q(Np,NeA), q0(Np,NeA), dqdt(Np,NeA) )
    allocate( u(Np,NeA), v(Np,NeA), w(Np,NeA) )

    !- Initialize a DG operator module
    call dg_optr_kernel_setup( DGOptrKernel_OptType )

    !- Initialize a advection equation module
    call setup_advect3d_eq_setup()

    !- Set initial condition

    !$omp parallel do
    do ke = 1, Ne
    do p = 1, Np
      q(p,ke) = sin( 2.0_RP*PI*pos_en(p,ke,1) ) &
              * sin( 2.0_RP*PI*pos_en(p,ke,2) ) &
              * sin( 2.0_RP*PI*pos_en(p,ke,3) )

      u(p,ke) = vel_x
      v(p,ke) = vel_y
      w(p,ke) = vel_z
    end do
    end do

    ! Move the field arrays to the device; halo exchange runs there.
    !$acc enter data copyin(q, u, v, w) create(q0, dqdt)

    call update_halo(q)
    call update_halo(u)
    call update_halo(v)
    call update_halo(w)

    call Timer_start(timer_main)
    return
  end subroutine init

!OCL SERIAL
  subroutine final()
    use mod_advect3d_eq, only: setup_advect3d_eq_finalize
    implicit none
    !-----------------------------------------------------------------------------

    call Timer_stop(timer_main)
    write(*,'(A)') "= Report of execution time [sec]"
    write(*,'(A30,ES24.5)') "Main:", Timer_elapsed(timer_main)
    write(*,'(A30,ES24.5)') "Cal_tend:", Timer_elapsed(timer_cal_tend)

    call setup_advect3d_eq_finalize()
    return
  end subroutine final
end program main
