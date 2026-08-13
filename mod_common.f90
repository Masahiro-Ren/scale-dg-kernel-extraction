module mod_common
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  integer, parameter :: RP = real64
  real(RP), parameter :: PI = 4.0_RP*atan(1.0_RP)
end module mod_common