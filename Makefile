ifeq ($(origin FC), default)
FC      = gfortran
endif
FFLAGS ?= -O3 -fopenmp

# GPU build (NVIDIA HPC SDK; -cuda enables the CUF/CUF_TC tendency kernels):
#   make FC=nvfortran FFLAGS="-O3 -cuda -acc=gpu -gpu=cc120 -Minfo=accel"   # RTX PRO 6000 Blackwell
#   make FC=nvfortran FFLAGS="-O3 -cuda -acc=gpu -gpu=cc90 -Minfo=accel"    # GH200

TARGET = scale-dg_extraction
OBJS   = mod_common.o         \
         mod_mesh.o                \
		 mod_dg_optr_kernel_opt1.o \
		 mod_dg_optr_kernel.o      \
		 mod_advect3d_eq_cuf.o     \
		 mod_advect3d_eq.o         \
		 main.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $^

%.f90 : %.F90.erb
	erb $< > $@

.SUFFIXES:
.SUFFIXES: .o .f90 .c .erb .mod

mod_dg_optr_kernel_opt1.f90: mod_dg_optr_kernel_opt1.F90.erb


%.o: %.f90
	$(FC) $(FFLAGS) -c $<

%.o: %.F90
	$(FC) $(FFLAGS) -c $<


# Dependency
mod_mesh.o: mod_common.o
mod_dg_optr_kernel_opt1.o: mod_common.o
mod_dg_optr_kernel.o: mod_common.o mod_dg_optr_kernel_opt1.o
mod_advect3d_eq_cuf.o: mod_common.o
mod_advect3d_eq.o: mod_common.o mod_dg_optr_kernel.o mod_advect3d_eq_cuf.o
main.o: mod_mesh.o mod_advect3d_eq.o

clean:
	rm -f $(TARGET) *.o *.mod
