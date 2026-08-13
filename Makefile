FC     = gfortran
FFLAGS = -O3 -fopenmp

TARGET = advect3d
OBJS   = mod_common.o \
         mod_mesh.o mod_advect3d_kernel.o main.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $^

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

mod_mesh.o: mod_common.o
mod_advect3d_kernel.o: mod_common.o
main.o: mod_mesh.o mod_advect3d_kernel.o

clean:
	rm -f $(TARGET) *.o *.mod