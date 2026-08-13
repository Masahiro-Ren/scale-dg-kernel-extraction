# SCALE-DG Kernel Extraction

## Overview

This repository provides a standalone extraction of the computational kernels used in the dynamical core of **SCALE-DG**, developed as part of the [**FE-Project**](https://github.com/ywkawai/FE-Project).

The original SCALE-DG code is a full atmospheric model including MPI parallelization, mesh management, I/O, and various physical parameterizations.
For discussions on computational performance, however, these surrounding components often obscure the essential numerical kernels.
This repository therefore extracts only the computational kernels required to perform a three-dimensional discontinuous Galerkin (DG) advection calculation on a structured hexahedral mesh with a simple cubic computational domain.

The objective is to provide a compact and self-contained code base that can be easily analyzed and optimized by computational scientists and CPU/GPU architecture developers.

## Features

Compared with the original FE-Project / SCALE-DG implementation,

- MPI communication is removed.
- NetCDF output is removed.
- SCALE libraries and external dependencies are removed.
- Only a minimal set of source files is retained.
- The DG computational kernels and memory access patterns are preserved as much
  as possible.

In particular,

- tensor-product differentiation,
- lifting operator,
- numerical flux evaluation,
- VMapM / VMapP based indirect addressing,
- halo-buffer based face-node access

follow the implementation used in the original SCALE-DG dynamical core.


## Purpose

The primary purpose of this repository is to facilitate discussions on

- CPU optimization,
- GPU implementation,
- memory layout,
- cache efficiency,
- SIMD/SIMT execution,
- programming models,
- compiler optimization,

without requiring users to build the entire FE-Project framework.

This repository should be regarded as a research kernel extracted from the
original SCALE-DG dynamical core rather than as an independent numerical model.


### Relation to FE-Project

The numerical algorithms implemented here originate from the FE-Project and　the SCALE-DG dynamical core based.
The DG operators (`operator/p*.dat`) are generated directly from FE-Project using an export utility so that the standalone implementation uses identical reference operators.


## Directory structure

```
(TOPDIR)/
    main.f90
    mod_common.f90
    mod_mesh.f90
    mod_advect3d_kernel.f90
    input.conf
    operator/
```


## Building

The provided `Makefile` is intentionally kept simple and is expected to be adapted to the target compilation environment if necessary.

Typical compilation is

```bash

make

```

Users may modify the compiler (`FC`) and compiler options (`FFLAGS`) in the `Makefile` to match their development environment.


## Running

The simulation is executed as

```bash
./advect3d input.conf
```

Simulation parameters are specified in `input.conf`, including

- mesh resolution (`NeX`, `NeY`, `NeZ`),
- polynomial order (`PolyOrder`),
- time step (`dt`),
- number of time steps (`nstep`), and
- constant advection velocity.

By changing the mesh resolution and polynomial order, users can easily evaluate the computational kernels for different problem sizes.


## Citation and acknowledgement

If this repository contributes to published research, software, or performance studies, we kindly request that appropriate acknowledgement be given to the original FE-Project / SCALE-DG development.

Depending on the scope of your work, please consider citing one or more of the following publications.

### General references for FE-Project / SCALE-DG

The following papers describe the numerical formulation, software design, and overall architecture of SCALE-DG.

- Kawai, Y. and Tomita, H. (2023):
  *Numerical Accuracy Necessary for Large-Eddy Simulation of Planetary Boundary Layer Turbulence using Discontinuous Galerkin Method*, Monthly Weather Review, **151**(6), 1479–1508.

- Kawai, Y. and Tomita, H. (2025):
  *Development of a High-Order Global Dynamical Core Using the Discontinuous Galerkin Method for an Atmospheric Large-Eddy Simulation (LES) and Proposal of Test Cases: SCALE-DG v0.8.0*, Geoscientific Model Development, **18**, 725–762.

### Computational performance
The following papers discuss computational performance analysis and　optimization of SCALE-DG kernels.

- Ren, X., Kawai, Y., Tomita, H., Nishizawa, S., Katagiri, T., Hoshino, T., Mukunoki, D., Kawai, M., and Nagai, T. (2025):
  *Performance Evaluation of Loop Body Splitting for Fast Modal Filtering in SCALE-DG on A64FX*, Proceedings of the 2025 International Conference on High Performance Computing in Asia-Pacific Region Workshops (HPCAsia 2025 Workshops), 36–44.

- Ren, X., Kawai, Y., Hoshino, T., Tomita, H., Katagiri, T., Mukunoki, D., and Nishizawa, S. (2026):
  *Learning-Augmented Performance Model for Tensor Product Factorization in High-Order FEM*, IEEE Access, **14**, 43679–43693.

Alternatively, users may cite the FE-Project software repository or acknowledge that the computational kernels originate from the SCALE-DG dynamical core.
These citations help acknowledge the original scientific and software development on which this kernel extraction is based.
Citation is appreciated but is not required by the software license.