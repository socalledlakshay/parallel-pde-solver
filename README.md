# parallel-pde-solver

## Overview
A scalable framework for solving scalar hyperbolic PDEs using 
parallel domain decomposition.

## Features
- Solves Linear Transport and Burgers' Equations
- Multiple Riemann solvers: Godunov, Rusanov, Engquist-Osher
- MPI-based domain decomposition via mpi4py
- PETSc integration for automated halo exchanges
- 3.2x speedup on 8-core architecture (N=25,000 mesh)

## Tech Stack
Python, PETSc (petsc4py), MPI (mpi4py), NumPy, Matplotlib

## Results
Achieved 3.2x speedup on 8-core architecture for 
high-resolution meshes (N=25,000).
