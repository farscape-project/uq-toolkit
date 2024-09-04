# HIVE
Repository for scripts and input files for HIVE-relevant simulations and workflows.

# Recommended Software
## Apollo
Required for running MOOSE/MFEM based FE simulations using the input files in `Apollo Inputs` directory.
Available from the [Apollo GitHub repository](https://github.com/aurora-multiphysics/apollo).

Docker build of Apollo (published weekly) can be downloaded via
```
docker pull alexanderianblair/apollo:master
```

## Cubit
Required for running scripts to adjust coil/target position and
remesh solids, using scripts in `Meshing` directory.
Cubit can be downloaded from the [Coreform website](https://coreform.com/products/downloads/) and requires a license.

## VacuumMesher
Required for generating vacuum mesh around solid meshes for coil and target, using scripts in `Meshing` directory.
VacuumMesher can be downloaded from the [VacuumMesher GitHub repository](https://github.com/aurora-multiphysics/VacuumMesher).

# Contributing
Updates to scripts, input files, and workflows in this project are encouraged; please make your changes in a new branch and create a PR to merge the new commits into `main`, including a short description of the additions/changes made.