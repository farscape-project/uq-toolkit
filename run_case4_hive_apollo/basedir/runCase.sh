#!/bin/bash
#SBATCH -J hive-apollo-freqdomain
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -o stdout.out
#SBATCH -e stderr.err
#SBATCH --mail-type=END
#SBATCH --mail-user=josh.williams@stfc.ac.uk
#SBATCH -p preemptable #scarf # queue name
#SBATCH --mem=48G

module load OpenMPI/4.1.1-GCC-11.2.0 Boost/1.77.0-GCC-11.2.0 CMake/3.22.1-GCCcore-11.2.0
module list


APOLLO_DIR=/work4/cse/scarf1324/apollo

input="Thermal.i"

# uses direct solve for complex AForm, which does not work in parallel # mpiexec -n 32 
${APOLLO_DIR}/apollo-opt -w -i SS316LSTC${input} | tee logRun


