#!/bin/bash
#SBATCH -J cube_uq
#SBATCH -p scarf
#SBATCH --ntasks-per-node=4
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.err


module list

./runpar.sh


