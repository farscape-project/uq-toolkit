#!/bin/bash
#SBATCH -J coolant
#SBATCH -p scarf
#SBATCH --ntasks-per-node=1
#SBATCH -N 1
#SBATCH -t 05:00:00
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.err


module list

./run.sh


