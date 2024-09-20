#!/bin/bash

#SBATCH --job-name=SSS
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=100:00:00
#SBATCH --mem-per-cpu=1000
#SBATCH --output=outfileSSS.txt

module load intel/compiler/latest
module load intel/mpi/latest
module load intel/compiler-rt/2021.3.0

# Run the Python script with SLURM
srun python3 main_server.py
