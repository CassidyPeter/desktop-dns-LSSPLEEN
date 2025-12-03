#!/bin/bash

#SBATCH --job-name=coarsesclean
#SBATCH --output=out.out
#SBATCH --error=out.txt
#SBATCH --partition=gpu
## SBATCH --partition=debug-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=60G
## SBATCH --gpus=1
#SBATCH --gpus=4
## SBATCH --time=02:00:00
#SBATCH --time=4:00:00
#SBATCH --account=turbodns
#SBATCH --mail-user=peter.cassidy@doct.uliege.be
#SBATCH --mail-type=ALL

#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
## SBATCH --ntasks=1

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')



. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded

module load nvidia/nvhpc/22.7

application="/gpfs/home/acad/ulg-desturb/pcassidy/binaries/threedns_ccall_cuda118_px"


#! Run options for the application:
options=""

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

mpirun -n 4 $application
