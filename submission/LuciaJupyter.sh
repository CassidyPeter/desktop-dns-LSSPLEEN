#!/bin/bash

#SBATCH --account=turbodns
#SBATCH --partition=visu
#SBATCH --job-name=jupyterserver
#SBATCH --output=%j_pv.out
#SBATCH --error=%j_pv.err
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --account=turbodns
#SBATCH --mail-user=peter.cassidy@doct.uliege.be
#SBATCH --mail-type=ALL


#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

source /gpfs/home/acad/ulg-desturb/pcassidy/jupyter/utils.sh

export port=$( findPort )


. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded

module load EasyBuild/2024a
module load Anaconda3/2024.06-1
# conda create -n jupyterdns python==3.11.7
echo ""
echo "----------------- Initiating conda env -----------------"
# source $(conda info --base)/etc/profile.d/conda.sh
conda init
conda activate jupyterdns # conda environment already created on HPC

echo "----------------- Initiating Jupyter Server -----------------"
date
echo "port: $port"
jupyter notebook --no-browser --ip="*" --port=$port --notebook-dir="/gpfs/home/acad/ulg-desturb/pcassidy/jupyter"
# /gpfs/home/acad/ulg-desturb/pcassidy/.conda/envs/jupyterdns/bin/python -m jupyter notebook --no-browser --ip="*" --port=$port --notebook-dir="/gpfs/home/acad/ulg-desturb/pcassidy/jupyter"
echo "----------------- Jupyter Server complete -----------------"
date
