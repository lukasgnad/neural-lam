#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --partition=cpuonly
#SBATCH --ntasks-per-node=50
#SBATCH --mail-type="END"
#SBATCH --mem=200gb
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=out/error_%j.log
#SBATCH --output=out/output_%j.log
#SBATCH --account=hk-project-p0024498

echo "===== SLURM Job Info ====="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "User:             $SLURM_JOB_USER"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Nodes allocated:  $SLURM_NNODES"
echo "Node list:        $SLURM_NODELIST"
echo "CPUs per node:    $SLURM_JOB_CPUS_PER_NODE"
echo "Tasks per node:   $SLURM_NTASKS_PER_NODE"
echo "Memory per Node:  $SLURM_MEM_PER_NODE"
echo "==========================="

cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam

ml devel/cuda/11.8
echo "Loaded Cuda"


source ~/miniconda3/etc/profile.d/conda.sh

conda activate regridding

python -u regrid_nextgems/regrid_nextgems_atmosphere.py

conda deactivate
