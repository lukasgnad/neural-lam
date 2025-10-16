#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --partition=dev_cpuonly
#SBATCH --ntasks-per-node=50
#SBATCH --mem=200gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=out/experimental_error_%j.log
#SBATCH --output=out/experimental_output_%j.log
#SBATCH --account=hk-project-p0024498

#exec > out/log_${SLURM_JOB_ID}.log 2>&1

echo "===== SLURM Job Info ====="
echo "Job ID:           $SLURM_JOB_ID"
echo "Job Name:         $SLURM_JOB_NAME"
echo "User:             $SLURM_JOB_USER"
echo "Partition:        $SLURM_JOB_PARTITION"
echo "Nodes allocated:  $SLURM_NNODES"
echo "Node list:        $SLURM_NODELIST"
echo "CPUs per node:    $SLURM_JOB_CPUS_PER_NODE"
echo "GPUs allocated:   $SLURM_JOB_GPUS"
echo "Tasks per node:   $SLURM_NTASKS_PER_NODE"
echo "Memory per Node:  $SLURM_MEM_PER_NODE"
echo "==========================="

source /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"
cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam
echo "Current working directory:"
pwd
echo ""
echo "----------------------------------------------"
echo "Running python file"

python -u pretraining.py --dataset global_era5_1990_2020_6h-128x64_equiangular_with_poles_conservative --dataset_path /hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/  --graph global_multilevel_era5 --hierarchical 0 --splits 4 --levels 4 --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31" --n_workers 40