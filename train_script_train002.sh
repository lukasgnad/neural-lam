#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=out/error_%j.log
#SBATCH --output=out/output_%j.log
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

source /home/hk-project-pai00005/xo8179/neural_lam_fork/venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"
cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam
echo "Current working directory:"
pwd
echo ""
echo "----------------------------------------------"
echo "Running python file with following command:"
echo "srun python train_model.py --name train002 --model graphcast --graph global_multilevel_train002 --dataset global_era5_train001 --dataset_path /hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/ --epochs 200
"
echo "----------------------------------------------"
echo ""
srun python train_model.py --name train002 --model graphcast --graph global_multilevel_train002 --dataset global_era5_train001 --dataset_path /hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/ --epochs 200

