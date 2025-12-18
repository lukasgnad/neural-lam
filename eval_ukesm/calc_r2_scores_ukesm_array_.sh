#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=dev_cpuonly
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=150gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/logs/error_r2_scores_%a.log
#SBATCH --output=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm/logs/output_r2_scores_%a.log
#SBATCH --account=hk-project-p0024498


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



cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam
source venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"

cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/eval_ukesm


echo "Running model index: $MODEL_ID"


srun --ntasks=1 python -u scores.py --model 0 --past 1 --score rmse

# srun --ntasks=1 python -u scores.py --model $MODEL_ID --past 0 --score r2