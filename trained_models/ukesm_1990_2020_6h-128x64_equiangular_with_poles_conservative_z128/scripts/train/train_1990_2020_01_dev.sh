#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=dev_accelerated
#SBATCH --ntasks-per-node=4
#SBATCH --mem=250gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/ukesm_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/logs/error_train_01_dev.log
#SBATCH --output=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/ukesm_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128/logs/output_train_01_dev.log
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

# --- Variables
run_name=ukesm_1990_2020_6h-128x64_equiangular_with_poles_conservative_z128
dataset_name=global_1990_2019_equiangular_wp_conservative
dataset_path=/hkfs/work/workspace/scratch/xo8179-ukesm/
graph_name=global_multilevel_ukesm_withpoles_conservative
model_name=graphcast
hidden_dim=128
epochs_first=1

lr_first=1e-3
lr_second=1e-4
lr_third=1e-4
unroll_first=1
unroll_second=4
unroll_third=8
periods="train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"
# --- Variables

cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam
source venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"


echo "Current working directory:"
pwd
echo ""
echo "----------------------------------------------"
echo "1/3 - Running python file with following command:"
echo "srun python train_model.py \\"
echo "    --name ${run_name}_01 \\"
echo "    --model ${model_name} \\"
echo "    --graph ${graph_name} \\"
echo "    --dataset ${dataset_name} \\"
echo "    --dataset_path ${dataset_path} \\"
echo "    --epochs ${epochs_first} \\"
echo "    --ar_steps ${unroll_first} \\"
echo "    --periods ${periods} \\"
echo "    --lr ${lr_first} \\"
echo "    --hidden_dim ${hidden_dim}"
echo "GPU state before 1st run:"
nvidia-smi
echo "----------------------------------------------"
echo ""
srun python -u train_model.py \
    --name ${run_name}_01 \
    --model ${model_name} \
    --graph ${graph_name} \
    --dataset ${dataset_name} \
    --dataset_path ${dataset_path} \
    --epochs ${epochs_first} \
    --ar_steps ${unroll_first} \
    --periods ${periods} \
    --lr ${lr_first} \
    --hidden_dim ${hidden_dim}