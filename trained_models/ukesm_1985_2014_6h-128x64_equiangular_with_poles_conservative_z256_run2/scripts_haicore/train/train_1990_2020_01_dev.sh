#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --mem=250gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=/hkfs/work/workspace_haic/scratch/xo8179-neural_lam_copy/neural-lam/trained_models/ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256/logs/error_train_01_dev.log
#SBATCH --output=/hkfs/work/workspace_haic/scratch/xo8179-neural_lam_copy/neural-lam/trained_models/ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z256/logs/output_train_01_dev.log

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
dataset_type=ukesm
run_name=ukesm_1985_2014_6h-128x64_equiangular_with_poles_conservative_z64
dataset_name=global_1985_2014_equiangular_wp_conservative_40_chunks
dataset_path=/hkfs/work/workspace/scratch/xo8179-ukesm/
graph_name=global_multilevel_ukesm_withpoles_conservative
model_name=graphcast
tmp_path=$TMPDIR
hidden_dim=256
epochs_first=1
epochs_second=20
epochs_third=20
batch_size=4

lr_first=1e-3
lr_second=1e-4
lr_third=1e-4
unroll_first=1
unroll_second=4
unroll_third=8
periods="train:1985-01-01,2010-12-30;val:2011-01-01,2012-12-30;test:2013-01-01,2014-12-30"
# --- Variables


cd /hkfs/work/workspace_haic/scratch/xo8179-neural_lam_copy/neural-lam
source venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"

python -c 'import os; print("RANK",os.environ.get("RANK"), "LOCAL_RANK",os.environ.get("LOCAL_RANK"), "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))'


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
srun -n1 python -u train_model.py \
    --name ${run_name}_01_dev \
    --model ${model_name} \
    --graph ${graph_name} \
    --dataset ${dataset_name} \
    --dataset_path ${dataset_path} \
    --epochs ${epochs_first} \
    --ar_steps ${unroll_first} \
    --periods ${periods} \
    --lr ${lr_first} \
    --hidden_dim ${hidden_dim} \
    --dataset_type ${dataset_type} \
    --batch_size ${batch_size}
