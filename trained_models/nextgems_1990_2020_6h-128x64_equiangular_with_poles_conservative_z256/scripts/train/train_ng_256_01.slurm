#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --ntasks-per-node=4
#SBATCH --mem=250gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/logs/error_train_01.log
#SBATCH --output=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/logs/output_train_01.log
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
dataset_type=nextgems
run_name=nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256
dataset_name=global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative
dataset_path=$(ws_find neural_lam)/data/data
graph_name=global_multilevel_nextgems_withpoles_conservative
model_name=graphcast
tmp_path=$TMPDIR
hidden_dim=256
epochs_first=70
epochs_second=20
epochs_third=20
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

# Copy data to tmp
start_time=$(date +%s)
cp -r $dataset_path/$dataset_name $tmp_path/$dataset_name
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Copied data in $elapsed seconds."

echo "Current working directory:"
pwd

echo "----------------------------------------------"
echo "1/3 - Running python file with following command:"
echo "srun python train_model.py \\"
echo "    --name ${run_name}_01 \\"
echo "    --model ${model_name} \\"
echo "    --graph ${graph_name} \\"
echo "    --dataset ${dataset_name} \\"
echo "    --dataset_path ${tmp_path} \\"
echo "    --epochs ${epochs_first} \\"
echo "    --ar_steps ${unroll_first} \\"
echo "    --periods ${periods} \\"
echo "    --lr ${lr_first} \\"
echo "    --hidden_dim ${hidden_dim} \\"
echo "    --dataset_type ${dataset_type}"
echo "GPU state before 1st run:"
nvidia-smi
echo "----------------------------------------------"
echo ""
srun python -u train_model.py \
    --name ${run_name}_01 \
    --model ${model_name} \
    --graph ${graph_name} \
    --dataset ${dataset_name} \
    --dataset_path ${tmp_path} \
    --epochs ${epochs_first} \
    --ar_steps ${unroll_first} \
    --periods ${periods} \
    --lr ${lr_first} \
    --hidden_dim ${hidden_dim} \
    --dataset_type ${dataset_type}

