#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:1
#SBATCH --partition=accelerated
#SBATCH --ntasks-per-node=4
#SBATCH --mem=250gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/logs/error_test_2016-2019_3_2046_2049_3_forecasts.log
#SBATCH --output=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z256/logs/output_test_2016-2019_3_2046_2049_3_forecasts.log
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

name=nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative
name_full=${name}_z256

dataset_path=$(ws_find neural_lam)/data/data
dataset_historical_name=global_${name}_test_only
dataset_future_name=global_nextgems_2046_2049_equiangular_with_poles_conservative

cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam
source venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"




# Eval checkpoint 3 WITH forecasts    
# srun python train_model.py\
#     --name ${name_full}_test_03_forecasts_long\
#     --dataset ${dataset_historical_name}\
#     --dataset_path ${dataset_path}\
#     --model graphcast\
#     --n_example_pred 0\
#     --eval_leads 40\
#     --hidden_dim 256\
#     --processor_layers 4\
#     --batch_size 8\
#     --graph global_multilevel_nextgems_withpoles_conservative\
#     --load trained_models/${name_full}/checkpoints/checkpoint_03/last.ckpt\
#     --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2016-01-01,2019-12-31"\
#     --wandb_output trained_models/${name_full}/test_runs/wandb_test_03_forecasts_long\
#     --save_forecasts 1\
#     --save_levels 500,700,850\
#     --eval test

# Eval checkpoint 3 WITH forecasts, on 2049
srun python train_model.py\
    --name ${name_full}_test_03_2049_forecasts_long\
    --dataset ${dataset_future_name}\
    --dataset_path ${dataset_path}\
    --model graphcast\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 256\
    --processor_layers 4\
    --batch_size 8\
    --graph global_multilevel_nextgems_withpoles_conservative\
    --load trained_models/${name_full}/checkpoints/checkpoint_03/last.ckpt\
    --periods "train:2046-01-01,2049-12-31;val:2046-01-01,2049-12-31;test:2046-01-01,2049-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_03_2049_forecasts_long\
    --save_forecasts 1\
    --save_levels 500,700,850\
    --eval test

