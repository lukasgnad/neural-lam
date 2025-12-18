#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=accelerated
#SBATCH --ntasks-per-node=1
#SBATCH --mem=250gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/era5_1990_2020_6h-128x64_z64/logs/error_test_2016-2019_3_2046_2049_3_forecasts.log
#SBATCH --output=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/era5_1990_2020_6h-128x64_z64/logs/output_test_2016-2019_3_2046_2049_3_forecasts.log
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

name=1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_forecasts
name_full=${name}_z64


dataset_path=$(ws_find neural_lam)/data/data
dataset_historical_name=global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_test_only
dataset_future_name=global_nextgems_2046_2049_equiangular_with_poles_conservative


cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam
source venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"

batch_size=32
hidden_dim=64
dataset_type=nextgems
graph_name=global_multilevel_era5



# Eval checkpoint 3 WITH forecasts    
srun -n1 python -u train_model.py\
    --name ${name_full}_past\
    --dataset ${dataset_historical_name}\
    --dataset_path ${dataset_path}\
    --dataset_type ${dataset_type}\
    --model graphcast\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim ${hidden_dim}\
    --processor_layers 4\
    --batch_size ${batch_size}\
    --graph ${graph_name}\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_03/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2016-01-01,2019-12-31"\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_forecasts_past\
    --save_forecasts 1\
    --forecast_save_name _era5\
    --save_levels 50,100,150,200,250,300,400,500,600,700,850,925,1000\
    --eval test




# Eval checkpoint 3 WITH forecasts, on 2049
srun -n1 python -u train_model.py\
    --name ${name_full}_future\
    --dataset ${dataset_future_name}\
    --dataset_path ${dataset_path}\
    --dataset_type ${dataset_type}\
    --model graphcast\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim ${hidden_dim}\
    --processor_layers 4\
    --batch_size ${batch_size}\
    --graph ${graph_name}\
    --load trained_models/era5_1990_2020_6h-128x64_z64/checkpoints/1990_2020_6h_128_64_z64_era5_03/last.ckpt\
    --periods "train:2046-01-01,2049-12-31;val:2046-01-01,2049-12-31;test:2046-01-01,2049-12-31"\
    --wandb_output trained_models/era5_1990_2020_6h-128x64_z64/test_runs/wandb_1990_2020_6h_128_64_z64_era5_graphcast_ON_NEXTGEMS_forecasts_future\
    --save_forecasts 1\
    --forecast_save_name _era5\
    --save_levels 50,100,150,200,250,300,400,500,600,700,850,925,1000\
    --eval test

