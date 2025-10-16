#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --ntasks-per-node=4
#SBATCH --mem=250gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/logs/error_test_01.log
#SBATCH --output=/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64/logs/output_test_01.log
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

name=nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative
name_full=${name}_z64

python train_model.py\
    --name ${name_full}_test_01\
    --dataset global_${name}\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems_withpoles_conservative\
    --load trained_models/${name_full}/checkpoints/checkpoint_01/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_01\
    --eval test

python train_model.py\
    --name ${name_full}_test_02\
    --dataset global_${name}\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 16\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems_withpoles_conservative\
    --load trained_models/${name_full}/checkpoints/checkpoint_02/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_02\
    --eval test

python train_model.py\
    --name ${name_full}_test_03\
    --dataset global_${name}\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model graphcast\
    --n_workers 4\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems_withpoles_conservative\
    --load trained_models/${name_full}/checkpoints/checkpoint_03/last.ckpt\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_03\
    --eval test

python train_model.py\
    --name ${name_full}_test_persistence\
    --dataset global_${name}\
    --dataset_path $(ws_find neural_lam)/data/data\
    --model persistence\
    --n_workers 50\
    --n_example_pred 0\
    --eval_leads 40\
    --hidden_dim 64\
    --processor_layers 4\
    --batch_size 1\
    --graph global_multilevel_nextgems_withpoles_conservative\
    --periods "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"\
    --wandb_output trained_models/${name_full}/test_runs/wandb_test_persistence\
    --eval test