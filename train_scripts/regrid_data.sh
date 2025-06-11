#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=4:00:00
#SBATCH --partition=cpuonly
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200gb
#SBATCH --mail-type="END"
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

# python create_global_mesh.py \
#     --dataset global_era5_1980_2022 \
#     --dataset_path /hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/ \
#     --graph global_multilevel_1980_2022 \
#     --hierarchical 0 \
#     --splits 4 \
#     --levels 4 \

# python create_global_grid_features.py \
#     --dataset global_era5_1980_2022 \
#     --dataset_path /hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/

# python create_global_forcing.py \
#     --dataset global_era5_1980_2022 \
#     --dataset_path /hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/

source ~/miniconda3/etc/profile.d/conda.sh

conda activate regridding

python regrid.py

conda deactivate

source venv/bin/activate
echo "Activated python"

python pretraining.py --dataset global_era5_2000_2022_3deg --dataset_path /hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/ --graph global_multilevel_2000_2022 --hierarchical 0 --splits 4 --levels 4 --periods "train:2000-01-01,2018-12-31;val:2019-01-01,2020-12-31;test:2021-01-01,2022-12-31"