#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=4
#SBATCH --mem=250gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=out/error_%j.log
#SBATCH --output=out/output_%j.log

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

workspace_path=/hkfs/work/workspace_haic/scratch/xo8179-neural_lam/data/data/
dataset_name_full=global_era5_1980_2022/
dataset_name_3deg=global_era5_1980_2022_3deg/
dataset_name_3deg_short=global_era5_2000_2022_3deg/
zarr_name=fields.zarr

cd ~/neural_lam/neural-lam

ml devel/cuda/11.8
echo "Loaded Cuda"

source ~/miniconda3/etc/profile.d/conda.sh

conda activate regridding

python download_era5.py --out ${workspace_path}${dataset_name_full}

python regrid.py \
    --input ${workspace_path}${dataset_name_full} \
    --out ${workspace_path}${dataset_path_3deg}${zarr_name} \
    --out_short ${workspace_path}${dataset_name_3deg_short}${zarr_name}

conda deactivate

source venv/bin/activate
echo "Activated python"

python pretraining.py \
    --dataset ${dataset_name_3deg_short} \
    --dataset_path ${workspace_path} \
    --graph global_multilevel_2000_2022 \
    --hierarchical 0 \
    --splits 4 \
    --levels 4 \
    --periods "train:2000-01-01,2018-12-31;val:2019-01-01,2020-12-31;test:2021-01-01,2022-12-31"