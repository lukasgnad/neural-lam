#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --partition=dev_cpuonly
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

# dataset=global_era5_1980_2022_6h-128x64_equiangular_with_poles_conservative
# dataset=global_1990_2019_equiangular_wp_conservative
dataset=global_1985_2014_equiangular_wp_conservative_40_chunks
dataset_path=/hkfs/work/workspace/scratch/xo8179-ukesm/

periods="train:1985-01-01,2010-12-31;val:2011-01-01,2012-12-31;test:2011-01-01,2014-12-31"
# periods="train:2046-01-01,2049-12-31;val:2046-01-01,2049-12-31;test:2046-01-01,2049-12-31"

graph_name="global_multilevel_ukesm_withpoles_conservative"

cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam
source venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"

python create_global_mesh.py \
    --dataset ${dataset} \
    --dataset_path ${dataset_path} \
    --graph ${graph_name} \
    --hierarchical 0 \
    --splits 4 \
    --levels 4

python create_global_grid_features.py \
    --dataset ${dataset} \
    --dataset_path ${dataset_path}

# python create_global_forcing.py \
python create_global_forcing_stable.py \
    --dataset ${dataset} \
    --dataset_path ${dataset_path} \
    --dataset_type ukesm

python create_parameter_weights.py \
    --dataset ${dataset} \
    --dataset_path ${dataset_path} \
    --periods ${periods} \
    --n_workers 16 \
    --dataset_type ukesm