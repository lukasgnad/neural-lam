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

dataset=global_2097_2100_equiangular_wp_conservative
dataset_path=/hkfs/work/workspace/scratch/xo8179-ukesm/

periods="train:2097-01-01,2100-12-30;val:2097-01-01,2100-12-30;test:2097-01-01,2100-12-30"

graph_name="global_multilevel_ukesm_withpoles_conservative"

cd /home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam
source venv/bin/activate
echo "Activated python"
ml devel/cuda/11.8
echo "Loaded Cuda"

# This one already exists
# python create_global_mesh.py \
#     --dataset ${dataset} \
#     --dataset_path ${dataset_path} \
#     --graph ${graph_name} \
#     --hierarchical 0 \
#     --splits 4 \
#     --levels 4

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