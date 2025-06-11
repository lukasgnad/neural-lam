#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=accelerated
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200gb
#SBATCH --mail-type="END"
#SBATCH --mail-user="xo8179@partner.kit.edu"
#SBATCH --error=out/error_%j.log
#SBATCH --output=out/output_%j.log
#SBATCH --account=hk-project-p0024498

#exec > out/log_${SLURM_JOB_ID}.log 2>&1

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
run_name=train002
dataset_name=global_era5_train001
dataset_path=$(ws_find neural_lam)/data/data
graph_name=global_multilevel_train002
model_name=graphcast
tmp_path=$TMPDIR
epochs=200
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
echo ""
echo "----------------------------------------------"
echo "Running python file with following command:"
echo "srun python train_model.py --name $run_name --model $model_name --graph $graph_name --dataset $dataset_name --dataset_path $tmp_path --epochs $epochs"
echo "----------------------------------------------"
echo ""
srun python train_model.py --name $run_name --model $model_name --graph $graph_name --dataset $dataset_name --dataset_path $tmp_path --epochs $epochs

