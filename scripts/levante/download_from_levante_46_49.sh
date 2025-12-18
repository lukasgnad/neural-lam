#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --time=15:00:00
#SBATCH --partition=cpuonly
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=30gb
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

cd /hkfs/work/workspace/scratch/xo8179-neural_lam/data/data/global_nextgems_2046_2049/
echo 'Starting to copy'
sshpass -p 'e,%R.T.fo:,tX1x<{X(w' scp b383582@levante.dkrz.de:/scratch/b/b383582/nextgems/3D_nextgems_2046_2049_6hourly_0.25deg.nc 3D_nextgems_2046_2049_6hourly_0.25deg.nc
echo 'Copied 3D var successfully'
sshpass -p 'e,%R.T.fo:,tX1x<{X(w' scp b383582@levante.dkrz.de:/scratch/b/b383582/nextgems/2D_nextgems_2046_2049_6hourly_0.25deg.nc 2D_nextgems_2046_2049_6hourly_0.25deg.nc
echo 'Copied 2D var successfully'
sshpass -p 'e,%R.T.fo:,tX1x<{X(w' scp b383582@levante.dkrz.de:/scratch/b/b383582/nextgems/2D_nextgems_2046_2049_tp_6hourly_0.25deg.nc 2D_nextgems_2046_2049_tp_6hourly_0.25deg.nc
echo 'Copied TP successfully'

