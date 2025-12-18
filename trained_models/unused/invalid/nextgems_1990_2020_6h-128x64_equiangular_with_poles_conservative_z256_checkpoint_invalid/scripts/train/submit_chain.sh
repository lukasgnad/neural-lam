#!/bin/bash

jid1=$(sbatch train_01.slurm | awk '{print $4}')
jid2=$(sbatch --dependency=afterok:$jid1 train_02.slurm | awk '{print $4}')
jid3=$(sbatch --dependency=afterok:$jid2 train_03.slurm | awk '{print $4}')

echo "Submitted jobs: $jid1 -> $jid2 -> $jid3"