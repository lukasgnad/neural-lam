#!/bin/bash

jid1=$(sbatch train_1985_2014_01.sh | awk '{print $4}')
jid2=$(sbatch --dependency=afterok:$jid1 train_1985_2014_02.sh | awk '{print $4}')
jid3=$(sbatch --dependency=afterok:$jid2 train_1985_2014_03.sh | awk '{print $4}')

echo "Submitted jobs: $jid1 -> $jid2 -> $jid3"