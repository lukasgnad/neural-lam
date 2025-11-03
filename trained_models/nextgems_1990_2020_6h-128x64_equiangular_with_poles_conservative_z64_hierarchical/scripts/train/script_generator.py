import os
from textwrap import dedent

# === Central variables ===
run_name = "nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative_z64_hierarchical"
dataset_name = (
    "global_nextgems_1990_2020_6h-128x64_equiangular_with_poles_conservative"
)
dataset_path = "$(ws_find neural_lam)/data/data"
dataset_type = "nextgems"
graph_name = "global_hierarchical_nextgems_withpoles_conservative"
model_name = "graph_fm"
tmp_path = "$TMPDIR"
hidden_dim = 64
epochs_first, epochs_second, epochs_third = 70, 20, 20
lr_first, lr_second, lr_third = "1e-3", "1e-4", "1e-4"
unroll_first, unroll_second, unroll_third = 1, 4, 8
periods = "train:1990-01-01,2015-12-31;val:2016-01-01,2017-12-31;test:2018-01-01,2019-12-31"

# === Paths for logs ===
base_logdir = f"/home/hk-project-pai00005/xo8179/neural_lam_fork/neural-lam/trained_models/{run_name}/logs"
execution_time_hours = 30


# === SLURM header (same for all) ===
def header(ex_time, run):
    return dedent(
        f"""\
    #!/bin/bash -x
    #SBATCH --nodes=1
    #SBATCH --time={ex_time}:00:00
    #SBATCH --gres=gpu:4
    #SBATCH --partition=accelerated
    #SBATCH --ntasks-per-node=4
    #SBATCH --mem=250gb
    #SBATCH --mail-type="END"
    #SBATCH --mail-user="xo8179@partner.kit.edu"
    #SBATCH --error={base_logdir}/error_train_{run:02d}.log
    #SBATCH --output={base_logdir}/output_train_{run:02d}.log
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

    # --- Variables
    run_name={run_name}
    dataset_type={dataset_type}
    dataset_name={dataset_name}
    dataset_path={dataset_path}
    graph_name={graph_name}
    model_name={model_name}
    tmp_path={tmp_path}
    hidden_dim={hidden_dim}
    epochs_first={epochs_first}
    epochs_second={epochs_second}
    epochs_third={epochs_third}
    lr_first={lr_first}
    lr_second={lr_second}
    lr_third={lr_third}
    unroll_first={unroll_first}
    unroll_second={unroll_second}
    unroll_third={unroll_third}
    periods="{periods}"
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
    """
    )


# === Step-specific templates ===
steps = {
    "step1": dedent(
        """\
        echo "----------------------------------------------"
        echo "1/3 - Running python file with following command:"
        echo "srun python train_model.py \\\\"
        echo "    --name ${run_name}_01 \\\\"
        echo "    --model ${model_name} \\\\"
        echo "    --graph ${graph_name} \\\\"
        echo "    --dataset ${dataset_name} \\\\"
        echo "    --dataset_path ${tmp_path} \\\\"
        echo "    --epochs ${epochs_first} \\\\"
        echo "    --ar_steps ${unroll_first} \\\\"
        echo "    --periods ${periods} \\\\"
        echo "    --lr ${lr_first} \\\\"
        echo "    --hidden_dim ${hidden_dim}"
        echo "GPU state before 1st run:"
        nvidia-smi
        echo "----------------------------------------------"
        echo ""
        srun python -u train_model.py \\
            --name ${run_name}_01 \\
            --model ${model_name} \\
            --graph ${graph_name} \\
            --dataset ${dataset_name} \\
            --dataset_path ${tmp_path} \\
            --epochs ${epochs_first} \\
            --ar_steps ${unroll_first} \\
            --periods ${periods} \\
            --lr ${lr_first} \\
            --hidden_dim ${hidden_dim} \\
            --dataset_type ${dataset_type}
        """
    ),
    "step2": dedent(
        """\
        echo "----------------------------------------------"
        echo "2/3 - Running python file with the following command:"
        echo "srun python train_model.py \\\\"
        echo "    --name ${run_name}_02 \\\\"
        echo "    --model ${model_name} \\\\"
        echo "    --graph ${graph_name} \\\\"
        echo "    --dataset ${dataset_name} \\\\"
        echo "    --dataset_path ${tmp_path} \\\\"
        echo "    --epochs ${epochs_second} \\\\"
        echo "    --ar_steps ${unroll_second} \\\\"
        echo "    --load saved_models/${run_name}_01/last.ckpt \\\\"
        echo "    --periods ${periods} \\\\"
        echo "    --lr ${lr_second} \\\\"
        echo "    --hidden_dim ${hidden_dim}"
        echo "GPU state before 2nd run:"
        nvidia-smi
        echo "----------------------------------------------"
        echo ""
        srun python -u train_model.py \\
            --name ${run_name}_02 \\
            --model ${model_name} \\
            --graph ${graph_name} \\
            --dataset ${dataset_name} \\
            --dataset_path ${tmp_path} \\
            --epochs ${epochs_second} \\
            --ar_steps ${unroll_second} \\
            --load saved_models/${run_name}_01/last.ckpt \\
            --periods ${periods} \\
            --lr ${lr_second} \\
            --hidden_dim ${hidden_dim} \\
            --dataset_type ${dataset_type}
        """
    ),
    "step3": dedent(
        """\
        echo "----------------------------------------------"
        echo "3/3 - Running python file with the following command:"
        echo "srun python train_model.py \\\\"
        echo "    --name ${run_name}_03 \\\\"
        echo "    --model ${model_name} \\\\"
        echo "    --graph ${graph_name} \\\\"
        echo "    --dataset ${dataset_name} \\\\"
        echo "    --dataset_path ${tmp_path} \\\\"
        echo "    --epochs ${epochs_third} \\\\"
        echo "    --ar_steps ${unroll_third} \\\\"
        echo "    --load saved_models/${run_name}_02/last.ckpt \\\\"
        echo "    --periods ${periods} \\\\"
        echo "    --lr ${lr_third} \\\\"
        echo "    --hidden_dim ${hidden_dim}"
        echo "GPU state before 3rd run:"
        nvidia-smi
        echo "----------------------------------------------"
        echo ""
        srun python -u train_model.py \\
            --name ${run_name}_03 \\
            --model ${model_name} \\
            --graph ${graph_name} \\
            --dataset ${dataset_name} \\
            --dataset_path ${tmp_path} \\
            --epochs ${epochs_third} \\
            --ar_steps ${unroll_third} \\
            --load saved_models/${run_name}_02/last.ckpt \\
            --periods ${periods} \\
            --lr ${lr_third} \\
            --hidden_dim ${hidden_dim} \\
            --dataset_type ${dataset_type}
        """
    ),
}

# === Write out the files ===
for i, (step_name, body) in enumerate(steps.items(), start=1):
    filename = f"train_ng_64_hierarch_{i:02d}.slurm"
    execution_time = int(
        execution_time_hours * 0.8 if i == 2 else execution_time_hours
    )
    with open(filename, "w") as f:
        f.write(header(execution_time, i) + "\n" + body + "\n")
    print(f"Written: {filename}")
