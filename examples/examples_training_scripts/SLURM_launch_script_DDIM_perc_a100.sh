#!/bin/bash

# TODO: adapt to multi-node setting:
#   - requires a dynamic ACCELERATE_CONFIG, notably machine_rank

echo -e "\n<--------------------------------------- launch_script_DDIM_perc_a100.sh --------------------------------------->\n"

# ------------------------------------------> Variables <------------------------------------------
exp_name=experiment_name # experiment name; common to all runs in the same experiment

exp_dirs_parent_folder=${SCRATCH}/comparison-experiments/experiments
model_configs_folder=${HOME}/sources/diffusion-comparison-experiments/models_configs

if [ "$1" == "--debug" ]; then
    launch_command="srun --pty"
    runtime="02:00:00"
    qos="qos_gpu-dev"
    outputs=""
else
    launch_command="sbatch"
    runtime="20:00:00"
    qos="qos_gpu-t3"
fi

# --------------------------------------> Accelerate config <--------------------------------------
if [[ "$num_GPUS" -gt 1 ]]; then
    acc_cfg="--multi_gpu"
else
    acc_cfg=""
fi

ACCELERATE_CONFIG="
${acc_cfg}
--machine_rank=0
--mixed_precision=fp16
--num_machines=1
--num_processes=${num_GPUS}
--rdzv_backend=static
--same_network
--dynamo_backend=no
"

# ----------------------------------------> Echo commands <----------------------------------------
echo -e "START TIME: $(date)\n"
echo -e "EXPERIMENT NAME: ${exp_name}\n"
echo -e "EXP_DIRS_PARENT_FOLDER: ${exp_dirs_parent_folder}\n"
echo -e "ACCELERATE_CONFIG: ${ACCELERATE_CONFIG}\n"

# --------------------------------------> Load compute envs <--------------------------------------
module purge

# Activate µmamba environment
source ${HOME}/.bashrc
micromamba deactivate
micromamba activate -p ${SCRATCH}/micromamba/envs/diffusion-experiments

# ----------------------------------------> Launch scripts <----------------------------------------
percentages=(100 50 10 5 3 1)

for perc in ${percentages[@]}; do

    run_name=${perc}_perc # output folder name

    if [ "$1" != "--debug" ]; then
        run_output_folder=${exp_dirs_parent_folder}/${exp_name}/${run_name}
        mkdir -p ${run_output_folder}
        outputs="--error=${run_output_folder}/jobid-%j.err
--output=${run_output_folder}/jobid-%j.out"
    fi

    num_GPUS=$((7 * perc / 100 + 1)) # *total* number of processes (i.e. accross all nodes) = number of GPUs

    # -----------------------------------------> SLURM params <----------------------------------------
    SBATCH_PARAMS="
    --job-name=${run_name}
    --constraint=a100
    --nodes=1
    --ntasks-per-node=1
    --gres=gpu:${num_GPUS}
    --cpus-per-task=$((64 * ${num_GPUS} / 8))
    --hint=nomultithread 
    --time=${runtime}
    --qos=${qos}
    --account=kio@a100
    --mail-user=yourmail@mailcompany.com
    --mail-type=FAIL
    ${outputs}
    "
    # Recap: (see http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm.html)
    #     QoS     | time limit | GPU limit / job
    #  qos_gpu-t3 |     20h    |	512 GPUs
    #  qos_gpu-t4 |    100h    |	 16 GPUs
    # qos_gpu-dev |      2h    |     32 GPUs
    # Note that qos_gpu-t4 is not available with A100 partitions!

    # ----------------------------------------> Script + args <----------------------------------------
    MAIN_SCRIPT=${HOME}/sources/diffusion-comparison-experiments/train.py

    MAIN_SCRIPT_ARGS="
    $1
    --exp_output_dirs_parent_folder ${exp_dirs_parent_folder}
    --experiment_name ${exp_name}
    --run_name ${run_name}
    --model_type DDIM
    --denoiser_config_path ${model_configs_folder}/denoiser/small_denoiser_config.json
    --noise_scheduler_config_path ${model_configs_folder}/noise_scheduler/3k_steps_clipping_rescaling.json
    --num_inference_steps 100
    --components_to_train denoiser
    --train_data_dir path/to/train/data
    --perc_samples ${perc}
    --seed 1234
    --resolution 128
    --train_batch_size 96
    --eval_batch_size 256
    --max_num_steps 30000
    --learning_rate 1e-4
    --mixed_precision fp16
    --eval_save_model_every_opti_steps 1000
    --nb_generated_images 4096
    --checkpoints_total_limit 3
    --checkpointing_steps 1000
    --use_ema
    --proba_uncond 0.1
    --compute_fid
    --compute_isc
    --compute_kid
    "

    # ----------------------------------------> Echo commands <----------------------------------------
    echo -e "MAIN_SCRIPT: ${MAIN_SCRIPT}\n"
    echo -e "MAIN_SCRIPT_ARGS: ${MAIN_SCRIPT_ARGS}\n"
    echo -e "SBATCH_PARAMS: ${SBATCH_PARAMS}\n"
    echo -e "RUN_NAME: ${run_name}\n"

    final_cmd="WANDB_MODE=offline ${launch_command} ${SBATCH_PARAMS} accelerate launch ${ACCELERATE_CONFIG} ${MAIN_SCRIPT} ${MAIN_SCRIPT_ARGS}"

    if [ "$1" == "--debug" ]; then
        eval ${final_cmd}
    else
        eval ${final_cmd} &
    fi
done

exit 0
