#!/bin/bash

# TODO: adapt to multi-node setting:
#   - requires a dynamic ACCELERATE_CONFIG, notably machine_rank

echo -e "\n<--------------------------------------- launch_script_SD_a100.sh --------------------------------------->\n"

# ------------------------------------------> Variables <------------------------------------------
exp_name=REMOVEME_functional_test # experiment and output folder name; common to all runs in the same experiment

run_name=test_run # wandb run display name

exp_dirs_parent_folder=./experiments
model_configs_folder=./models_configs

num_GPUS=1 # *total* number of processes (i.e. accross all nodes) = number of GPUs

# --------------------------------------> Accelerate config <--------------------------------------
ACCELERATE_CONFIG="
--machine_rank=0
--main_training_function=main
--mixed_precision=fp16
--num_machines=1
--num_processes=${num_GPUS}
--rdzv_backend=static
--same_network
--dynamo_backend=no
"

# ----------------------------------------> Script + args <----------------------------------------
MAIN_SCRIPT=main.py

MAIN_SCRIPT_ARGS="
$1
--exp_output_dirs_parent_folder ${exp_dirs_parent_folder}
--experiment_name ${exp_name}
--run_name ${run_name}
--model_type StableDiffusion
--pretrained_model_name_or_path stabilityai/stable-diffusion-2-1
--components_to_train denoiser class_embedding
--noise_scheduler_config_path ${model_configs_folder}/noise_scheduler/better_SD_config_test.json
--num_inference_steps 100
--train_data_dir /projects/deepdevpath/Thomas/data/BBBC021_comp_conc_nice_phen_high_conc_balanced
--seed 42
--resolution 128
--train_batch_size 64
--eval_batch_size 256
--num_epochs 900
--learning_rate 1e-4
--mixed_precision fp16
--eval_save_model_every_epochs 50
--nb_generated_images 6144
--enable_xformers_memory_efficient_attention
--checkpoints_total_limit 2
--checkpointing_steps 1000
--use_ema
--proba_uncond 0.1
--compute_fid
--compute_isc
--compute_kid
"

# ----------------------------------------> Echo commands <----------------------------------------
echo -e "START TIME: $(date)\n"
echo -e "EXPERIMENT NAME: ${exp_name}\n"
echo -e "RUN NAME: ${run_name}\n"
echo -e "EXP_DIRS_PARENT_FOLDER: ${exp_dirs_parent_folder}\n"
echo -e "ACCELERATE_CONFIG: ${ACCELERATE_CONFIG}\n"
echo -e "MAIN_SCRIPT: ${MAIN_SCRIPT}\n"
echo -e "MAIN_SCRIPT_ARGS: ${MAIN_SCRIPT_ARGS}\n"

# Launch the job
accelerate launch ${ACCELERATE_CONFIG} ${MAIN_SCRIPT} ${MAIN_SCRIPT_ARGS}

exit 0
