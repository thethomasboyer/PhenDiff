# Copyright 2023 The HuggingFace Team and Thomas Boyer. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from argparse import Namespace


def parse_args() -> Namespace:
    # define parser
    parser = argparse.ArgumentParser(description="The main training script.")

    # define args
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Run the training script in debug mode, ie with: eval_save_model_every_{epochs,opti_steps}={1,10}, nb_generated_images=eval_batch_size, num_train_timesteps=10, num_inference_steps=5, checkpoints_total_limit=1, checkpointing_steps=30, kid_subset_size=min(1000, nb_generated_images)",
    )
    parser.add_argument(
        "--model_type", type=str, choices=["DDIM", "StableDiffusion"], required=True
    )
    parser.add_argument(
        "--components_to_train",
        nargs="+",
        type=str,
        choices=["denoiser", "autoencoder", "class_embedding"],
        help="The components to train.",
        required=True,
    )
    parser.add_argument(
        "--attention_fine_tuning",
        default=False,
        action="store_true",
        help="Whether to fine-tune only the attention layers or not.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="The split to use. Pass an empty string if there is no split in the dataset structure.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        help="The name of or path to the pretrained pipeline. Must not be set if denoiser_config_path or noise_scheduler_config_path is set.",
    )
    parser.add_argument(
        "--denoiser_config_path",
        type=str,
        help="The path to the denoiser config. Must not be set if pretrained_model_name_or_path is set.",
    )
    parser.add_argument(
        "--noise_scheduler_config_path",
        type=str,
        help="The path to the noise scheduler config. Must not be set if pretrained_model_name_or_path is set.",
    )
    parser.add_argument(
        "--learn_denoiser_from_scratch",
        default=False,
        action="store_true",
        help="Wether or not to use the weights of the pretrained denoiser ('unet').",
    )
    parser.add_argument(
        "--revision",
        type=str,
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--perc_samples",
        type=float,
        help="The percentage of samples (∈ ]0; 100]) to use from the training dataset *inside each class*.",
    )
    parser.add_argument(
        "--data_aug_on_the_fly",
        action="store_true",
        default=True,
        help="Whether to apply data augmentation in the data loader or not. Only a horizontal & vertical random flip is applied if True.",
    )
    parser.add_argument(
        "--compute_metrics_full_dataset",
        action="store_true",
        default=True,
        help="Whether to compute the metrics w.r.t. the full nonsubsampled dataset, or on the training dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="A seed to use, for the base random library only, to subsample the dataset if perc_samples is not None (or 100). Allows to resume runs with the same subset.",
    )
    parser.add_argument(
        "--use_pytorch_loader",
        default=True,
        action="store_true",
        help="Whether to use the PyTorch ImageFolder loader instead of the HF Dataset loader. Usefull for folder symlinks...",
    )
    parser.add_argument(
        "--exp_output_dirs_parent_folder",
        type=str,
        required=True,
        help="The common parent directory of all the experiment-specific folders where the files/folders common to all experiments will be saved (e.g. Inceptionv3 checkpoints for FID computation).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        required=True,
        help="The name of the Weights & Biases entity",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="The name of the experiment. Will be used as the name of the experiment-specific folder (where the predictions and checkpoints will be written) and as the WandB project name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="The wandb run name.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        required=True,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        required=True,
        help="The batch size (per GPU) for evaluation.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument(
        "--max_num_epochs",
        type=int,
        help="Number of epochs to train for. Must be provided if --num_steps is not.",
    )
    parser.add_argument(
        "--max_num_steps",
        type=int,
        help="Number of optimizer steps to train for. Must be provided if --num_epochs is not.",
    )
    parser.add_argument(
        "--eval_save_model_every_epochs",
        type=int,
        help="Evaluate and save (if --main_metric is the best recorded to date) the model during training every x epochs. Either this or --eval_save_model_every_opti_steps must be provided.",
    )
    parser.add_argument(
        "--eval_save_model_every_opti_steps",
        type=int,
        help="Evaluate and save (if --main_metric is the best recorded to date) the model during training every x optimization steps. Either this or --eval_save_model_every_epochs must be provided.",
    )
    parser.add_argument(
        "--precise_first_n_epochs",
        type=int,
        help="Whether to evaluate the model every epoch during the first n epochs. Ignored if None.",
    )
    parser.add_argument("--compute_fid", action="store_true", default=True)
    parser.add_argument("--compute_isc", action="store_true", default=True)
    parser.add_argument("--compute_kid", action="store_true", default=True)
    help_msg = "How many images to generate (per class) for metrics computation. "
    help_msg += (
        "Only a fraction of the first batch will be logged; the rest will be lost."
    )
    parser.add_argument("--nb_generated_images", type=int, required=True, help=help_msg)
    parser.add_argument(
        "--kid_subset_size",
        type=int,
        default=1000,
        help="Change this if generating very few images (<1000, for testing purposes only)",
    )
    parser.add_argument(
        "--guidance_factor",
        type=float,
        help=(
            "The scaling factor of the guidance. It corresponds to 'ω' in the Imagen paper (https://arxiv.org/pdf/2205.11487.pdf). Note that different models might have other implementations."
        ),
    )
    parser.add_argument(
        "--proba_uncond",
        type=float,
        default=0.1,
        help="The probability of sampling unconditionally instead of conditionally for the CLF. Set to 1 for unconditional generation only.",
    )
    parser.add_argument(
        "--class_embedding_dim",
        type=int,
        default=1024,
        help="The dimension of the class embedding.",
    )
    # TODO: To be used if testing img2img while training
    # parser.add_argument(
    #     "--denoising_starting_point",
    #     type=float,
    #     help="The starting point of the denoising schedule (between 0 and 1).",
    # )
    # TODO: allow gradient accumulation back (deactivated for now because multiple models cannot be passed to accumulate();
    # see https://github.com/huggingface/accelerate/issues/668 for solutions; HF team appears to be working on it 🥰;
    # an easy fix would be to wrap everything inside a super nn.Module model)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Initial learning rate (after the potential warmup period) to use. Will be multiplied by the square root of the number of processes (as it multiplies total effective batch size).",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.95,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-6,
        help="Weight decay magnitude for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        default=True,
        help="Whether to use Exponential Moving Average for the final model weights.",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="The inverse gamma value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="The power value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="The maximum decay magnitude for EMA.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_private_repo",
        action="store_true",
        help="Whether or not to create a private repository.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb"],
        help=("Only [wandb](https://www.wandb.ai) is supported."),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        choices=["epsilon", "sample", "velocity"],
        help=(
            "Whether the model should predict the 'epsilon'/noise error, directly the reconstructed image 'x0', "
            "or the velocity (see https://arxiv.org/abs/2202.00512). If None will use the prediction type of the pretrained model or of the given config."
        ),
    )
    parser.add_argument(
        "--num_train_timesteps",
        type=int,
        help="If None will use the value of the pretrained model / given config.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        help="If None will use the value of the pretrained model.",
    )
    parser.add_argument(
        "--main_metric",
        type=str,
        default="frechet_inception_distance",
        help="The metric to use to decide whether a model is the best to date and hence should be saved, erasing the previous one. The mean over all classes will be used.",
    )
    parser.add_argument(
        "--beta_schedule",
        type=str,
        help="If None will use the value of the pretrained model.",
    )
    parser.add_argument(
        "--beta_start",
        type=float,
        help="If None will use the value of the pretrained model.",
    )
    parser.add_argument(
        "--beta_end",
        type=float,
        help="If None will use the value of the pretrained model.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        required=True,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        required=True,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    # parse args
    args: Namespace = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
