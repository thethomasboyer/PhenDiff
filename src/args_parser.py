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
        help="Run the training script in debug mode, ie with: save_model_epochs=1, generate_images_epochs=1, nb_generated_images=eval_batch_size, num_training_steps=10, num_inference_steps=5, checkpoints_total_limit=1, checkpointing_steps=30, kid_subset_size=min(1000, nb_generated_images)",
    )
    parser.add_argument(
        "--components_to_train",
        default=["denoiser", "class_embedding"],
        nargs="+",
        type=str,
        choices=["denoiser", "autoencoder", "class_embedding"],
        help="The components to train.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
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
        required=True,
        help="The name of or path to the pretrained pipeline.",
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
        default=None,
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--use_pytorch_loader",
        default=False,
        action="store_true",
        help="Whether to use the PyTorch ImageFolder loader instead of the HF Dataset loader. Usefull for folder symlinks...",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written. Will be used as the WandB project name!",
    )
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=False,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="The batch size (per GPU) for evaluation.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
            " process."
        ),
    )
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument(
        "--generate_images_epochs",
        type=int,
        default=100,
        help="How often to save images during training.",
    )
    parser.add_argument("--compute_fid", action="store_true")
    parser.add_argument("--compute_isc", action="store_true")
    parser.add_argument("--compute_kid", action="store_true")
    help_msg = "How many images to generate (per class) for metrics computation. "
    help_msg += (
        "Only a fraction of the first batch will be logged; the rest will be lost."
    )
    parser.add_argument("--nb_generated_images", type=int, default=1000, help=help_msg)
    parser.add_argument(
        "--kid_subset_size",
        type=int,
        default=1000,
        help="Change this if generating very few images (for testing purposes only)",
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=100,
        help="How often to save the model during training.",
    )
    parser.add_argument(
        "--guidance_factor",
        type=float,
        help="The scaling factor of the guidance ('ω' in the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf; *not* the same definition that in the Classifier-Free Diffusion Guidance paper!). Set to <= 1 to disable guidance.",
    )
    parser.add_argument(
        "--proba_uncond",
        type=float,
        default=0.1,
        help="The probability of sampling unconditionally instead of conditionally for the CLF.",
    )
    parser.add_argument(
        "--class_embedding_type",
        type=str,
        default="embedding",
        choices=["OHE", "embedding"],
        help="The kind of class embedding to use.",
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
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
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
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
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
        default="epsilon",
        choices=["epsilon", "sample", "velocity"],
        help="Whether the model should predict the 'epsilon'/noise error, directly the reconstructed image 'x0', or the velocity (see https://arxiv.org/abs/2202.00512)",
    )
    parser.add_argument("--num_training_steps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=1000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=5,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    # parse args
    args: Namespace = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
