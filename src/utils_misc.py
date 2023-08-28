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

import logging
import os
from argparse import Namespace
from math import ceil
from pathlib import Path
from typing import Optional

import datasets
import diffusers
import numpy as np
import torch
from accelerate.logging import MultiProcessAdapter
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, whoami
from packaging import version


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def split(l, n, idx) -> list[int]:
    """
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

    Should probably be replaced by Accelerator.split_between_processes.
    """
    k, m = divmod(len(l), n)
    l = [l[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]
    return l[idx]


def args_checker(
    args: Namespace, logger: MultiProcessAdapter, first_check_pass: bool = True
) -> None:
    """Performs some checks on the passed arguments."""
    assert args.use_pytorch_loader, "Only PyTorch loader is supported for now."

    if args.dataset_name is None and args.train_data_dir is None:
        msg = "You must specify either a dataset name from the hub "
        msg += "or a train data directory."
        raise ValueError(msg)

    if (
        isinstance(args.guidance_factor, float)
        and args.guidance_factor <= 1
        and args.model_type == "StableDiffusion"
        and first_check_pass
    ):
        logger.warning(
            "The guidance factor is <= 1: classifier free guidance will not be performed"
        )

    if args.proba_uncond == 1 and first_check_pass:
        logger.warning(
            "The probability of unconditional pass is 1: the model will be trained unconditionally"
        )

        match args.model_type:
            case "DDIM":
                assert (
                    args.guidance_factor is None
                ), "The guidance factor must be None for unconditional training"
            case "StableDiffusion":
                raise NotImplementedError("Need to check this")

    if args.proba_uncond != 1:
        assert isinstance(
            args.proba_uncond, float
        ), "proba_uncond must be a float if conditional training is performed (i.e. proba_uncond != 1)"

    if args.compute_kid and (args.nb_generated_images < args.kid_subset_size):
        if args.debug:
            pass  # when debug flag, kid_subset_size is modified
        else:
            raise ValueError(
                f"'nb_generated_images' (={args.nb_generated_images}) must be >= 'kid_subset_size' (={args.kid_subset_size})"
            )

    if args.gradient_accumulation_steps != 1:
        raise NotImplementedError("Gradient accumulation is not yet supported; TODO!")

    if args.gradient_accumulation_steps > 1 and first_check_pass:
        logger.warning(
            "Gradient accumulation may (probably) fail as the class embedding is not wrapped inside `accelerate.accumulate` context manager; TODO!"
        )

    for c in args.components_to_train:
        if c not in ["denoiser", "class_embedding", "autoencoder"]:
            raise ValueError(
                f"Unknown component '{c}' in 'components_to_train' argument. Should be in ['denoiser', 'class_embedding', 'autoencoder']"
            )

    if args.model_type == "DDIM":
        assert (
            "autoencoder" not in args.components_to_train
        ), "DDIM does not have any autoencoder"
        assert (
            "class_embedding" not in args.components_to_train
        ), "DDIM does not have a custom class embedding"

    if (
        args.pretrained_model_name_or_path is not None
        and args.denoiser_config_path is not None
    ):
        raise ValueError(
            "Cannot set both pretrained_model_name_or_path and denoiser_config_path"
        )

    # TODO: adapt below to support LDM *not* pulled from a pretrained model
    # TODO: adapt below to support DDIM pulled from a pretrained model
    if args.pretrained_model_name_or_path is None:
        assert (
            args.noise_scheduler_config_path is not None
            and args.denoiser_config_path is not None
        ), "If not using a pretrained model, a config must be provided for both the denoiser and noise scheduler"

    if (
        args.pretrained_model_name_or_path is not None
        and not args.learn_denoiser_from_scratch
    ):
        assert (
            args.denoiser_config_path is None
        ), "Cannot provide a denoiser config for now when a pretrained model is used and the denoiser is not learned from scratch"

    if args.perc_samples is not None:
        assert 0 < args.perc_samples <= 100, "perc_samples must be in ]0; 100]"

        if args.seed is None and first_check_pass:
            logger.warning(
                "\033[1;33mSUBSAMPLING THE DATASET BUT NO SEED PROVIDED; RESUMING THIS RUN WILL NOT BE POSSIBLE\033[0m\n"
            )

    assert (
        args.max_num_epochs is not None or args.max_num_steps is not None
    ), "Either max_num_epochs or max_num_steps must be provided"


def create_repo_structure(
    args: Namespace, accelerator, logger: MultiProcessAdapter
) -> tuple[Path, Path, Path, None]:
    repo = None
    this_experiment_folder = Path(
        args.exp_output_dirs_parent_folder, args.experiment_name
    )

    if args.push_to_hub:
        raise NotImplementedError()
        # if args.hub_model_id is None:
        #     repo_name = get_full_repo_name(
        #         Path(this_experiment_folder).name, token=args.hub_token
        #     )
        # else:
        #     repo_name = args.hub_model_id
        #     create_repo(repo_name, exist_ok=True, token=args.hub_token)
        # repo = Repository(this_experiment_folder, clone_from=repo_name, token=args.hub_token)

        # with open(os.path.join(this_experiment_folder, ".gitignore"), "w+") as gitignore:
        #     if "step_*" not in gitignore:
        #         gitignore.write("step_*\n")
        #     if "epoch_*" not in gitignore:
        #         gitignore.write("epoch_*\n")
    elif accelerator.is_main_process:
        os.makedirs(this_experiment_folder, exist_ok=True)

    # Create a folder to save the pipeline during training
    full_pipeline_save_folder = Path(this_experiment_folder, "full_pipeline_save")
    if accelerator.is_main_process:
        os.makedirs(full_pipeline_save_folder, exist_ok=True)

    # Create a folder to save the *initial*, pretrained pipeline
    # HF saves other things when downloading the pipeline (blobs, refs)
    # that we are not interested in(?), hence the two folders.
    # **This folder is shared between all experiments.**
    initial_pipeline_save_folder = Path(
        args.exp_output_dirs_parent_folder, ".initial_pipeline_save"
    )
    if accelerator.is_main_process:
        os.makedirs(initial_pipeline_save_folder, exist_ok=True)

    # Create a temporary folder to save the generated images during training.
    # Used for metrics computations; a small number of these (eval_batch_size) is logged
    image_generation_tmp_save_folder = Path(
        this_experiment_folder, ".tmp_image_generation_folder"
    )

    # verify that the checkpointing folder is empty if not resuming run from a checkpoint
    chckpt_save_path = Path(this_experiment_folder, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(chckpt_save_path, exist_ok=True)
        chckpts = list(chckpt_save_path.iterdir())
        if not args.resume_from_checkpoint and len(chckpts) > 0:
            msg = (
                "\033[1;33mTHE CHECKPOINTING FOLDER IS NOT EMPTY BUT THE CURRENT RUN WILL NOT RESUME FROM A CHECKPOINT. "
                "THIS WILL RESULT IN ERASING THE JUST-SAVED CHECKPOINTS DURING ALL TRAINING "
                "UNTIL IT REACHES THE LAST CHECKPOINTING STEP ALREADY PRESENT IN THE FOLDER.\033[0m\n"
            )
            logger.warning(msg)

    return (
        image_generation_tmp_save_folder,
        initial_pipeline_save_folder,
        full_pipeline_save_folder,
        repo,
    )


def setup_logger(logger: MultiProcessAdapter, accelerator) -> None:
    # set default logging format
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # print one message per process
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def setup_xformers_memory_efficient_attention(
    model: diffusers.ModelMixin, logger: MultiProcessAdapter
) -> None:
    if is_xformers_available():
        import xformers

        xformers_version = version.parse(xformers.__version__)
        if xformers_version == version.parse("0.0.16"):
            logger.warn(
                "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
            )
        model.enable_xformers_memory_efficient_attention()
    else:
        raise ValueError(
            "xformers is not available. Make sure it is installed correctly"
        )


def modify_args_for_debug(
    logger: MultiProcessAdapter, args: Namespace, nb_tot_training_examples: int
) -> None:
    logger.warning("\033[1;33mDEBUG FLAG: MODIFYING PASSED ARGS\033[0m")
    args.eval_save_model_every_epochs = 1
    args.nb_generated_images = args.eval_batch_size
    args.num_train_timesteps = 10
    args.num_inference_steps = 5
    args.checkpoints_total_limit = 1
    if args.max_num_epochs is not None:
        args.max_num_epochs = 3
    if args.max_num_steps is not None:
        args.max_num_steps = 100
    # 3 checkpoints during the debug training
    num_update_steps_per_epoch = ceil(
        nb_tot_training_examples / args.gradient_accumulation_steps
    )
    max_train_steps = (
        args.max_num_epochs * num_update_steps_per_epoch
        if args.max_num_epochs is not None
        else 3 * num_update_steps_per_epoch
    )
    args.checkpointing_steps = max_train_steps // 3
    args.kid_subset_size = min(1000, args.nb_generated_images)


def is_it_best_model(
    main_metric_values: list[float],
    best_metric: float,
    logger: MultiProcessAdapter,
    args: Namespace,
) -> tuple[bool, float]:
    current_value = np.mean(main_metric_values)
    if current_value < best_metric:
        logger.info(f"New best model with metric {args.main_metric}={current_value}")
        best_metric = current_value
        best_model_to_date = True
    else:
        best_model_to_date = False

    return best_model_to_date, best_metric


def get_initial_best_metric() -> float:
    return float("inf")


def get_HF_component_names(components_to_train: list[str]) -> list[str]:
    """The names of the components are badly chosen (the vae is a unet...
    scheduler? you mean learning rate scheduler?...) so we *need* to use our own
    –and then re-use the original ones 🙃
    """
    components_to_train_transcribed = []

    if "denoiser" in components_to_train:
        components_to_train_transcribed.append("unet")
    if "autoencoder" in components_to_train:
        components_to_train_transcribed.append("vae")
    if "class_embedding" in components_to_train:
        components_to_train_transcribed.append("class_embedding")

    assert len(components_to_train_transcribed) == len(components_to_train)

    return components_to_train_transcribed


# From https://stackoverflow.com/a/56877039/12723904
# The 'Table of Content' [TOC] style print function
def _pretty_info_log(
    logger: MultiProcessAdapter,
    key: str,
    val,
    space_char: str = "_",
    val_loc: int = 78,
) -> None:
    # key:        This would be the TOC item equivalent
    # val:        This would be the TOC page number equivalent
    # space_char: This is the spacing character between key and val (often a dot for a TOC), must be >= 5
    # val_loc:    This is the location in the string where the first character of val would be located

    val_loc = max(5, val_loc)

    if val_loc <= len(key):
        # if val_loc is within the space of key, truncate key and
        cut_str = "{:." + str(val_loc - 4) + "}"
        key = cut_str.format(key) + "..." + space_char

    space_str = "{:" + space_char + ">" + str(val_loc - len(key) + len(str(val))) + "}"
    to_print = key + space_str.format("\033[1m" + str(val) + "\033[0m")
    logger.info(to_print)


def print_info_at_run_start(
    logger: MultiProcessAdapter,
    args: Namespace,
    pipeline_components: list[str],
    components_to_train_transcribed: list[str],
    noise_scheduler: DDIMScheduler,
    tot_nb_samples: int,
    total_batch_size: int,
    tot_training_steps: int,
):
    logger.info("\033[1m" + "*" * 46 + " Running training " + "*" * 46 + "\033[0m")
    _pretty_info_log(
        logger,
        "Model",
        args.model_type,
    )
    _pretty_info_log(
        logger,
        "Experiment name",
        args.experiment_name,
    )
    this_experiment_folder = Path(
        args.exp_output_dirs_parent_folder, args.experiment_name
    )
    _pretty_info_log(
        logger, "Experiment output folder", this_experiment_folder.as_posix()
    )
    _pretty_info_log(logger, "Pretrained model", args.pretrained_model_name_or_path)
    _pretty_info_log(logger, "Components to train", str(args.components_to_train))
    _pretty_info_log(
        logger,
        "Components kept frozen",
        str(set(pipeline_components) - set(components_to_train_transcribed)),
    )
    _pretty_info_log(
        logger,
        "Num diffusion discretization steps",
        noise_scheduler.config.num_train_timesteps,
    )
    _pretty_info_log(
        logger,
        "Num diffusion generation steps",
        args.num_inference_steps,
    )
    _pretty_info_log(
        logger,
        "Guidance Factor",
        args.guidance_factor,
    )
    _pretty_info_log(
        logger,
        "Probability of unconditional pass",
        args.proba_uncond,
    )
    _pretty_info_log(
        logger,
        "Prediction type",
        noise_scheduler.config.prediction_type,
    )
    _pretty_info_log(
        logger,
        "Learning rate",
        args.learning_rate,
    )
    _pretty_info_log(
        logger,
        "Num examples",
        tot_nb_samples,
    )
    _pretty_info_log(
        logger,
        "Maximum number of epochs",
        args.max_num_epochs,
    )
    _pretty_info_log(
        logger,
        "Maximum number of training steps",
        args.max_num_steps,
    )
    _pretty_info_log(
        logger,
        "Instantaneous batch size per device",
        args.train_batch_size,
    )
    _pretty_info_log(
        logger,
        "Total train batch size (w. parallel, distributed & accumulation)",
        total_batch_size,
    )
    _pretty_info_log(
        logger,
        "Gradient Accumulation steps",
        args.gradient_accumulation_steps,
    )
    _pretty_info_log(
        logger,
        "Use EMA",
        args.use_ema,
    )
    _pretty_info_log(
        logger,
        "Total optimization steps",
        tot_training_steps,
    )
    _pretty_info_log(
        logger,
        "Num steps between checkpoints",
        args.checkpointing_steps,
    )
    tot_nb_chckpts = tot_training_steps // args.checkpointing_steps
    _pretty_info_log(
        logger,
        "Num checkpoints during training",
        tot_nb_chckpts,
    )
    _pretty_info_log(
        logger,
        "Num epochs between model evaluation",
        args.eval_save_model_every_epochs,
    )
    _pretty_info_log(
        logger,
        "Num generated images",
        args.nb_generated_images,
    )
