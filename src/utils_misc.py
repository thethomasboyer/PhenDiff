# Utilities.

import logging
import os
from pathlib import Path
from typing import Optional

import datasets
import diffusers
import torch
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version

from .cond_unet_2d import CondUNet2DModel


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


def split(l, n, idx):
    """
    https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length

    Should probably be replaced by Accelerator.split_between_processes.
    """
    k, m = divmod(len(l), n)
    l = [l[i * k + min(i, m): (i + 1) * k + min(i + 1, m)] for i in range(n)]
    return l[idx]


def args_checker(args):
    assert args.use_pytorch_loader, "Only PyTorch loader is supported for now."

    if args.guidance_factor == 0:
        msg = "The guidance factor is null "
        msg += "but the probability to generate unconditionally is not 🧐"
        assert args.proba_uncond == 0, msg

    if args.proba_uncond == 0:
        msg = "The probability to generate unconditionally is null "
        msg += "but the guidance factor is not 🧐"
        assert args.guidance_factor == 0, msg

    if args.dataset_name is None and args.train_data_dir is None:
        msg = "You must specify either a dataset name from the hub "
        msg += "or a train data directory."
        raise ValueError(msg)

    if args.proba_uncond == 1:
        msg = "The probability to generate unconditionally is 1 "
        msg += "but the guidance factor is not None 🧐"
        assert args.guidance_factor is None, msg

    if args.prediction_type == "velocity":
        raise NotImplementedError(
            "Velocity prediction is not implemented yet; TODO!")


# create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
def save_model_hook(models, weights, output_dir, args, ema_model):
    if args.use_ema:
        ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

    for model in models:
        model.save_pretrained(os.path.join(output_dir, "unet"))

        # make sure to pop weight so that corresponding model is not saved again
        weights.pop()


def load_model_hook(models, input_dir, args, ema_model, accelerator):
    if args.use_ema:
        load_model = EMAModel.from_pretrained(
            os.path.join(input_dir, "unet_ema"), CondUNet2DModel
        )
        ema_model.load_state_dict(load_model.state_dict())
        ema_model.to(accelerator.device)
        del load_model

    for _ in range(len(models)):
        # pop models so that they are not loaded again
        model = models.pop()

        # load diffusers style into model
        load_model = CondUNet2DModel.from_pretrained(
            input_dir, subfolder="unet")
        model.register_to_config(**load_model.config)

        model.load_state_dict(load_model.state_dict())
        del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)


def create_repo_structure(args, accelerator):
    repo = None
    if args.push_to_hub:
        raise NotImplementedError()
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(args.output_dir).name, token=args.hub_token
            )
        else:
            repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
        repo = Repository(
            args.output_dir, clone_from=repo_name, token=args.hub_token)

        with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
            if "step_*" not in gitignore:
                gitignore.write("step_*\n")
            if "epoch_*" not in gitignore:
                gitignore.write("epoch_*\n")
    elif args.output_dir is not None and accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create a folder to save the *full* pipeline
    full_pipeline_save_folder = Path(args.output_dir, "full_pipeline_save")
    if accelerator.is_main_process:
        os.makedirs(full_pipeline_save_folder, exist_ok=True)

    # Create a temporary folder to save the generated images during training.
    # Used for metrics computations; a small number of these (eval_batch_size) is logged
    image_generation_tmp_save_folder = Path(
        args.output_dir, ".tmp_image_generation_folder"
    )

    return image_generation_tmp_save_folder, full_pipeline_save_folder, repo


def setup_logger(logger, accelerator):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def setup_xformers_memory_efficient_attention(model, logger):
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
