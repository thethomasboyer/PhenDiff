# Copyright 2023 Thomas Boyer. All rights reserved.
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

import os
from math import ceil
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import PartialState
from datasets import load_dataset
from diffusers.schedulers import DDIMInverseScheduler
from omegaconf import DictConfig
from PIL import Image
from scipy.stats import normaltest
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from src.custom_pipeline_stable_diffusion_img2img import (
    CustomStableDiffusionImg2ImgPipeline,
)
from src.pipeline_conditional_ddim import ConditionalDDIMPipeline


@torch.no_grad()
def check_Gaussianity(gauss: Tensor) -> None:
    print(
        f"Checking Gausianity of components of tensor of shape {tuple(gauss.shape)}..."
    )
    _, axes = plt.subplots(nrows=1, ncols=len(gauss), figsize=(16, 3))
    for idx, itm in enumerate(gauss):
        axes[idx].hist(itm.cpu().numpy().flatten(), bins=100, range=(-3, 3))
        print(
            f"Gaussian(?) {idx}: mean={itm.mean().item()}, std={itm.std().item()}",
            end="; ",
        )
        _, p = normaltest(itm.cpu().numpy().flatten())
        print(f"2-sided Χ² probability for the normality hypothesis: {p}")
    plt.show()


@torch.no_grad()
def tensor_to_PIL(
    tensor: Tensor,
    channel: int | str = "mean",
) -> Image.Image | list[Image.Image]:
    assert tensor.ndim == 4, "Expecting a tensor of shape (N, C, H, W)"
    assert channel in ["mean"] + list(
        range(tensor.shape[1])
    ), f"Expecting a channel in {list(range(tensor.shape[1]))} or 'mean', got {channel}"

    img_to_show = tensor.clone().detach()
    # these are latent vectors => return a grayscale image
    if tensor.shape[1] == 4:
        # min-max normalization
        img_to_show -= img_to_show.min()
        img_to_show /= img_to_show.max()
        img_to_show = img_to_show.clamp(0, 1)
        if isinstance(channel, int):
            # take the given channel only for visualization
            img_to_show = img_to_show[:, channel].view(
                tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]
            )
        elif channel == "mean":
            # convert to grayscale taking the mean over the 4 channels
            img_to_show = img_to_show.mean(dim=1, keepdim=True)
    # these are "true" images => return a color image
    elif tensor.shape[1] == 3:
        assert tensor.min() >= -1 and tensor.max() <= 1, "Expecting values in [-1, 1]"
        if tensor.min() != -1:
            print(
                f"Warning in tensor_to_PIL: tensor.min() = {tensor.min().item()} != -1"
            )
        if tensor.max() != 1:
            print(
                f"Warning in tensor_to_PIL: tensor.max() = {tensor.max().item()} != -1"
            )
        img_to_show = (img_to_show / 2 + 0.5).clamp(0, 1)

    # convert to PIL image
    img_to_show = img_to_show.cpu().permute(0, 2, 3, 1).numpy()
    img_to_show = (img_to_show * 255).round().astype("uint8")
    if img_to_show.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [
            Image.fromarray(image.squeeze(), mode="L") for image in img_to_show
        ]
    else:
        pil_images = [Image.fromarray(image) for image in img_to_show]
    # if only one image return it directly
    if len(pil_images) == 1:
        return pil_images[0]

    return pil_images


def print_grid(
    list_PIL_images: list[Image.Image],
    nb_img_per_row: int = 5,
    titles: list[str] | None = None,
    figsize=(16, 4),
) -> None:
    if titles is not None:
        assert len(titles) == len(list_PIL_images)
    num_images = len(list_PIL_images)
    nrows = ceil(num_images / nb_img_per_row)
    _, axes = plt.subplots(
        nrows=nrows, ncols=nb_img_per_row, figsize=(figsize[0], figsize[1] * nrows)
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]
    for i in range(num_images):
        row_nb = i // nb_img_per_row
        col_nb = i % nb_img_per_row
        axes[row_nb, col_nb].imshow(list_PIL_images[i].convert("RGB"))
        axes[row_nb, col_nb].axis("off")
        if titles is not None:
            axes[row_nb, col_nb].set_title(titles[i], fontsize=10)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def hack_class_embedding(cl_embed: Tensor) -> Tensor:
    """Hack to match the expected encoder_hidden_states shape"""
    assert cl_embed.ndim == 2, "Expecting a tensor of shape (N, E)"
    (bs, ed) = cl_embed.shape
    cl_embed = cl_embed.reshape(bs, 1, ed)
    padding = torch.zeros_like(cl_embed).repeat(1, 76, 1).to(cl_embed.device)
    cl_embed = torch.cat([cl_embed, padding], dim=1)
    return cl_embed


def Lp_loss(
    x: torch.Tensor, y: torch.Tensor, p: int | float | Literal["inf", "-inf"] = 2
) -> torch.Tensor:
    """Returns the L_p norms of the flattened `(x[i] - y[i])` or `(x[i] - y)` vectors for each `i` in the batch.

    Arguments
    ---------
    - x: `torch.Tensor`, shape `(N, C, H, W)`
    - y: `torch.Tensor`, shape `(C, H, W)` or `(N, C, H, W)`
    - p: `int | float | "inf" | "-inf"`; default to `2`

    Returns
    -------
    `torch.linalg.vector_norm(x - y, dim=(1, 2, 3), ord=p)`, that is:
    ```
        torch.linalg.vector_norm(x[i] - y, ord=p) for i in range(N)
    ```
    """
    assert (
        x.shape[1:] == y.shape or x.shape == y.shape
    ), f"y.shape={y.shape} should be equal to either x.shape or x.shape[1:] (x.shape={x.shape})"
    assert len(y.shape) in (
        3,
        4,
    ), f"y.shape={y.shape} should be (C, H, W) or (N, C, H, W)"
    return torch.linalg.vector_norm(x - y, dim=(1, 2, 3), ord=p)


def linear_interp_custom_guidance_inverted_start(
    pipes: dict,
    train_dataset: Dataset,
    test_dataset: Dataset,
    cfg: DictConfig,
    output_dir: str,
):
    """TODO: docstring"""

    for pipename, pipe in pipes.items():
        for split, dataset in zip(["train", "test"], [train_dataset, test_dataset]):
            # create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.pipeline[pipename].eval_batch_size,
                shuffle=True,
            )

            # move pipe to GPU
            distributed_state = PartialState()
            pipe = pipe.to(distributed_state.device)

            # create save dir
            save_dir = (
                output_dir
                + f"/linear_interp_custom_guidance_inverted_start/{pipename}/{split}"
            )
            os.makedirs(save_dir)

            for batch in tqdm(
                dataloader, desc=f"Running {pipename} on {split} dataset"
            ):
                clean_images = batch["images"].to(distributed_state.device)
                orig_class_labels = batch["class_labels"].to(distributed_state.device)
                # only works for the binary case...
                target_class_labels = 1 - orig_class_labels
                filenames = batch["file_basenames"]

                # perform inversion
                inverted_gauss = _inversion(
                    pipe, pipename, clean_images, orig_class_labels, cfg
                )

                # perform guided generation
                image = _custom_guided_generation(
                    pipe, pipename, inverted_gauss, target_class_labels, cfg
                )

                images_to_save = tensor_to_PIL(image)
                if isinstance(images_to_save, Image.Image):
                    images_to_save = [images_to_save]

                for i, image_to_save in enumerate(images_to_save):
                    save_filename = (
                        save_dir + f"/{filenames[i]}_to_{target_class_labels[i]}.png"
                    )
                    image_to_save.save(save_filename)


def load_datasets(cfg: DictConfig, dataset_name: str) -> Tuple[Dataset, Dataset]:
    """TODO: docstring"""
    # data preprocessing
    normalize = cfg.dataset[dataset_name].normalize
    preproc = transforms.Compose(
        [
            transforms.Resize(
                cfg.dataset[dataset_name].resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([normalize[0]], [normalize[1]]),
        ]
    )

    def transform_images(examples):
        images = [preproc(image.convert("RGB")) for image in examples["image"]]
        file_basenames = [
            os.path.splitext(os.path.basename(image.filename))[0]
            for image in examples["image"]
        ]
        class_labels = torch.tensor(examples["label"]).long()
        return {
            "images": images,
            "file_basenames": file_basenames,
            "class_labels": class_labels,
        }

    # instantiate datasets
    train_dataset: Dataset = load_dataset(
        "imagefolder",
        data_dir=cfg.dataset[dataset_name].root,
        drop_labels=False,
        split="train",
    )
    test_dataset: Dataset = load_dataset(
        "imagefolder",
        data_dir=cfg.dataset[dataset_name].root,
        drop_labels=False,
        split="test",
    )
    train_dataset.set_transform(transform_images)
    test_dataset.set_transform(transform_images)

    return train_dataset, test_dataset


def _custom_guided_generation(
    pipe: ConditionalDDIMPipeline | CustomStableDiffusionImg2ImgPipeline,
    pipename: str,
    input_images: Tensor,
    target_class_labels: Tensor,
    cfg: DictConfig,
) -> Tensor:
    """TODO: docstring"""
    # duplicate the input images
    images = input_images.clone().detach()

    # setup the scheduler
    pipe.scheduler.set_timesteps(cfg.pipeline[pipename].num_inference_steps)

    # perform guided generation
    for t in tqdm(
        pipe.scheduler.timesteps, desc="Computing guided generation", leave=False
    ):
        # 0. reset grads on images
        images = images.detach().requires_grad_()

        # 1. predict noise/velocity/etc.
        model_output = pipe.unet(
            sample=images,
            timestep=t,
            class_labels=target_class_labels,
        ).sample

        # 2. get x_0 prediction
        x0 = pipe.scheduler.step(
            model_output,
            t,
            images,
        ).pred_original_sample

        # 3. compute loss
        # each image in batch has its own loss with respect to the original sample
        # hence losses.shape = (batch_size,)
        losses = Lp_loss(
            x0,
            input_images,
            cfg.class_transfer_method.linear_interp_custom_guidance_inverted_start.p,
        )

        # 4. get gradient
        losses_seq = [losses[i] for i in range(len(input_images))]
        guidance_grad = torch.autograd.grad(losses_seq, images)[0]

        # 5. modify the image based on this gradient
        guidance_loss_scale = (
            cfg.class_transfer_method.linear_interp_custom_guidance_inverted_start.guidance_loss_scale
        )
        images = images.detach() - guidance_loss_scale * guidance_grad

        # 6. x_t -> x_t-1
        images = pipe.scheduler.step(
            model_output,
            t,
            images,
        ).prev_sample

    return images


@torch.no_grad()
def _inversion(
    pipe: ConditionalDDIMPipeline | CustomStableDiffusionImg2ImgPipeline,
    pipename: str,
    input_images: Tensor,
    class_labels: Tensor,
    cfg: DictConfig,
) -> Tensor:
    """TODO: docstring"""
    # duplicate the input images
    gauss = input_images.clone().detach()

    # setup the inverted DDIM scheduler
    DDIM_inv_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(
        pipe.scheduler.config,
    )
    DDIM_inv_scheduler.set_timesteps(cfg.pipeline[pipename].num_inference_steps)

    # invert the diffeq
    for t in tqdm(
        DDIM_inv_scheduler.timesteps, desc="Computing inverted Gaussians", leave=False
    ):
        model_output = pipe.unet(gauss, t, class_labels).sample

        gauss = DDIM_inv_scheduler.step(
            model_output,
            t,
            gauss,
        ).prev_sample

    return gauss
