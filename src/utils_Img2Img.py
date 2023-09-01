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
from logging import Logger
from math import ceil
from pathlib import Path
from typing import Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_fidelity
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

import wandb
from src.custom_pipeline_stable_diffusion_img2img import (
    CustomStableDiffusionImg2ImgPipeline,
)
from src.pipeline_conditional_ddim import ConditionalDDIMPipeline

DEBUG_BATCHES_LIMIT = 0


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
        assert (
            tensor.min() >= -1 and tensor.max() <= 1
        ), f"Expecting values in [-1, 1], got tensor.min()={tensor.min()} and tensor.max()={tensor.max()}"
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


@torch.no_grad()
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


def perform_class_transfer_experiment(
    class_transfer_method: str,
    pipes: dict,
    train_dataset: Dataset,
    test_dataset: Dataset,
    cfg: DictConfig,
    output_dir: str,
    logger: Logger,
    batch_sizes: dict[str, dict[str, dict[str, int]]],
    num_inference_steps: Optional[int],
):
    match class_transfer_method:
        case "linear_interp_custom_guidance_inverted_start":
            _linear_interp_custom_guidance_inverted_start(
                pipes,
                train_dataset,
                test_dataset,
                cfg,
                output_dir,
                logger,
                batch_sizes,
                num_inference_steps,
            )
        case "classifier_free_guidance_forward_start":
            _classifier_free_guidance_forward_start(
                pipes,
                train_dataset,
                test_dataset,
                cfg,
                output_dir,
                logger,
                batch_sizes,
                num_inference_steps,
            )
        case "ddib":
            _ddib(
                pipes,
                train_dataset,
                test_dataset,
                cfg,
                output_dir,
                logger,
                batch_sizes,
                num_inference_steps,
            )
        case _:
            raise ValueError(f"Unknown class transfer method: {class_transfer_method}")


def compute_metrics(
    pipes: dict,
    cfg: DictConfig,
    dataset_name: str,
    logger: Logger,
    output_dir: str,
    class_transfer_method: str,
    batch_sizes: dict[str, dict[str, dict[str, int]]],
):
    for pipename in pipes:
        for split, dataset in zip(
            ["train", "test"], [cfg.dataset[dataset_name].root] * 2
        ):
            # 1. Unconditional
            logger.info("Computing metrics (unconditional case)")
            # get the images to compare
            true_images = Path(dataset, split).as_posix()
            generated_images = (
                output_dir + f"/{class_transfer_method}/{pipename}/{split}"
            )
            # compute metrics
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=true_images,
                input2=generated_images,
                cuda=True,
                batch_size=batch_sizes[cfg.gpu][pipename][class_transfer_method] * 4,
                isc=cfg.compute_isc,
                fid=cfg.compute_fid,
                kid=cfg.compute_kid,
                verbose=False,
                # cache_root=fidelity_cache_root,
                # input1_cache_name=f"{class_name}",  # forces caching
                kid_subset_size=cfg.kid_subset_size,
                samples_find_deep=True,
            )
            # log metrics
            logger.info(
                f"Metrics for {class_transfer_method} with {pipename} on {split} split: {metrics_dict}"
            )
            wandb.log(
                {
                    f"{class_transfer_method}/{pipename}/{split}": metrics_dict,
                }
            )

            # 2. Conditional TODO
            logger.info("Computing metrics (conditional case)")


def _ddib(
    pipes: dict,
    train_dataset: Dataset,
    test_dataset: Dataset,
    cfg: DictConfig,
    output_dir: str,
    logger: Logger,
    batch_sizes: dict[str, dict[str, dict[str, int]]],
    num_inference_steps: Optional[int],
):
    """TODO: docstring"""
    for pipename, pipe in pipes.items():
        # get the number of inference steps
        if num_inference_steps is None:
            num_inference_steps = cfg.pipeline[pipename].num_inference_steps

        for split, dataset in zip(["train", "test"], [train_dataset, test_dataset]):
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_sizes[cfg.gpu][pipename]["ddib"],
                shuffle=True,
            )

            # Move pipe to GPU
            distributed_state = PartialState()
            pipe = pipe.to(distributed_state.device)

            # Create save dir
            save_dir = output_dir + f"/ddib/{pipename}/{split}"
            os.makedirs(save_dir)

            # Iterate over batches
            for step, batch in enumerate(
                tqdm(dataloader, desc=f"Running {pipename} on {split} dataset")
            ):
                # Get batch
                clean_images = batch["images"].to(distributed_state.device)
                orig_class_labels = batch["class_labels"].to(distributed_state.device)
                # only works for the binary case...
                target_class_labels = 1 - orig_class_labels
                filenames = batch["file_basenames"]

                # Preprocess inputs if LDM
                # target_class_labels must be saved for filename
                if isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
                    clean_images, [orig_class_cond] = _LDM_preprocess(
                        pipe, clean_images, [orig_class_labels]
                    )
                else:
                    orig_class_cond = orig_class_labels
                # Perform inversion
                inverted_gauss = _inversion(
                    pipe,
                    pipename,
                    clean_images,
                    orig_class_cond,
                    cfg,
                    num_inference_steps,
                )

                # Perform generation
                if isinstance(pipe, ConditionalDDIMPipeline):
                    images_to_save = pipe(
                        class_labels=target_class_labels,
                        w=None,
                        num_inference_steps=num_inference_steps,
                        start_image=inverted_gauss,
                        frac_diffusion_skipped=0,
                    ).images
                elif isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
                    images_to_save = pipe(
                        image=clean_images,
                        class_labels=target_class_labels,
                        strength=cfg.class_transfer_method.classifier_free_guidance_forward_start.frac_diffusion_skipped,
                        add_forward_noise_to_image=False,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=0,  # guidance_scale <= 1.0 disables guidance
                    )
                else:
                    raise NotImplementedError

                # Save images to disk
                if isinstance(images_to_save, Image.Image):
                    images_to_save = [images_to_save]
                for i, image_to_save in enumerate(images_to_save):
                    save_filename = (
                        save_dir + f"/{filenames[i]}_to_{target_class_labels[i]}.png"
                    )
                    image_to_save.save(save_filename)

                # Stop if debug flag
                if cfg.debug and step >= DEBUG_BATCHES_LIMIT:
                    logger.warn(
                        f"debug flag: stopping after {DEBUG_BATCHES_LIMIT+1} batches"
                    )
                    break


def _classifier_free_guidance_forward_start(
    pipes: dict,
    train_dataset: Dataset,
    test_dataset: Dataset,
    cfg: DictConfig,
    output_dir: str,
    logger: Logger,
    batch_sizes: dict[str, dict[str, dict[str, int]]],
    num_inference_steps: Optional[int],
):
    """TODO: docstring"""
    for pipename, pipe in pipes.items():
        # get the number of inference steps
        if num_inference_steps is None:
            num_inference_steps = cfg.pipeline[pipename].num_inference_steps

        for split, dataset in zip(["train", "test"], [train_dataset, test_dataset]):
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_sizes[cfg.gpu][pipename][
                    "classifier_free_guidance_forward_start"
                ],
                shuffle=True,
            )

            # Move pipe to GPU
            distributed_state = PartialState()
            pipe = pipe.to(distributed_state.device)

            # Create save dir
            save_dir = (
                output_dir
                + f"/classifier_free_guidance_forward_start/{pipename}/{split}"
            )
            os.makedirs(save_dir)

            # Iterate over batches
            for step, batch in enumerate(
                tqdm(dataloader, desc=f"Running {pipename} on {split} dataset")
            ):
                # Get batch
                clean_images = batch["images"].to(distributed_state.device)
                orig_class_labels = batch["class_labels"].to(distributed_state.device)
                # only works for the binary case...
                target_class_labels = 1 - orig_class_labels
                filenames = batch["file_basenames"]

                # Perform guided generation
                if isinstance(pipe, ConditionalDDIMPipeline):
                    images_to_save = pipe(
                        class_labels=target_class_labels,
                        w=cfg.class_transfer_method.classifier_free_guidance_forward_start.guidance_scale,
                        num_inference_steps=num_inference_steps,
                        start_image=clean_images,
                        frac_diffusion_skipped=cfg.class_transfer_method.classifier_free_guidance_forward_start.frac_diffusion_skipped,
                    ).images
                elif isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
                    images_to_save = pipe(
                        image=clean_images,
                        class_labels=target_class_labels,
                        strength=cfg.class_transfer_method.classifier_free_guidance_forward_start.frac_diffusion_skipped,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=cfg.class_transfer_method.classifier_free_guidance_forward_start.guidance_scale,
                    )
                else:
                    raise NotImplementedError

                # Save images to disk
                if isinstance(images_to_save, Image.Image):
                    images_to_save = [images_to_save]
                for i, image_to_save in enumerate(images_to_save):
                    save_filename = (
                        save_dir + f"/{filenames[i]}_to_{target_class_labels[i]}.png"
                    )
                    image_to_save.save(save_filename)

                # Stop if debug flag
                if cfg.debug and step >= DEBUG_BATCHES_LIMIT:
                    logger.warn(
                        f"debug flag: stopping after {DEBUG_BATCHES_LIMIT+1} batches"
                    )
                    break


def _linear_interp_custom_guidance_inverted_start(
    pipes: dict,
    train_dataset: Dataset,
    test_dataset: Dataset,
    cfg: DictConfig,
    output_dir: str,
    logger: Logger,
    batch_sizes: dict[str, dict[str, dict[str, int]]],
    num_inference_steps: Optional[int],
):
    """TODO: docstring"""
    for pipename, pipe in pipes.items():
        # get the number of inference steps
        if num_inference_steps is None:
            num_inference_steps = cfg.pipeline[pipename].num_inference_steps

        for split, dataset in zip(["train", "test"], [train_dataset, test_dataset]):
            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_sizes[cfg.gpu][pipename][
                    "linear_interp_custom_guidance_inverted_start"
                ],
                shuffle=True,
            )

            # Move pipe to GPU
            distributed_state = PartialState()
            pipe = pipe.to(distributed_state.device)

            # Create save dir
            save_dir = (
                output_dir
                + f"/linear_interp_custom_guidance_inverted_start/{pipename}/{split}"
            )
            os.makedirs(save_dir)

            # Iterate over batches
            for step, batch in enumerate(
                tqdm(dataloader, desc=f"Running {pipename} on {split} dataset")
            ):
                # Get batch
                clean_images = batch["images"].to(distributed_state.device)
                orig_class_labels = batch["class_labels"].to(distributed_state.device)
                # only works for the binary case...
                target_class_labels = 1 - orig_class_labels
                filenames = batch["file_basenames"]

                # Preprocess inputs if LDM
                # target_class_labels must be saved for filename
                target_class_embeds = None
                if isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
                    clean_images, (
                        orig_class_labels,
                        target_class_embeds,
                    ) = _LDM_preprocess(
                        pipe, clean_images, [orig_class_labels, target_class_labels]
                    )

                # Perform inversion
                inverted_gauss = _inversion(
                    pipe,
                    pipename,
                    clean_images,
                    orig_class_labels,
                    cfg,
                    num_inference_steps,
                )

                # Perform guided generation
                target_cond = (
                    target_class_embeds
                    if target_class_embeds is not None
                    else target_class_labels
                )
                image = _custom_guided_generation(
                    pipe,
                    pipename,
                    inverted_gauss,
                    target_cond,
                    cfg,
                    num_inference_steps,
                )

                # Decode from latent space if LDM
                if isinstance(pipe, CustomStableDiffusionImg2ImgPipeline):
                    image = _decode_to_images(pipe, image)
                    # also normalize image back to [-1, 1]
                    image -= image.min()
                    image /= image.max()
                    image = image * 2 - 1

                # Save images to disk
                images_to_save = tensor_to_PIL(image)
                if isinstance(images_to_save, Image.Image):
                    images_to_save = [images_to_save]
                for i, image_to_save in enumerate(images_to_save):
                    save_filename = (
                        save_dir + f"/{filenames[i]}_to_{target_class_labels[i]}.png"
                    )
                    image_to_save.save(save_filename)

                # Stop if debug flag
                if cfg.debug and step >= DEBUG_BATCHES_LIMIT:
                    logger.warn(
                        f"debug flag: stopping after {DEBUG_BATCHES_LIMIT+1} batches"
                    )
                    break


def _custom_guided_generation(
    pipe: ConditionalDDIMPipeline | CustomStableDiffusionImg2ImgPipeline,
    pipename: str,
    input_images: Tensor,
    target_class_labels: Tensor,
    cfg: DictConfig,
    num_inference_steps: int,
) -> Tensor:
    """TODO: docstring"""
    # duplicate the input images
    images = input_images.clone().detach()

    # setup the scheduler
    pipe.scheduler.set_timesteps(num_inference_steps)

    # perform guided generation
    for t in tqdm(
        pipe.scheduler.timesteps, desc="Computing guided generation", leave=False
    ):
        # 0. reset grads on images
        images = images.detach().requires_grad_()

        # 1. predict noise/velocity/etc.
        model_output = pipe.unet(
            images,
            t,
            target_class_labels,
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
    num_inference_steps: int,
) -> Tensor:
    """TODO: docstring"""
    # duplicate the input images
    gauss = input_images.clone().detach()

    # setup the inverted DDIM scheduler
    DDIM_inv_scheduler: DDIMInverseScheduler = DDIMInverseScheduler.from_config(
        pipe.scheduler.config,
    )
    DDIM_inv_scheduler.set_timesteps(num_inference_steps)

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


@torch.no_grad()
def _LDM_preprocess(
    pipe: CustomStableDiffusionImg2ImgPipeline,
    images: Tensor,
    class_labels_seq: Optional[list[Tensor]] = None,
) -> tuple[Tensor, Optional[list[Tensor]]]:
    # encode images to latent space...
    latents = _encode_to_latents(pipe, images)
    # ...and class labels to class embedding
    if class_labels_seq is None:
        class_labels_embeds_seq = None
    else:
        class_labels_embeds_seq = []
        for class_labels in class_labels_seq:
            class_labels_embeds = pipe._encode_class(
                class_labels=class_labels,
                device=class_labels.device,
                do_classifier_free_guidance=False,
            )
            # hack to match the expected encoder_hidden_states shape
            class_labels_embeds = hack_class_embedding(class_labels_embeds)
            class_labels_embeds_seq.append(class_labels_embeds)
    return latents, class_labels_embeds_seq


@torch.no_grad()
def _encode_to_latents(
    pipe: CustomStableDiffusionImg2ImgPipeline, gauss: Tensor
) -> Tensor:
    # encode
    latent = pipe.vae.encode(gauss).latent_dist.sample()
    # scale
    latent *= pipe.vae.config.scaling_factor
    return latent


@torch.no_grad()
def _decode_to_images(
    pipe: CustomStableDiffusionImg2ImgPipeline, latents: Tensor
) -> Tensor:
    # unscale
    latents /= pipe.vae.config.scaling_factor
    # decode
    images = pipe.vae.decode(latents, return_dict=False)[0]
    return images
