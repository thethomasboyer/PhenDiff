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

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.stats import normaltest
from torch import Tensor


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
