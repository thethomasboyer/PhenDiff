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

# TODO's:
# +++ add the other methods
# +++ adapt to Jean Zay
# ++ add conditional metrics computation (requires != folders w.r.t. target labels + find_deep=True for uncond case)
# ++ parallelize inference
# ++ plug wandb
# + check if fp16 is used + other inference time optimizations
# + save inverted Gaussians once per noise scheduler config?

import logging
from pathlib import Path

import hydra
import torch_fidelity
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

from src.utils_Img2Img import (
    classifier_free_guidance,
    ddib,
    linear_interp_custom_guidance_inverted_start,
    load_datasets,
)

BATCH_SIZES: dict[str, dict[str, dict[str, int]]] = {
    "rtx8000": {
        "DDIM": {
            "linear_interp_custom_guidance_inverted_start": 48,
            "classifier_free_guidance": -1,
            "ddib": -1,
        },
        "SD": {
            "linear_interp_custom_guidance_inverted_start": 64,
            "classifier_free_guidance": -1,
            "ddib": -1,
        },
    },
    "a100": {
        "DDIM": {
            "linear_interp_custom_guidance_inverted_start": -1,
            "classifier_free_guidance": -1,
            "ddib": -1,
        },
        "SD": {
            "linear_interp_custom_guidance_inverted_start": -1,
            "classifier_free_guidance": -1,
            "ddib": -1,
        },
    },
}

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="img2img_comparison_conf",
    config_name="general_config",
)
def main(cfg: DictConfig) -> None:
    # ------------------------------------------- Misc. -------------------------------------------
    logger.info(f"Passed config:\n{OmegaConf.to_yaml(cfg)}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir: str = hydra_cfg["runtime"]["output_dir"]

    # --------------------------------- Load pretrained pipelines ---------------------------------
    logger.info(f"Loading pipelines")
    pipes = call(cfg.pipeline)

    # ---------------------------------------- Load dataset ---------------------------------------
    # assume only one dataset
    dataset_name = next(iter(cfg.dataset))

    # load dataset TODO: directly instantiate from hydra?
    logger.info(f"Loading dataset {dataset_name}")
    train_dataset, test_dataset = load_datasets(cfg, dataset_name)

    # --------------------------------- Class transfer experiments --------------------------------
    for class_transfer_method in cfg.class_transfer_method:
        match class_transfer_method:
            case "linear_interp_custom_guidance_inverted_start":
                logger.info("Running linear_interp_custom_guidance_inverted_start")
                linear_interp_custom_guidance_inverted_start(
                    pipes,
                    train_dataset,
                    test_dataset,
                    cfg,
                    output_dir,
                    logger,
                    BATCH_SIZES,
                )
            case "classifier_free_guidance":
                logger.info("Running classifier_free_guidance")
                classifier_free_guidance(
                    pipes,
                    train_dataset,
                    test_dataset,
                    cfg,
                    output_dir,
                    logger,
                    BATCH_SIZES,
                )
            case "ddib":
                logger.info("Running ddib")
                ddib(
                    pipes,
                    train_dataset,
                    test_dataset,
                    cfg,
                    output_dir,
                    logger,
                    BATCH_SIZES,
                )
            case _:
                raise ValueError(
                    f"Unknown class transfer method: {class_transfer_method}"
                )

    # ------------------------------------ Metrics computation ------------------------------------
    for class_transfer_method in cfg.class_transfer_method:
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
                    batch_size=BATCH_SIZES[cfg.gpu][pipename][class_transfer_method],
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

                # 2. Conditional TODO


if __name__ == "__main__":
    main()
