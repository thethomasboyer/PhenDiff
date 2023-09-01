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
# +++ adapt to Jean Zay
# +++ parallelize inference
# ++ add conditional metrics computation (requires != folders w.r.t. target labels)
# ++ sweep
# ++ log some images pairs
# ++ add all (start type)/(transfer method) variants
# + check if fp16 is used + other inference time optimizations
# + save inverted Gaussians once per noise scheduler config?

import logging

import hydra
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

import wandb
from src.utils_Img2Img import (
    compute_metrics,
    load_datasets,
    perform_class_transfer_experiment,
)

BATCH_SIZES: dict[str, dict[str, dict[str, int]]] = {
    "rtx8000": {
        "DDIM": {
            "linear_interp_custom_guidance_inverted_start": 48,
            "classifier_free_guidance_forward_start": 128,
            "ddib": 128,
        },
        "SD": {
            "linear_interp_custom_guidance_inverted_start": 128,
            "classifier_free_guidance_forward_start": 256,
            "ddib": 256,
        },
    },
    "a100": {
        "DDIM": {
            "linear_interp_custom_guidance_inverted_start": -1,
            "classifier_free_guidance_forward_start": -1,
            "ddib": -1,
        },
        "SD": {
            "linear_interp_custom_guidance_inverted_start": -1,
            "classifier_free_guidance_forward_start": -1,
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

    # ------------------------------------------- Wandb -------------------------------------------
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(name=cfg.wandb.run_name, project=cfg.wandb.project, save_code=True)

    # --------------------------------- Load pretrained pipelines ---------------------------------
    logger.info(f"\033[1m==========================> Loading pipelines\033[0m")
    pipes = call(cfg.pipeline)

    # ---------------------------------------- Load dataset ---------------------------------------
    # assume only one dataset
    dataset_name = next(iter(cfg.dataset))

    # load dataset TODO: directly instantiate from hydra?
    logger.info(
        f"\033[1m==========================> Loading dataset {dataset_name}\033[0m"
    )
    train_dataset, test_dataset = load_datasets(cfg, dataset_name)

    # ---------------------------------------- Experiments ----------------------------------------
    for class_transfer_method in cfg.class_transfer_method:
        # Quick gen if debug flag
        if cfg.debug:
            num_inference_steps = 10
            logger.warning(
                f"Debug mode: setting num_inference_steps to {num_inference_steps}"
            )
        else:  # else let each pipeline have its own param
            num_inference_steps = None
        # ------------------------------------- Class transfer ------------------------------------
        logger.info(
            f"\033[1m==========================> Running {class_transfer_method}\033[0m"
        )
        perform_class_transfer_experiment(
            class_transfer_method,
            pipes,
            train_dataset,
            test_dataset,
            cfg,
            output_dir,
            logger,
            BATCH_SIZES,
            num_inference_steps,
        )
        # ---------------------------------- Metrics computation ----------------------------------
        logger.info(f"\033[1m==========================> Computing metrics\033[0m")
        compute_metrics(
            pipes,
            cfg,
            dataset_name,
            logger,
            output_dir,
            class_transfer_method,
            BATCH_SIZES,
        )


if __name__ == "__main__":
    main()
