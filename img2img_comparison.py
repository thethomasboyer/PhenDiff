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

###################################### img2img_comparison.py ######################################
# This script launches a series of experiments to compare class-to-class image transfer methods.
#
# Its config is located in the img2img_comparison_conf folder and managed with hydra
# (https://hydra.cc/).
#
# The experiments are logged with wandb (https://wandb.ai) and run sequentially,
# with metrics computed at the end of each experiment with torch-fidelity.


# TODO's:
# ++ sweep
# ++ add all (start type)/(transfer method) variants
# + check if fp16 is used + other inference time optimizations
# + save inverted Gaussians once per noise scheduler config?

import hydra
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

from src.utils_Img2Img import (
    ClassTransferExperimentParams,
    compute_metrics,
    load_datasets,
    modify_debug_args,
    perform_class_transfer_experiment,
)
from src.utils_misc import setup_logger

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
    "M40-12GB": {
        "DDIM": {
            "linear_interp_custom_guidance_inverted_start": 12,
            "classifier_free_guidance_forward_start": 32,
            "ddib": 32,
        },
        "SD": {
            "linear_interp_custom_guidance_inverted_start": 32,
            "classifier_free_guidance_forward_start": 64,
            "ddib": 64,
        },
    },
}


logger = get_logger(__name__, log_level="INFO")


@hydra.main(
    version_base=None,
    config_path="img2img_comparison_conf",
    config_name="general_config",
)
def main(cfg: DictConfig) -> None:
    # ---------------------------------------- Accelerator ----------------------------------------
    accelerator = Accelerator(
        mixed_precision=cfg.mixed_precision,
        log_with="wandb",
    )

    # ------------------------------------------- WandB -------------------------------------------
    setup_logger(logger, accelerator)
    logger.info(f"Logging to project/run: {cfg.project}/{cfg.run_name}")
    accelerator.init_trackers(
        project_name=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
        # save metadata to the "wandb" directory
        # inside the *parent* folder common to all *experiments*
        init_kwargs={
            "wandb": {
                "dir": "outputs",  # hardcoded for now TODO
                "name": cfg.run_name,
                "save_code": True,
            }
        },
    )

    # ------------------------------------------- Misc. -------------------------------------------
    logger.info(f"Passed config:\n{OmegaConf.to_yaml(cfg)}")
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    output_dir: str = hydra_cfg["runtime"]["output_dir"]

    # --------------------------------- Debug ---------------------------------
    num_inference_steps = modify_debug_args(cfg, logger)

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
    # Params common to all experiments
    transfer_exp_common_params = {
        "pipes": pipes,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "cfg": cfg,
        "output_dir": output_dir,
        "accelerator": accelerator,
        "logger": logger,
        "batch_sizes": BATCH_SIZES,
        "dataset_name": dataset_name,
    }

    # Sweep over experiments
    for class_transfer_method in cfg.class_transfer_method:
        # args
        exp_args = ClassTransferExperimentParams(
            class_transfer_method=class_transfer_method,
            num_inference_steps=num_inference_steps,
            **transfer_exp_common_params,
        )

        ############# Class transfer ############
        logger.info(
            f"\033[1m==========================> Running {class_transfer_method}\033[0m"
        )
        perform_class_transfer_experiment(exp_args)
        accelerator.wait_for_everyone()

        ########## Metrics computation ##########
        logger.info(f"\033[1m==========================> Computing metrics\033[0m")
        if accelerator.is_main_process:
            compute_metrics(exp_args)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
