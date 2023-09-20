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


from pathlib import Path

import hydra
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from hydra.utils import call
from omegaconf import DictConfig, OmegaConf

from src.utils_Img2Img import (
    ClassTransferExperimentParams,
    compute_metrics,
    get_config_path_and_name,
    load_datasets,
    modify_debug_args,
    perform_class_transfer_experiment,
)
from src.utils_misc import setup_logger

logger = get_logger(__name__, log_level="INFO")


@hydra.main(
    version_base=None,
    config_path="my_img2img_comparison_conf",
    config_name="general_config",
)
def main(cfg: DictConfig) -> None:
    # ---------------------------------------- Accelerator ----------------------------------------
    accelerator = Accelerator(
        mixed_precision=cfg.accelerate.launch_args.mixed_precision,
        log_with="wandb",
    )

    # ------------------------------------------- WandB -------------------------------------------
    setup_logger(logger, accelerator)
    logger.info(
        f"Logging to entity/project/run: {cfg.entity}/{cfg.project}/{cfg.run_name}"
    )
    accelerator.init_trackers(
        project_name=cfg.project,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
        # save metadata to the "wandb" directory
        # inside the *parent* folder common to all *experiments*
        init_kwargs={
            "wandb": {
                "entity": cfg.entity,
                "dir": cfg.exp_parent_folder,
                "name": cfg.run_name,
                "save_code": True,
            }
        },
    )

    # ------------------------------------------- Misc. -------------------------------------------
    # get Hydra config & output dir
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    output_dir: str = hydra_cfg["runtime"]["output_dir"]
    # show config
    config_path, config_name = get_config_path_and_name(cfg, hydra_cfg)
    logger.info(f"Config path: {config_path}")
    logger.info(f"Config name: {config_name}")
    logger.info(f"Passed config:\n{OmegaConf.to_yaml(cfg)}")
    # set cache folders
    fidelity_cache_root: Path = Path(cfg.exp_parent_folder, ".fidelity_cache")
    torch_hub_cache_dir = Path(cfg.exp_parent_folder, ".torch_hub_cache")
    torch.hub.set_dir(torch_hub_cache_dir)

    # ------------------------------------------- Debug -------------------------------------------
    num_inference_steps, cfg = modify_debug_args(cfg, logger)

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
        "dataset_name": dataset_name,
        "fidelity_cache_root": fidelity_cache_root,
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
