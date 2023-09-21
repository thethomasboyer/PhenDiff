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

################################## img2img_comparison_launcher.py #################################
# This script acts as a wrapper-launcher of the img2img_comparison.py script.
#
# It performs the following tasks:
# - copy the experiment config to the experiment folder
# (to ensure that is is not modified if the actual job launch is delayed)
# - modify this copied config so that Hydra uses one single output folder
# (instead of one output folder per process + job launch time)
# - configure accelerate
# - configure SLURM (if enabled)
# - set some environment variables
# - submit the task


import os
import sys
from pathlib import Path

import hydra
import submitit
from omegaconf import DictConfig, ListConfig

from src.utils_Img2Img import duplicate_config_to_experiment_folder

# hardcoded config paths
DEFAULT_CONFIG_PATH = "my_img2img_comparison_conf"
DEFAULT_CONFIG_NAME = "general_config"


class Task:
    """
    Represents an `accelerate launch img2img_comparison.py` command call to the system.

    When `__call__`'ed, `Task` will:
    - pass the given config to `img2img_comparison.py`
    - configure `accelerate` with the given config
    - set some environment variables
    - submit the command with `os.system`

    Can be called directly or submitted to SLURM with `submitit`.
    """

    def __init__(
        self,
        cfg: DictConfig,
        overrides: ListConfig,
        task_config_path: Path,
        task_config_name: Path,
    ):
        self.cfg: DictConfig = cfg
        self.overrides: ListConfig = overrides
        self.task_config_path: Path = task_config_path
        self.task_config_name: Path = task_config_name

    def __call__(self):
        # Accelerate config
        accelerate_cfg = ""
        for cfg_item_name, cfg_item_value in self.cfg.accelerate.launch_args.items():
            if cfg_item_value is True or cfg_item_value in ["True", "true"]:
                accelerate_cfg += f"--{cfg_item_name} "
            elif cfg_item_value is False or cfg_item_value in ["False", "false"]:
                pass
            else:
                accelerate_cfg += f"--{cfg_item_name} {cfg_item_value} "

        if self.cfg.debug:
            accelerate_cfg += "--debug"

        if self.cfg.accelerate.offline:
            offline_vars = "WANDB_MODE=offline HF_DATASETS_OFFLINE=1 "
        else:
            offline_vars = ""

        # Launched command
        final_cmd = f"{offline_vars}accelerate launch {accelerate_cfg} {self.cfg.path_to_script_parent_folder}/img2img_comparison.py --config-path {self.task_config_path} --config-name {self.task_config_name}"

        for override in self.overrides:
            final_cmd += f" {override}"

        print("Executing command: ", final_cmd)

        # Execute command
        exit_code = os.system(final_cmd)
        if exit_code != 0:
            raise RuntimeError(f"Command {final_cmd} failed with exit code {exit_code}")


@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name=DEFAULT_CONFIG_NAME,
)
def main(cfg: DictConfig) -> None:
    if cfg.slurm.enabled:
        # SLURM
        executor = submitit.AutoExecutor(folder=cfg.slurm.output_folder)

        if cfg.debug:
            runtime = "02:00:00"
            qos = "qos_gpu-dev"
        else:
            runtime = "20:00:00"
            qos = "qos_gpu-t3"

        additional_parameters = {
            "hint": "nomultithread",
            "mail_user": cfg.slurm.email,
            "mail_type": "FAIL",
        }
        if cfg.debug:
            pass  # TODO: find how to use pty with submitit
        else:
            additional_parameters["output"] = f"{cfg.slurm.output_folder}/jobid-%j.out"
            additional_parameters["error"] = f"{cfg.slurm.output_folder}/jobid-%j.err"

        executor.update_parameters(
            slurm_job_name=f"{cfg.project}-{cfg.run_name}",
            slurm_constraint="a100",
            slurm_nodes=1,
            slurm_ntasks_per_node=1,
            slurm_gres=f"gpu:{cfg.slurm.num_gpus}",
            slurm_cpus_per_task=int(64 * cfg.slurm.num_gpus / 8),
            slurm_additional_parameters=additional_parameters,
            slurm_time=runtime,
            slurm_qos=qos,
            slurm_account="kio@a100",
        )

    # CL overrides
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    overrides: ListConfig = hydra_cfg.overrides.task

    # Create experiment folder & copy config
    # (to prevent config modif when delaying launches)
    # hydra.run.dir will be set to the timestamped subfolder created by hydra
    # when *this* *launcher* job was launched
    task_config_path, launcher_config_name = duplicate_config_to_experiment_folder(
        cfg, hydra_cfg
    )

    # Task
    task = Task(cfg, overrides, task_config_path, launcher_config_name)

    # Submit
    if cfg.slurm.enabled:
        job = executor.submit(task)  # type: ignore
    else:
        task()

    # Monitor
    if cfg.slurm.enabled and cfg.slurm.monitor:
        submitit.helpers.monitor_jobs([job])  # type: ignore

    # Get minimal stacktrace
    if cfg.slurm.enabled:
        output = job.result()  # type: ignore
        print(output)


if __name__ == "__main__":
    sys.exit(main())
