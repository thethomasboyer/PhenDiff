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
import sys

import hydra
import submitit
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


class Task:
    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg

    def __call__(self):
        # Accelerate config
        accelerate_cfg = "--machine_rank=0"
        accelerate_cfg += " --mixed_precision=fp16"
        accelerate_cfg += " --num_machines=1"
        accelerate_cfg += f" --num_processes={self.cfg.num_GPUs}"
        accelerate_cfg += " --rdzv_backend=static"
        accelerate_cfg += " --same_network"
        accelerate_cfg += " --dynamo_backend=no"
        if self.cfg.num_GPUs != 1:
            accelerate_cfg += " --multi_gpu"

        # Launched command # TODO:
        final_cmd = (
            "WANDB_MODE=offline HF_DATASETS_OFFLINE=1 accelerate launch "
            + accelerate_cfg
            + " img2img_comparison.py"
        )

        # pass (non-hydra-specific) CL overrides to img2img_comparison
        overrides = HydraConfig.get().overrides.task
        print("################### DEBUG overrides", overrides)

        print("Executing command: ", final_cmd)

        # Execute command
        exit_code = os.system(final_cmd)
        if exit_code != 0:
            raise RuntimeError(f"Command {final_cmd} failed with exit code {exit_code}")


@hydra.main(
    version_base=None,
    config_path="img2img_comparison_conf",
    config_name="general_config",
)
def main(cfg: DictConfig) -> None:
    # SLURM
    executor = submitit.AutoExecutor(folder=cfg.output_folder)

    if cfg.debug:
        runtime = "02:00:00"
        qos = "qos_gpu-dev"
    else:
        runtime = "20:00:00"
        qos = "qos_gpu-t3"

    executor.update_parameters(
        slurm_job_name=f"{cfg.project}-{cfg.run_name}",
        slurm_constraint="a100",
        slurm_nodes=1,
        slurm_ntasks_per_node=1,
        slurm_gres=f"gpu:{cfg.num_GPUs}",
        slurm_cpus_per_task=int(64 * cfg.num_GPUs / 8),
        slurm_additional_parameters={
            "hint": "nomultithread",
            "mail_user": "tboyer@bio.ens.psl.eu",
            "mail_type": "FAIL",
        },
        slurm_time=runtime,
        slurm_qos=qos,
        slurm_account="kio@a100",
    )
    if cfg.debug:
        pass  # TODO: find how to use pty with submitit
        # executor.update_parameters(slurm_srun_args=["--pty"])
    else:
        executor.update_parameters(
            slurm_error=f"{cfg.output_folder}/jobid-%j.err",
            slurm_output=f"{cfg.output_folder}/jobid-%j.out",
        )

    # Task
    task = Task(cfg)

    # Submit
    job = executor.submit(task)

    # Monitor
    if cfg.monitor:
        submitit.helpers.monitor_jobs([job])

    # Get minimal stacktrace
    output = job.result()
    print(output)


if __name__ == "__main__":
    sys.exit(main())
