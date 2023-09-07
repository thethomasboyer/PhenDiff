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
from omegaconf import DictConfig


class Task:
    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg

    def __call__(self):
        # Accelerate
        accelerate_cfg = "--machine_rank=0"
        accelerate_cfg += "--mixed_precision=fp16"
        accelerate_cfg += "--num_machines=1"
        accelerate_cfg += f"--num_processes={self.cfg.num_GPUs}"
        accelerate_cfg += "--rdzv_backend=static"
        accelerate_cfg += "--same_network"
        accelerate_cfg += "--dynamo_backend=no"
        if self.cfg.num_GPUs != 1:
            accelerate_cfg += "--multi_gpu"

        final_cmd = (
            "WANDB_MODE=offline accelerate launch "
            + accelerate_cfg
            + " img2img_comparison.py"
        )

        print("Executing command: ", final_cmd)
        os.system(final_cmd)


@hydra.main(
    version_base=None,
    config_path="img2img_comparison_conf",
    config_name="general_config",
)
def main(cfg: DictConfig) -> None:
    # SLURM
    executor = submitit.AutoExecutor(folder=f"{cfg.project}/{cfg.run_name}")

    if cfg.debug:
        runtime = "02:00:00"
        qos = "qos_gpu-dev"
    else:
        runtime = "20:00:00"
        qos = "qos_gpu-t3"

    run_output_folder = "{cfg.exp_dirs_parent_folder}/{cfg.project}/{cfg.run_name}"

    executor.update_parameters(
        job_name=f"{cfg.project}-{cfg.run_name}",
        constraint="a100",
        nodes=1,
        ntasks_per_node=1,
        gres=f"gpu:{cfg.num_GPUs}",
        cpus_per_task=64 * cfg.num_GPUs / 8,
        hint="nomultithread",
        time=runtime,
        qos=qos,
        account="kio@a100",
        mail_user="tboyer@bio.ens.psl.eu",
        mail_type="FAIL",
        error=f"{run_output_folder}/jobid-%j.err",
        output=f"{run_output_folder}/jobid-%j.out",
    )

    # Task
    task = Task(cfg)

    # Submit
    job = executor.submit(task)
    submitit.helpers.monitor_jobs([job])


if __name__ == "__main__":
    sys.exit(main())
