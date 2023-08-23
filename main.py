# Copyright 2023 The HuggingFace Team and Thomas Boyer. All rights reserved.
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

from argparse import Namespace
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from src.args_parser import parse_args
from src.utils_dataset import setup_dataset
from src.utils_misc import (
    args_checker,
    create_repo_structure,
    get_HF_component_names,
    get_initial_best_metric,
    modify_args_for_debug,
    setup_logger,
)
from src.utils_models import load_initial_pipeline
from src.utils_training import (
    generate_samples_and_compute_metrics,
    get_training_setup,
    perform_training_epoch,
    resume_from_checkpoint,
    save_pipeline,
)

logger: MultiProcessAdapter = get_logger(__name__, log_level="INFO")


def main(args: Namespace):
    # ---------------------------------- Accelerator ---------------------------------
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        automatic_checkpoint_naming=False,
        project_dir=Path(
            args.exp_output_dirs_parent_folder, args.experiment_name
        ).as_posix(),
    )

    # unused parameters when unconditionally denoising samples in CLF guidance training for DDIM;
    # not needed for SD as the HF code passes zeros instead of skipping the conditioning part of the network
    # TODO: interesting to think about the implications of these two different methods!
    kwargs_handlers = (
        [DistributedDataParallelKwargs(find_unused_parameters=True)]
        if args.model_type == "DDIM"
        else None
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=kwargs_handlers,
    )

    # ------------------------------------- WandB ------------------------------------
    logger.info(f"Logging to project {args.experiment_name}")
    accelerator.init_trackers(
        project_name=args.experiment_name,
        config=vars(args),
        # save metadata to the "wandb" directory
        # inside the *parent* folder common to all experiments
        init_kwargs={
            "wandb": {
                "dir": args.exp_output_dirs_parent_folder,
                "name": args.run_name,
                "save_code": True,
            }
        },
    )

    # Make one log on every process with the configuration for debugging.
    setup_logger(logger, accelerator)

    # ----------------------------- Repository Structure -----------------------------
    (
        image_generation_tmp_save_folder,
        initial_pipeline_save_folder,
        full_pipeline_save_folder,
        repo,
    ) = create_repo_structure(args, accelerator, logger)
    accelerator.wait_for_everyone()

    fidelity_cache_root: Path = Path(
        args.exp_output_dirs_parent_folder, ".fidelity_cache"
    )

    torch_hub_cache_dir: Path = Path(
        args.exp_output_dirs_parent_folder, ".torch_hub_cache"
    )
    torch.hub.set_dir(torch_hub_cache_dir)

    # ------------------------------------ Checks ------------------------------------
    if accelerator.is_main_process:
        args_checker(args, logger)

    # ------------------------------------ Dataset -----------------------------------
    dataset, raw_dataset, nb_classes = setup_dataset(args, logger)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # ------------------------------------ Debug -------------------------------------
    if args.debug:
        modify_args_for_debug(logger, args, len(train_dataloader))
        if accelerator.is_main_process:
            args_checker(args, logger, False)  # check again after debug modifications 😈

    # --------------------------------- Load Pipeline --------------------------------
    # Download the full (possibly pretrained) pipeline
    # Note that HF pipelines are not meant to used for training;
    # here they are only used as convenient "supermodel" wrappers
    pipeline = load_initial_pipeline(
        args, initial_pipeline_save_folder, logger, nb_classes, accelerator
    )

    # --------------------------- Move & Freeze Components ---------------------------
    # Move components to device
    pipeline.to(accelerator.device)

    # ❄️ >>> Freeze components <<< ❄️
    if "autoencoder" not in args.components_to_train and hasattr(pipeline, "vae"):
        pipeline.vae.requires_grad_(False)
    if "denoiser" not in args.components_to_train:
        pipeline.unet.requires_grad_(False)
    if "class_embedding" not in args.components_to_train and hasattr(
        pipeline, "class_embedding"
    ):
        pipeline.class_embedding.requires_grad_(False)

    # --------------------------------- Miscellaneous --------------------------------
    # Create EMA for the models
    ema_models = {}
    components_to_train_transcribed = get_HF_component_names(args.components_to_train)
    if args.use_ema:
        for module_name, module in pipeline.components.items():
            if module_name in components_to_train_transcribed:
                ema_models[module_name] = EMAModel(
                    module.parameters(),
                    decay=args.ema_max_decay,
                    use_ema_warmup=True,
                    inv_gamma=args.ema_inv_gamma,
                    power=args.ema_power,
                    model_cls=module.__class__,
                    model_config=module.config,
                )
                ema_models[module_name].to(accelerator.device)
        logger.info(
            f"Created EMA weights for the following models: {list(ema_models)} (corresponding to the (unordered) following args: {args.components_to_train})"
        )

    # Use memory efficient attention
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    # Track gradients of the denoiser
    if accelerator.is_main_process:
        wandb.watch(pipeline.unet)

    # ----------------------------- Save Custom Pipeline -----------------------------
    if accelerator.is_main_process:
        save_pipeline(
            accelerator=accelerator,
            args=args,
            pipeline=pipeline,
            full_pipeline_save_folder=full_pipeline_save_folder,
            repo=repo,
            epoch=0,
            logger=logger,
            ema_models=ema_models,
            components_to_train_transcribed=components_to_train_transcribed,
            first_save=True,
        )
    accelerator.wait_for_everyone()

    # ----------------------------------- Optimizer ----------------------------------
    params_to_optimize = []
    for module_name, module in pipeline.components.items():
        if module_name in components_to_train_transcribed:  # was EMA'ed
            params_to_optimize += list(pipeline.components[module_name].parameters())

    optimizer = torch.optim.AdamW(  # TODO: different params for different components
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ---------------------------- Learning Rate Scheduler ----------------------------
    lr_scheduler = get_scheduler(  # TODO: different params for different components
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # ----------------------------- Distributed Compute  -----------------------------
    # get the total len of the dataloader before distributing it
    total_dataloader_len = len(train_dataloader)

    # prepare distributed training with 🤗's magic
    # first all general training helpers
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )

    # then model-specific submodels (is it beyond hackyness acceptability?)
    match args.model_type:
        case "DDIM":
            denoiser = accelerator.prepare(pipeline.unet)
            pipeline.unet = denoiser
        case "StableDiffusion":
            denoiser, autoencoder, class_embedding = accelerator.prepare(
                pipeline.unet, pipeline.vae, pipeline.class_embedding
            )
            pipeline.unet = denoiser
            pipeline.vae = autoencoder
            pipeline.class_embedding = class_embedding
        case _:
            raise ValueError(f"Unknown model type {args.model_type}")

    # synchronize the conditional & unconditional passes of CLF guidance training between the GPUs
    # to circumvent a nasty and unexplained bug...
    do_uncond_pass_across_all_procs: torch.BoolTensor = torch.zeros(
        (total_dataloader_len,), device=accelerator.device, dtype=torch.bool
    )
    if accelerator.is_main_process:
        # fill tensor on main proc
        for batch_idx in range(total_dataloader_len):
            do_uncond_pass_across_all_procs[batch_idx] = (
                torch.rand(1) < args.proba_uncond
            )  # always true if proba_uncond == 1 as torch.rand -> [0;1[
        # broadcast tensor to all procs
        main_proc_rank = torch.distributed.get_rank()
        assert main_proc_rank == 0, f"Main proc rank is not 0 but {main_proc_rank}"
    torch.distributed.broadcast(do_uncond_pass_across_all_procs, 0)

    # -------------------------------- Training Setup --------------------------------
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     run = os.path.split(__file__)[-1].split(".")[0]
    #     accelerator.init_trackers(run)

    first_epoch = 0
    global_step = 0
    resume_step = 0

    (
        num_update_steps_per_epoch,
        actual_eval_batch_sizes_for_this_process,
    ) = get_training_setup(
        args,
        accelerator,
        train_dataloader,
        logger,
        list(pipeline.components),
        components_to_train_transcribed,
        len(dataset),
        pipeline.scheduler,
    )

    # ---------------------------- Resume from Checkpoint ----------------------------
    if args.resume_from_checkpoint:
        first_epoch, resume_step, global_step = resume_from_checkpoint(
            args, logger, accelerator, num_update_steps_per_epoch, global_step
        )

    # ----------------------------- Initial best metrics -----------------------------
    if accelerator.is_main_process:
        best_metric = get_initial_best_metric()

    # --------------------------------- Training loop --------------------------------
    for epoch in range(first_epoch, args.num_epochs):
        # Training epoch
        global_step = perform_training_epoch(
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            accelerator=accelerator,
            pipeline=pipeline,
            ema_models=ema_models,
            components_to_train_transcribed=components_to_train_transcribed,
            epoch=epoch,
            train_dataloader=train_dataloader,
            args=args,
            first_epoch=first_epoch,
            resume_step=resume_step,
            global_step=global_step,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
            do_uncond_pass_across_all_procs=do_uncond_pass_across_all_procs,
            params_to_clip=params_to_optimize,
        )

        # Generate sample images for visual inspection & metrics computation
        if epoch % args.eval_save_model_every_epochs == 0 or (
            args.precise_first_n_epochs is not None
            and epoch < args.precise_first_n_epochs
        ):
            best_model_to_date, best_metric = generate_samples_and_compute_metrics(
                args=args,
                accelerator=accelerator,
                pipeline=pipeline,
                image_generation_tmp_save_folder=image_generation_tmp_save_folder,
                fidelity_cache_root=fidelity_cache_root,
                actual_eval_batch_sizes_for_this_process=actual_eval_batch_sizes_for_this_process,
                epoch=epoch,
                global_step=global_step,
                ema_models=ema_models,
                components_to_train_transcribed=components_to_train_transcribed,
                nb_classes=nb_classes,
                logger=logger,
                dataset=dataset,
                raw_dataset=raw_dataset,
                best_metric=best_metric if accelerator.is_main_process else None,
            )
            # save model if best to date
            if accelerator.is_main_process:
                # log best model indicator
                accelerator.log(
                    {
                        "best_model_to_date": int(best_model_to_date),
                    },
                    step=global_step,
                )
                if epoch != 0 and best_model_to_date:
                    save_pipeline(
                        accelerator=accelerator,
                        args=args,
                        pipeline=pipeline,
                        full_pipeline_save_folder=full_pipeline_save_folder,
                        repo=repo,
                        epoch=epoch,
                        logger=logger,
                        ema_models=ema_models,
                        components_to_train_transcribed=components_to_train_transcribed,
                    )

        # do not start new epoch before generation & pipeline saving is done
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args: Namespace = parse_args()
    main(args)
