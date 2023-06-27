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

import os
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter, get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

import wandb
from src.args_parser import parse_args
from src.custom_embedding import CustomEmbedding
from src.utils_dataset import setup_dataset
from src.utils_misc import (
    args_checker,
    create_repo_structure,
    setup_logger,
    setup_xformers_memory_efficient_attention,
)
from src.utils_training import (
    generate_samples_and_compute_metrics,
    get_training_setup,
    perform_training_epoch,
    resume_from_checkpoint,
    save_pipeline,
)

logger: MultiProcessAdapter = get_logger(__name__, log_level="INFO")


def main(args):
    # ------------------------- Checks -------------------------
    args_checker(args, logger)

    # ----------------------- Accelerator ----------------------
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        automatic_checkpoint_naming=False,
        project_dir=args.output_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
    )

    # -------------------------- WandB -------------------------
    wandb_project_name = args.output_dir.lstrip("experiments/")
    logger.info(f"Logging to project {wandb_project_name}")
    accelerator.init_trackers(
        project_name=wandb_project_name,
        config=args,
    )

    # Make one log on every process with the configuration for debugging.
    setup_logger(logger, accelerator)

    # ------------------ Repository Structure ------------------
    (
        image_generation_tmp_save_folder,
        initial_pipeline_save_folder,
        full_pipeline_save_folder,
        repo,
    ) = create_repo_structure(args, accelerator)

    # ------------------------- Dataset ------------------------
    dataset, nb_classes = setup_dataset(args, logger)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # ------------------- Pretrained Pipeline ------------------
    # Download the full pretrained pipeline.
    # Note that the actual folder to pull components from is
    # initial_pipeline_save_folder/snapshots/<gibberish>/ (probably a hash?)
    # hence the need to get the *true* save folder (initial_pipeline_save_path)
    initial_pipeline_save_path = StableDiffusionImg2ImgPipeline.download(
        args.pretrained_model_name_or_path,
        cache_dir=initial_pipeline_save_folder,
    )

    # Load the pretrained components
    autoencoder_model: AutoencoderKL = AutoencoderKL.from_pretrained(
        initial_pipeline_save_path,
        subfolder="vae",
        local_files_only=True,
    )
    if args.learn_denoiser_from_scratch:
        denoiser_model_config = UNet2DConditionModel.load_config(
            Path(initial_pipeline_save_path, "unet", "config.json"),
        )
        denoiser_model: UNet2DConditionModel = UNet2DConditionModel.from_config(
            denoiser_model_config,
        )
    else:
        denoiser_model: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
            initial_pipeline_save_path,
            subfolder="unet",
            local_files_only=True,
        )

    # ----------------- Custom Class Embeddings ----------------
    # create a custom class inheriting from diffusers.ModelMixin
    # in order to use Hugging Face's routines
    match args.class_embedding_type:
        case "one_hot":
            raise NotImplementedError(
                "Dimensions will mismatch with one-hot encoding; TODO: fix"
            )
            # class_embedding = torch.nn.functional.one_hot(torch.arange(nb_classes))
            # ..?
            # class_embedding.to(accelerator.device)
        case "embedding":
            class_embedding = CustomEmbedding(nb_classes, args.class_embedding_dim)
        case _:
            raise ValueError(
                f"Unrecognized class embedding type: {args.class_embedding_type}"
            )

    # ---------------- Move & Freeze Components ----------------
    # Move components to device
    autoencoder_model.to(accelerator.device)
    denoiser_model.to(accelerator.device)
    class_embedding.to(accelerator.device)

    # ❄️ >>> Freeze components <<< ❄️
    autoencoder_model.requires_grad_(False)

    # --------------------- Noise Scheduler --------------------
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=args.num_training_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )

    # ---------------------- Miscellaneous ---------------------

    # Create EMA for the unet model
    if args.use_ema:
        ema_unet = EMAModel(
            denoiser_model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DConditionModel,
            model_config=denoiser_model.config,
        )
    else:
        ema_unet = None

    if args.enable_xformers_memory_efficient_attention:
        setup_xformers_memory_efficient_attention(denoiser_model, logger)
        setup_xformers_memory_efficient_attention(autoencoder_model, logger)

    # track gradients
    if accelerator.is_main_process:
        wandb.watch(denoiser_model)

    # ------------------ Save Custom Pipeline ------------------
    if accelerator.is_main_process:
        save_pipeline(
            accelerator=accelerator,
            denoiser_model=denoiser_model,
            class_embedding=class_embedding,
            args=args,
            ema_model=ema_unet,
            noise_scheduler=noise_scheduler,
            full_pipeline_save_folder=full_pipeline_save_folder,
            repo=repo,
            epoch=0,
            logger=logger,
            first_save=True,
            autoencoder_model=autoencoder_model,
        )
    accelerator.wait_for_everyone()

    # ------------------------ Optimizer -----------------------
    optimizer = torch.optim.AdamW(
        list(denoiser_model.parameters()) + list(class_embedding.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ----------------- Learning Rate Scheduler -----------------
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # ------------------ Distributed Compute  ------------------
    # get the total len of the dataloader before distributing it
    total_dataloader_len = len(train_dataloader)

    # prepare distributed training with 🤗's magic
    (
        denoiser_model,
        optimizer,
        train_dataloader,
        lr_scheduler,
        class_embedding,
        autoencoder_model,
    ) = accelerator.prepare(
        denoiser_model,
        optimizer,
        train_dataloader,
        lr_scheduler,
        class_embedding,
        autoencoder_model,
    )

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
            )
        # broadcast tensor to all procs
        main_proc_rank = torch.distributed.get_rank()
        assert main_proc_rank == 0, f"Main proc rank is not 0 but {main_proc_rank}"
    torch.distributed.broadcast(do_uncond_pass_across_all_procs, 0)

    # --------------------- Training Setup ---------------------
    if args.use_ema:
        ema_unet.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    first_epoch = 0
    global_step = 0
    resume_step = 0

    (
        num_update_steps_per_epoch,
        actual_eval_batch_sizes_for_this_process,
    ) = get_training_setup(args, accelerator, train_dataloader, logger, dataset)

    # ----------------- Resume from Checkpoint -----------------
    if args.resume_from_checkpoint:
        first_epoch, resume_step, global_step = resume_from_checkpoint(
            args, logger, accelerator, num_update_steps_per_epoch, global_step
        )

    # ---------------------- Training loop ---------------------
    for epoch in range(first_epoch, args.num_epochs):
        # Training epoch
        global_step = perform_training_epoch(
            denoiser_model=denoiser_model,
            autoencoder_model=autoencoder_model,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            accelerator=accelerator,
            epoch=epoch,
            train_dataloader=train_dataloader,
            args=args,
            first_epoch=first_epoch,
            resume_step=resume_step,
            noise_scheduler=noise_scheduler,
            global_step=global_step,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema_model=ema_unet,
            logger=logger,
            class_embedding=class_embedding,
            do_uncond_pass_across_all_procs=do_uncond_pass_across_all_procs,
        )

        # Generate sample images for visual inspection & metrics computation
        if epoch % args.generate_images_epochs == 0:
            generate_samples_and_compute_metrics(
                args=args,
                accelerator=accelerator,
                denoiser_model=denoiser_model,
                class_embedding=class_embedding,
                ema_model=ema_unet,
                noise_scheduler=noise_scheduler,
                image_generation_tmp_save_folder=image_generation_tmp_save_folder,
                actual_eval_batch_sizes_for_this_process=actual_eval_batch_sizes_for_this_process,
                epoch=epoch,
                global_step=global_step,
                full_pipeline_save_path=full_pipeline_save_folder,
                nb_classes=nb_classes,
                logger=logger,
                dataset=dataset,
            )

        if (
            accelerator.is_main_process
            and epoch % args.save_model_epochs == 0
            and epoch != 0
        ):
            save_pipeline(
                accelerator=accelerator,
                denoiser_model=denoiser_model,
                class_embedding=class_embedding,
                args=args,
                ema_model=ema_unet,
                noise_scheduler=noise_scheduler,
                full_pipeline_save_folder=full_pipeline_save_folder,
                repo=repo,
                epoch=epoch,
                logger=logger,
            )

        # do not start new epoch before generation & pipeline saving is done
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
