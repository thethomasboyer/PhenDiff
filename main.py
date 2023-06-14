import inspect
import os

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from src.args_parser import parse_args
from src.cond_unet_2d import CondUNet2DModel
from src.utils_dataset import setup_dataset
from src.utils_misc import (
    args_checker,
    create_repo_structure,
    setup_logger,
    setup_xformers_memory_efficient_attention,
)
from src.utils_training import (
    checkpoint_model,
    generate_samples_and_compute_metrics,
    get_training_setup,
    perform_training_epoch,
    resume_from_checkpoint,
)

logger = get_logger(__name__, log_level="INFO")


def main(args):
    # ------------------------- Checks -------------------------
    args_checker(args)

    # ----------------------- Accelerator ----------------------
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        automatic_checkpoint_naming=False,
        project_dir=args.output_dir,
    )

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
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

    # ------------------- Repository scruture ------------------
    (
        image_generation_tmp_save_folder,
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
    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )
    unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    # Move components to device and cast frozen ones to float16
    vae.to(accelerator.device, dtype=torch.float16)
    unet.to(accelerator.device)
    text_encoder.to(accelerator.device, dtype=torch.float16)
    
    # Freeze components
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # --------------------- Noise scheduler --------------------
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # Create EMA for the unet model
    if args.use_ema:
        ema_unet = EMAModel(
            unet.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DConditionModel,
            model_config=unet.config,
        )
    else:
        ema_unet = None

    if args.enable_xformers_memory_efficient_attention:
        setup_xformers_memory_efficient_attention(unet, logger)

    # track gradients
    if accelerator.is_main_process:
        wandb.watch(unet)

    # ------------------------ Optimizer -----------------------
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ----------------- Learning rate scheduler -----------------
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # ------------------ Distributed compute  ------------------
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # --------------------- Training setup ---------------------
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
        tot_nb_eval_batches,
        actual_eval_batch_sizes_for_this_process,
    ) = get_training_setup(args, accelerator, train_dataloader, logger, dataset)

    # ----------------- Resume from checkpoint -----------------
    if args.resume_from_checkpoint:
        first_epoch, resume_step, global_step = resume_from_checkpoint(
            args, logger, accelerator, num_update_steps_per_epoch, global_step
        )

    # ---------------------- Seeds & RNGs ----------------------
    rng = np.random.default_rng()  # TODO: seed this

    # ---------------------- Training loop ---------------------
    for epoch in range(first_epoch, args.num_epochs):
        # Training epoch
        global_step = perform_training_epoch(
            unet,
            num_update_steps_per_epoch,
            accelerator,
            epoch,
            train_dataloader,
            args,
            first_epoch,
            resume_step,
            noise_scheduler,
            rng,
            global_step,
            optimizer,
            lr_scheduler,
            ema_unet,
            logger,
        )

        # Generate sample images for visual inspection & metrics computation
        if (
            epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1
        ) and epoch > 0:
            generate_samples_and_compute_metrics(
                args,
                accelerator,
                unet,
                ema_unet,
                noise_scheduler,
                image_generation_tmp_save_folder,
                dataset,
                actual_eval_batch_sizes_for_this_process,
                args.guidance_factor,
                epoch,
                global_step,
                nb_classes,
            )

        if (
            accelerator.is_main_process
            and (epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1)
            and epoch != 0
        ):
            checkpoint_model(
                accelerator,
                unet,
                args,
                ema_unet,
                noise_scheduler,
                full_pipeline_save_folder,
                repo,
                epoch,
            )

        # do not start new epoch before generation & checkpointing is done
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
