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

import math
import os
from argparse import Namespace
from math import ceil
from pathlib import Path
from shutil import rmtree

import numpy as np
import torch
import torch.nn.functional as F
import torch_fidelity
from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.training_utils import EMAModel
from PIL.Image import Image
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

import wandb
from src.custom_embedding import CustomEmbedding
from src.custom_pipeline_stable_diffusion_img2img.custom_pipeline_stable_diffusion_img2img import (
    CustomStableDiffusionImg2ImgPipeline,
)

from .utils_misc import extract_into_tensor, is_it_best_model, split


def resume_from_checkpoint(
    args,
    logger: MultiProcessAdapter,
    accelerator,
    num_update_steps_per_epoch: int,
    global_step: int,
):
    if args.resume_from_checkpoint != "latest":
        path = os.path.basename(args.resume_from_checkpoint)
        path = os.path.join(args.output_dir, "checkpoints", path)
    else:
        # Get the most recent checkpoint
        chckpnts_dir = Path(args.output_dir, "checkpoints")
        if not Path.exists(chckpnts_dir) and accelerator.is_main_process:
            logger.warning(
                f"No 'checkpoints' directory found in output_dir {args.output_dir}; creating one."
            )
            os.makedirs(chckpnts_dir)
        accelerator.wait_for_everyone()
        dirs = os.listdir(chckpnts_dir)
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        path = Path(chckpnts_dir, dirs[-1]).as_posix() if len(dirs) > 0 else None

    if path is None:
        accelerator.print(
            f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
        )
        args.resume_from_checkpoint = None
        first_epoch, resume_step = 0, 0
    else:
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(path)
        global_step = int(path.split("_")[-1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % (
            num_update_steps_per_epoch * args.gradient_accumulation_steps
        )
    return first_epoch, resume_step, global_step


def get_training_setup(
    args,
    accelerator,
    train_dataloader,
    logger: MultiProcessAdapter,
    dataset,
    noise_scheduler: DDIMScheduler,
):
    """
    Returns
    -------
    - `Tuple[int, List[int]]`
        A tuple containing:
        - the total number of update steps per epoch,
        - the list of actual evaluation batch sizes *for this process*
    """
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # distribute "batches" for image generation
    # tot_nb_eval_batches is the total number of batches for image generation
    # *across all GPUs* and *per class*
    tot_nb_eval_batches = ceil(args.nb_generated_images / args.eval_batch_size)
    glob_eval_bs = [args.eval_batch_size] * (tot_nb_eval_batches - 1)
    glob_eval_bs += [
        args.nb_generated_images - args.eval_batch_size * (tot_nb_eval_batches - 1)
    ]
    nb_proc = accelerator.num_processes
    actual_eval_batch_sizes_for_this_process = split(
        glob_eval_bs, nb_proc, accelerator.process_index
    )
    accelerator.print("")
    logger.info(
        "\033[1m******************************* Running training *******************************\033[0m"
    )
    logger.info(f"  Num examples = \033[1m{len(dataset)}\033[0m")
    logger.info(f"  Num epochs = \033[1m{args.num_epochs}\033[0m")
    logger.info(
        f"  Instantaneous batch size per device = \033[1m{args.train_batch_size}\033[0m"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = \033[1m{total_batch_size}\033[0m"
    )
    logger.info(
        f"  Gradient Accumulation steps = \033[1m{args.gradient_accumulation_steps}\033[0m"
    )
    logger.info(f"  Total optimization steps = \033[1m{max_train_steps}\033[0m")
    logger.info(
        f"  Num steps between checkpoints = \033[1m{args.checkpointing_steps}\033[0m"
    )
    tot_nb_chckpts = max_train_steps // args.checkpointing_steps
    logger.info(f"  Num checkpoints during training = \033[1m{tot_nb_chckpts}\033[0m")
    logger.info(f"  Components to train = \033[1m{args.components_to_train}\033[0m")
    logger.info(f"  Output dir = \033[1m{args.output_dir}\033[0m")
    logger.info(
        f"  Probability of unconditional pass = \033[1m{args.proba_uncond}\033[0m"
    )
    logger.info(f"  Learning rate = \033[1m{args.learning_rate}\033[0m")
    logger.info(f"  Num generated images = \033[1m{args.nb_generated_images}\033[0m")
    logger.info(
        f"  Num diffusion discretization steps = \033[1m{noise_scheduler.config.num_train_timesteps}\033[0m"
    )
    logger.info(
        f"  Num diffusion generation steps = \033[1m{args.num_inference_steps}\033[0m"
    )

    return (
        num_update_steps_per_epoch,
        actual_eval_batch_sizes_for_this_process,
    )


def perform_training_epoch(
    denoiser_model: UNet2DConditionModel,
    autoencoder_model: AutoencoderKL,
    num_update_steps_per_epoch: int,
    accelerator,
    epoch: int,
    train_dataloader,
    args,
    first_epoch: int,
    resume_step: int,
    noise_scheduler,
    global_step: int,
    optimizer,
    lr_scheduler,
    ema_model: None | EMAModel,
    logger: MultiProcessAdapter,
    class_embedding: CustomEmbedding,
    do_uncond_pass_across_all_procs: torch.BoolTensor,
):
    # set model to train mode
    denoiser_model.train()

    # give me a pretty progress bar 🤩
    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Epoch {epoch}")

    # iterate over all batches
    for step, batch in enumerate(train_dataloader):
        # Skip steps until we reach the resumed step
        if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update()
            continue

        if args.use_pytorch_loader:
            clean_images = batch[0]
            class_labels = batch[1]
        else:
            clean_images = batch["images"]
            class_labels = batch["class_labels"]

        # Convert images to latent space
        # autoencoder_model = accelerator.unwrap_model(autoencoder_model)
        latents = autoencoder_model.module.encode(clean_images).latent_dist.sample()
        latents = latents * autoencoder_model.module.config.scaling_factor
        # TODO: what is this ↑?

        # Sample noise that we'll add to the images
        noise = torch.randn(latents.shape).to(latents.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Unconditional pass for CLF guidance training
        do_unconditional_pass: bool = do_uncond_pass_across_all_procs[step].item()

        with accelerator.accumulate(denoiser_model):
            loss_value = _forward_backward_pass(
                args=args,
                accelerator=accelerator,
                global_step=global_step,
                denoiser_model=denoiser_model,
                noisy_images=noisy_latents,
                timesteps=timesteps,
                class_labels=class_labels,
                noise=noise,
                noise_scheduler=noise_scheduler,
                clean_images=latents,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                class_embedding=class_embedding,
                do_unconditional_pass=do_unconditional_pass,
                logger=logger,
            )

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            global_step = _syn_training_state(
                args,
                ema_model,
                denoiser_model,
                progress_bar,
                global_step,
                accelerator,
                logger,
            )

        # log some values & define W&B alert
        logs = {
            "loss": loss_value,
            "lr": lr_scheduler.get_last_lr()[0],
            "step": global_step,
            "epoch": epoch,
        }
        if args.use_ema:
            logs["ema_decay"] = ema_model.cur_decay_value
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        if math.isnan(loss_value) and accelerator.is_main_process:
            msg = f"Loss is NaN at step {global_step} / epoch {epoch}"
            wandb.alert(
                title="NaN loss",
                text=msg,
                level=wandb.AlertLevel.ERROR,
                wait_duration=21600,  # 6 hours
            )
            logger.error(msg)
    progress_bar.close()

    # wait for everybody at each end of training epoch
    accelerator.wait_for_everyone()

    return global_step


def _forward_backward_pass(
    args,
    accelerator,
    global_step,
    denoiser_model: UNet2DConditionModel,
    noisy_images,
    timesteps,
    class_labels: torch.Tensor,
    noise,
    noise_scheduler,
    clean_images,
    optimizer,
    lr_scheduler,
    class_embedding: CustomEmbedding,
    do_unconditional_pass: bool,
    logger: MultiProcessAdapter,
):
    # classifier-free guidance: randomly discard conditioning to train unconditionally
    accelerator.log(
        {
            "unconditional step": int(do_unconditional_pass),
        },
        step=global_step,
    )
    if do_unconditional_pass:
        class_emb = torch.zeros(
            (
                class_labels.shape[0],
                77,  # TODO: hardcoded dumb repeat, fix this
                args.class_embedding_dim,
            )
        ).to(accelerator.device)
    else:
        class_emb = class_embedding(class_labels)
        # TODO: hardcoded dumb cast, fix this
        (bs, ed) = class_emb.shape
        class_emb = class_emb.reshape(bs, 1, ed)
        padding = torch.zeros_like(class_emb).repeat(1, 76, 1).to(class_emb.device)
        class_emb = torch.cat([class_emb, padding], dim=1)

    # use the class embedding as the "encoder hidden state"
    encoder_hidden_states = class_emb

    # obtain the model prediction
    model_output = denoiser_model(
        sample=noisy_images,
        timestep=timesteps,
        encoder_hidden_states=encoder_hidden_states,
    ).sample

    if args.prediction_type == "epsilon":
        loss = F.mse_loss(model_output, noise)
    elif args.prediction_type == "sample":
        alpha_t = extract_into_tensor(
            noise_scheduler.alphas_cumprod,
            timesteps,
            (clean_images.shape[0], 1, 1, 1),
        )
        snr_weights = alpha_t / (1 - alpha_t)
        loss = snr_weights * F.mse_loss(
            model_output, clean_images, reduction="none"
        )  # use SNR weighting from distillation paper
        loss = loss.mean()
    elif noise_scheduler.config.prediction_type == "v_prediction":
        velocity = noise_scheduler.get_velocity(clean_images, noise, timesteps)
        loss = F.mse_loss(model_output, velocity)
    else:
        raise ValueError(f"Unsupported prediction type: {args.prediction_type}")

    accelerator.backward(loss)

    if accelerator.sync_gradients:
        params_to_clip = list(denoiser_model.parameters()) + list(
            class_embedding.parameters()
        )
        gard_norm = accelerator.clip_grad_norm_(params_to_clip, 1.0)
        accelerator.log({"gradient norm": gard_norm}, step=global_step)
        if gard_norm.isnan().any() and accelerator.is_main_process:
            msg = f"Gradient norm is NaN at step {global_step}"
            wandb.alert(
                title="NaN gradient norm",
                text=msg,
                level=wandb.AlertLevel.ERROR,
                wait_duration=21600,  # 6 hours
            )
            logger.error(msg)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.item()


def _syn_training_state(
    args, ema_model, model, progress_bar, global_step, accelerator, logger
) -> int:
    if args.use_ema:
        ema_model.step(model.parameters())
    progress_bar.update()
    global_step += 1

    if global_step % args.checkpointing_steps == 0:
        # time to save a checkpoint!
        if accelerator.is_main_process:
            save_path = Path(
                args.output_dir, "checkpoints", f"checkpoint_{global_step}"
            )
            accelerator.save_state(save_path.as_posix())
            logger.info(f"Checkpointed step {global_step} at {save_path}")
            # Delete old checkpoints if needed
            checkpoints_list = os.listdir(Path(args.output_dir, "checkpoints"))
            nb_checkpoints = len(checkpoints_list)
            if nb_checkpoints > args.checkpoints_total_limit:
                to_del = sorted(checkpoints_list, key=lambda x: int(x.split("_")[1]))[
                    : -args.checkpoints_total_limit
                ]
                if len(to_del) > 1:
                    logger.warning(
                        f"\033[1;33m=====> MORE THAN 1 CHECKPOINT TO DELETE:\033[0m\n {to_del}"
                    )
                for dir in to_del:
                    logger.info(f"Deleting {dir}...")
                    rmtree(Path(args.output_dir, "checkpoints", dir))

    return global_step


@torch.no_grad()
def generate_samples_and_compute_metrics(
    args,
    accelerator,
    autoencoder_model: AutoencoderKL,
    denoiser_model: UNet2DConditionModel,
    class_embedding: CustomEmbedding,
    ema_model,
    noise_scheduler,
    image_generation_tmp_save_folder,
    actual_eval_batch_sizes_for_this_process,
    epoch,
    global_step,
    full_pipeline_save_path,
    nb_classes: int,
    logger: MultiProcessAdapter,
    dataset,
    best_metric: float | None,
) -> tuple[bool | None, float | None]:
    progress_bar = tqdm(
        total=len(actual_eval_batch_sizes_for_this_process),
        desc=f"Generating images on process {accelerator.process_index}",
        disable=not accelerator.is_local_main_process,
    )
    denoiser_model = accelerator.unwrap_model(denoiser_model)
    class_embedding = accelerator.unwrap_model(class_embedding)
    autoencoder_model = accelerator.unwrap_model(autoencoder_model)

    if args.use_ema:
        ema_model.store(denoiser_model.parameters())
        ema_model.copy_to(denoiser_model.parameters())

    # load the previously downloaded pipeline
    pipeline = CustomStableDiffusionImg2ImgPipeline.from_pretrained(
        full_pipeline_save_path,
        local_files_only=True,  # do not pull from hub during training
        # then override modified components:
        vae=autoencoder_model,
        unet=denoiser_model,
        scheduler=noise_scheduler,
        class_embedding=class_embedding,
    ).to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # set manual seed in order to observe the outputs produced from the same Gaussian noise
    generator = torch.Generator(device=accelerator.device).manual_seed(5742877512)

    # list containing the values of the main metric (defined by user) for each class
    # (only instantiated on main process as metrics are themselves computed on main process only)
    if accelerator.is_main_process:
        main_metric_values = []

    # Run pipeline in inference (sample random noise and denoise)
    # Generate args.nb_generated_images in batches *per class*
    # for metrics computation
    for class_label in range(nb_classes):
        # clean image_generation_tmp_save_folder (it's per-class)
        if accelerator.is_local_main_process:
            if os.path.exists(image_generation_tmp_save_folder):
                rmtree(image_generation_tmp_save_folder)
            os.makedirs(image_generation_tmp_save_folder, exist_ok=False)
        accelerator.wait_for_everyone()

        # get class name
        class_name = dataset.classes[class_label]

        # pretty bar
        postfix_str = f"Current class: {class_name} ({class_label+1}/{nb_classes})"
        progress_bar.set_postfix_str(postfix_str)

        # loop over eval batches for this process
        for batch_idx, actual_bs in enumerate(actual_eval_batch_sizes_for_this_process):
            images: np.ndarray = pipeline(  # pyright: ignore reportGeneralTypeIssues
                image=None,  # start generation from pure Gaussian noise
                latent_shape=(actual_bs, 4, 16, 16),  # TODO: parametrize
                class_labels=torch.tensor(
                    [class_label] * actual_bs, device=accelerator.device
                ).long(),
                strength=1,  # no noise will actually be added if image=None
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_factor,
                generator=generator,
                output_type="np",
                device=accelerator.device,
            )

            # save images to disk
            images_pil: list[Image] = pipeline.numpy_to_pil(images)
            for idx, img in enumerate(images_pil):
                tot_idx = args.eval_batch_size * batch_idx + idx
                filename = (
                    f"process_{accelerator.local_process_index}_sample_{tot_idx}.png"
                )
                assert not Path(
                    filename
                ).exists(), "Rewriting existing generated image file!"
                img.save(
                    Path(
                        image_generation_tmp_save_folder,
                        filename,
                    )
                )

            # denormalize the images and save to logger if first batch
            # (first batch of main process only to prevent "logger overflow")
            if batch_idx == 0 and accelerator.is_main_process:
                images_processed = (images * 255).round().astype("uint8")
                accelerator.log(
                    {
                        f"generated_samples/{class_name}": [
                            wandb.Image(img) for img in images_processed[:50]
                        ],
                        "epoch": epoch,
                    },
                    step=global_step,
                )
            progress_bar.update()

        # wait for all processes to finish generating+saving images
        # before computing metrics
        accelerator.wait_for_everyone()

        # now compute metrics on main process
        if accelerator.is_main_process:
            logger.info(f"Computing metrics for class {class_name}...")
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=args.train_data_dir + "/train" + f"/{class_name}",
                input2=image_generation_tmp_save_folder.as_posix(),
                cuda=True,
                batch_size=args.eval_batch_size,
                isc=args.compute_isc,
                fid=args.compute_fid,
                kid=args.compute_kid,
                verbose=False,
                cache_root=".fidelity_cache",
                input1_cache_name=f"{class_name}",  # forces caching
                rng_seed=42,
                kid_subset_size=args.kid_subset_size,
            )

            for metric_name, metric_value in metrics_dict.items():
                accelerator.log(
                    {
                        f"{metric_name}/{class_name}": metric_value,
                    },
                    step=global_step,
                )
                if metric_name == args.main_metric:
                    main_metric_values.append(metric_value)

            # clear the tmp folder for this class
            rmtree(image_generation_tmp_save_folder)

        # resync everybody for each class
        accelerator.wait_for_everyone()

    # is it the best model to date?
    if accelerator.is_main_process:
        best_model_to_date, best_metric = is_it_best_model(
            main_metric_values, best_metric
        )
    else:
        best_model_to_date = None
        best_metric = None

    if args.use_ema:
        ema_model.restore(denoiser_model.parameters())

    return best_model_to_date, best_metric


def save_pipeline(
    accelerator: Accelerator,
    autoencoder_model: AutoencoderKL,
    denoiser_model: UNet2DConditionModel,
    class_embedding: CustomEmbedding,
    args: Namespace,
    ema_model: EMAModel | None,
    noise_scheduler: DDIMScheduler,
    full_pipeline_save_folder: Path,
    repo,
    epoch: int,
    logger: MultiProcessAdapter,
    first_save: bool = False,
):
    if (
        first_save
        and full_pipeline_save_folder.exists()
        and len(list(full_pipeline_save_folder.iterdir())) != 0
    ):
        logger.warn(
            "Not overwriting already populated pipeline save folder at run start"
        )
        return

    denoiser_model = accelerator.unwrap_model(denoiser_model)
    class_embedding = accelerator.unwrap_model(class_embedding)

    if args.use_ema:
        ema_model.store(denoiser_model.parameters())
        ema_model.copy_to(denoiser_model.parameters())

    if isinstance(autoencoder_model, DistributedDataParallel):
        # for some reason .save_pretrained does not unwrap the vae?
        autoencoder_model = autoencoder_model.module

    pipeline = CustomStableDiffusionImg2ImgPipeline(
        vae=autoencoder_model,
        unet=denoiser_model,
        scheduler=noise_scheduler,
        class_embedding=class_embedding,
    )

    # save to full_pipeline_save_folder (≠ initial_pipeline_save_path...)
    logger.info(f"Saving full pipeline to {full_pipeline_save_folder} at epoch {epoch}")
    pipeline.save_pretrained(full_pipeline_save_folder)

    if args.use_ema:
        ema_model.restore(denoiser_model.parameters())

    if args.push_to_hub:
        repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)
