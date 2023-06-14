import os
from math import ceil
from pathlib import Path
from shutil import rmtree

import torch
import torch.nn.functional as F
import torch_fidelity
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version
from PIL.Image import Image
from tqdm.auto import tqdm

import wandb

from .pipeline_conditional_ddim import ConditionialDDIMPipeline
from .utils_misc import extract_into_tensor, split


def resume_from_checkpoint(
    args, logger, accelerator, num_update_steps_per_epoch, global_step
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
        path = Path(chckpnts_dir, dirs[-1]
                    ).as_posix() if len(dirs) > 0 else None

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


def get_training_setup(args, accelerator, train_dataloader, logger, dataset):
    """
    Returns
    -------
    - `Tuple[int, int, List[int]]`
        A tuple containing:
        - the total number of update steps per epoch,
        - the total number of batches for image generation across all GPUs and *per class*
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
        args.nb_generated_images - args.eval_batch_size *
        (tot_nb_eval_batches - 1)
    ]
    nb_proc = accelerator.num_processes
    actual_eval_batch_sizes_for_this_process = split(
        glob_eval_bs, nb_proc, accelerator.process_index
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    return (
        num_update_steps_per_epoch,
        tot_nb_eval_batches,
        actual_eval_batch_sizes_for_this_process,
    )


def perform_training_epoch(
    model,
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
    ema_model: None | EMAModel,
    logger,
):
    # set model to train mode
    model.train()

    # give me a pretty progress bar 🤩
    progress_bar = tqdm(
        total=num_update_steps_per_epoch,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in enumerate(train_dataloader):
        # Skip steps until we reach the resumed step
        if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
            if step % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
            continue

        if args.use_pytorch_loader:
            clean_images = batch[0]
            class_labels = batch[1]
        else:
            clean_images = batch["images"]
            class_labels = batch["class_labels"]

        # Sample noise that we'll add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(
            clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            loss_value = _forward_backward_pass(
                rng,
                args,
                accelerator,
                epoch,
                global_step,
                model,
                noisy_images,
                timesteps,
                class_labels,
                noise,
                noise_scheduler,
                clean_images,
                optimizer,
                lr_scheduler,
            )

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            global_step = _syn_training_state(
                args, ema_model, model, progress_bar, global_step, accelerator, logger
            )

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
    progress_bar.close()

    # wait for everybody at each end of training epoch
    accelerator.wait_for_everyone()

    return global_step


def _forward_backward_pass(
    rng,
    args,
    accelerator,
    epoch,
    global_step,
    model,
    noisy_images,
    timesteps,
    class_labels,
    noise,
    noise_scheduler,
    clean_images,
    optimizer,
    lr_scheduler,
):
    # classifier-free guidance: randomly discard conditioning to train unconditionally
    do_unconditional_pass = rng.random() < args.proba_uncond
    accelerator.log(
        {
            "unconditional step": int(do_unconditional_pass),
            "epoch": epoch,
        },
        step=global_step,
    )
    if do_unconditional_pass:
        class_emb = torch.zeros(
            (class_labels.shape[0], accelerator.unwrap_model(
                model).time_embed_dim)
        ).to(accelerator.device)
        class_labels = None
    else:
        class_emb = None

    # Predict the noise residual
    model_output = model(
        sample=noisy_images,
        timestep=timesteps,
        class_labels=class_labels,
        class_emb=class_emb,
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
    else:
        raise ValueError(
            f"Unsupported prediction type: {args.prediction_type}")

    accelerator.backward(loss)

    if accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.item()


def _syn_training_state(
    args, ema_model, model, progress_bar, global_step, accelerator, logger
) -> int:
    if args.use_ema:
        ema_model.step(model.parameters())
    progress_bar.update(1)
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
                        "More than 1 checkpoint to delete? Previous delete must have failed..."
                    )
                for dir in to_del:
                    rmtree(Path(args.output_dir, "checkpoints", dir))

    return global_step


def generate_samples_and_compute_metrics(
    args,
    accelerator,
    model,
    ema_model,
    noise_scheduler,
    image_generation_tmp_save_folder,
    dataset,
    actual_eval_batch_sizes_for_this_process,
    guidance_factor,
    epoch,
    global_step,
    nb_classes: int,
):
    progress_bar = tqdm(
        total=len(actual_eval_batch_sizes_for_this_process),
        desc=f"Generating images on process {accelerator.process_index}",
        # disable=not accelerator.is_local_main_process,
    )
    unet = accelerator.unwrap_model(model)

    if args.use_ema:
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())

    pipeline = ConditionialDDIMPipeline(
        unet=unet,
        scheduler=noise_scheduler,
    )
    pipeline.set_progress_bar_config(disable=True)

    # set manual seed in order to observe the "same" images
    generator = torch.Generator(device=pipeline.device).manual_seed(0)

    # run pipeline in inference (sample random noise and denoise)
    # -> generate args.nb_generated_images in batches *per class*
    # if uncond gen, no need to loop over classes for generation
    if args.proba_uncond == 1:
        nb_classes = 1
    for label in range(nb_classes):
        # clean image_generation_tmp_save_folder (it's per-class)
        if accelerator.is_local_main_process:
            if os.path.exists(image_generation_tmp_save_folder):
                rmtree(image_generation_tmp_save_folder)
            os.makedirs(image_generation_tmp_save_folder, exist_ok=False)
        accelerator.wait_for_everyone()

        # get class name
        class_name = dataset.classes[label]

        # pretty bar
        if args.proba_uncond != 1:
            postfix_str = f"Current class: {class_name} ({label+1}/{nb_classes})"
            progress_bar.set_postfix_str(postfix_str)

        # loop over eval batches for this process
        for batch_idx, actual_bs in enumerate(actual_eval_batch_sizes_for_this_process):
            if args.proba_uncond == 1:
                class_emb = torch.zeros(
                    (actual_bs, accelerator.unwrap_model(model).time_embed_dim)
                ).to(accelerator.device)
                class_labels = None
            else:
                class_labels = torch.full(
                    (actual_bs,), label, device=pipeline.device
                ).long()
                class_emb = None
            images = pipeline(
                class_labels,
                class_emb,
                guidance_factor,
                generator=generator,
                batch_size=actual_bs,
                num_inference_steps=args.ddim_num_inference_steps,
                output_type="numpy",
            ).images

            # save images to disk
            images_pil: list[Image] = pipeline.numpy_to_pil(images)
            for idx, img in enumerate(images_pil):
                tot_idx = args.eval_batch_size * batch_idx + idx
                filename = (
                    f"process_{accelerator.local_process_index}_sample_{tot_idx}.png"
                )
                assert not Path(filename).exists(
                ), "Rewriting existing generated image file!"
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

                if args.logger == "tensorboard":
                    if is_accelerate_version(">=", "0.17.0.dev0"):
                        tracker = accelerator.get_tracker(
                            "tensorboard", unwrap=True)
                    else:
                        tracker = accelerator.get_tracker("tensorboard")
                    tracker.add_images(
                        "test_samples",
                        images_processed.transpose(0, 3, 1, 2),
                        epoch,
                    )
                elif args.logger == "wandb":
                    # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
                    if args.proba_uncond == 1:
                        class_name = "unconditional"
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
        accelerator.wait_for_everyone()

        # Compute metrics
        if accelerator.is_main_process:
            if args.proba_uncond == 1:
                accelerator.print(
                    f"Skipping metrics computation (unconditional generation)... TODO: implement!")
            else:
                accelerator.print(
                    f"Computing metrics for class {class_name}...")
                metrics_dict = torch_fidelity.calculate_metrics(
                    input1=args.train_data_dir + "/train" + f"/{class_name}",
                    input2=image_generation_tmp_save_folder.as_posix(),
                    cuda=True,
                    batch_size=args.eval_batch_size,
                    # isc=True,
                    fid=True,
                    # kid=True,
                    # ppl=True,
                    verbose=False,
                    cache_root=".fidelity_cache",
                    input1_cache_name=f"{class_name}",  # forces caching
                    rng_seed=42,
                )
                for metric_name, metric_value in metrics_dict.items():
                    accelerator.log(
                        {
                            f"{metric_name}/{class_name}": metric_value,
                            "epoch": epoch,
                        },
                        step=global_step,
                    )

        # resync everybody for each class
        accelerator.wait_for_everyone()

    if args.use_ema:
        ema_model.restore(unet.parameters())


def checkpoint_model(
    accelerator,
    model,
    args,
    ema_model,
    noise_scheduler,
    full_pipeline_save_folder,
    repo,
    epoch,
):
    # save the model
    unet = accelerator.unwrap_model(model)

    if args.use_ema:
        ema_model.store(unet.parameters())
        ema_model.copy_to(unet.parameters())

    pipeline = ConditionialDDIMPipeline(
        unet=unet,
        scheduler=noise_scheduler,
    )

    pipeline.save_pretrained(full_pipeline_save_folder)

    if args.use_ema:
        ema_model.restore(unet.parameters())

    if args.push_to_hub:
        repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)
