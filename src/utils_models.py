from argparse import Namespace
from pathlib import Path

from accelerate import Accelerator
from accelerate.logging import MultiProcessAdapter
from diffusers import (
    DDIMScheduler,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)

from .cond_unet_2d import CustomCondUNet2DModel
from .custom_embedding import CustomEmbedding
from .custom_pipeline_stable_diffusion_img2img import (
    CustomStableDiffusionImg2ImgPipeline,
)
from .pipeline_conditional_ddim import ConditionalDDIMPipeline


# -------------------------------------------- Main Function --------------------------------------------
def load_initial_pipeline(
    args: Namespace,
    initial_pipeline_save_folder: Path,
    logger: MultiProcessAdapter,
    nb_classes: int,
    accelerator: Accelerator,
) -> CustomStableDiffusionImg2ImgPipeline | ConditionalDDIMPipeline:
    """Loads and customizes the initial pipeline (either pretrained or not).

    At the end of this function the entire pipeline should be in its 'final' initial state,
    i.e. all custom params applied and models ready to be trained right away.
    """
    logger.info("Loading initial pipeline...")

    if args.pretrained_model_name_or_path is not None:
        match args.model_type:
            case "StableDiffusion":
                pipeline = _load_custom_SD(
                    args, initial_pipeline_save_folder, logger, nb_classes, accelerator
                )
            case "DDIM":
                raise NotImplementedError("TODO: test this")
                # pipeline = _load_custom_DDIM(args, initial_pipeline_save_folder, logger)
    else:
        match args.model_type:
            case "StableDiffusion":
                # pipeline = _load_custom_SD_from_scratch(...)
                raise NotImplementedError(
                    "TODO: allow using SD without using a pretrained model"
                )
            case "DDIM":
                pipeline = _load_custom_DDIM_from_scratch(args, logger, accelerator)

    logger.info("Initial pipeline loaded")

    return pipeline


# ---------------------------------------------- Pipelines ----------------------------------------------
# All pipelines should at least call _load_and_override_noise_scheduler
def _load_custom_SD(
    args: Namespace,
    initial_pipeline_save_folder: Path,
    logger: MultiProcessAdapter,
    nb_classes: int,
    accelerator: Accelerator,
) -> CustomStableDiffusionImg2ImgPipeline:
    # 1. Download the pipeline in initial_pipeline_save_folder
    initial_pipeline_save_path = StableDiffusionImg2ImgPipeline.download(
        args.pretrained_model_name_or_path,
        cache_dir=initial_pipeline_save_folder,
        local_files_only=True,
    )

    # 2. Customize the pipeline components
    # denoiser
    if args.learn_denoiser_from_scratch:
        denoiser_model_config = UNet2DConditionModel.load_config(
            Path(initial_pipeline_save_path, "unet", "config.json"),
        )
        denoiser_model: UNet2DConditionModel = UNet2DConditionModel.from_config(
            denoiser_model_config,
        )
    else:
        denoiser_model = None  # hack to be able *not* to override the denoiser
    denoiser_model_kwarg = {"denoiser_model": denoiser_model} if denoiser_model else {}
    # noise scheduler
    noise_scheduler = _load_and_override_noise_scheduler(
        args, logger, initial_pipeline_save_path, accelerator
    )
    # custom class embedding: create a custom class inheriting from diffusers.ModelMixin
    # in order to use Hugging Face's routines
    class_embedding = CustomEmbedding(nb_classes, args.class_embedding_dim)

    # 3 Create the final pipeline
    pipeline = CustomStableDiffusionImg2ImgPipeline.from_pretrained(
        initial_pipeline_save_path,
        local_files_only=True,
        # vae=autoencoder_model, # the VAE is never modified at this point
        scheduler=noise_scheduler,
        class_embedding=class_embedding,
        **denoiser_model_kwarg,
    )
    return pipeline


def _load_custom_DDIM(
    args: Namespace,
    initial_pipeline_save_folder: Path,
    logger: MultiProcessAdapter,
    accelerator: Accelerator,
):
    # 1. Download the pipeline in initial_pipeline_save_folder
    initial_pipeline_save_path = ConditionalDDIMPipeline.download(
        args.pretrained_model_name_or_path,
        cache_dir=initial_pipeline_save_folder,
    )

    # 2. Customize the pipeline components
    # noise scheduler
    noise_scheduler = _load_and_override_noise_scheduler(
        args, logger, initial_pipeline_save_path, accelerator
    )

    # 3 Create the final pipeline
    pipeline = ConditionalDDIMPipeline.from_pretrained(
        initial_pipeline_save_path, local_files_only=True, scheduler=noise_scheduler
    )

    return pipeline


def _load_custom_DDIM_from_scratch(
    args: Namespace, logger: MultiProcessAdapter, accelerator: Accelerator
) -> ConditionalDDIMPipeline:
    # Load denoiser from config
    denoiser_model_config = CustomCondUNet2DModel.load_config(
        args.denoiser_config_path,
    )
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
        wandb_tracker.config["denoiser_model_config"] = denoiser_model_config
    denoiser_model: CustomCondUNet2DModel = CustomCondUNet2DModel.from_config(
        denoiser_model_config,
    )
    # Load noise scheduler from config & overide it
    noise_scheduler = _load_and_override_noise_scheduler(
        args, logger, None, accelerator
    )
    # Assemble pipeline
    pipeline = ConditionalDDIMPipeline(unet=denoiser_model, scheduler=noise_scheduler)
    return pipeline


# ---------------------------------------------- Components ---------------------------------------------
def _load_and_override_noise_scheduler(
    args: Namespace,
    logger: MultiProcessAdapter,
    initial_pipeline_save_path: Path | None,
    accelerator: Accelerator,
) -> DDIMScheduler:
    # if relevant args are None then use the "pretrained" values
    noise_scheduler_kwargs = [
        "num_train_timesteps",
        "beta_start",
        "beta_end",
        "beta_schedule",
        "prediction_type",
    ]
    noise_scheduler_kwargs_vals = {
        k: v
        for k, v in vars(args).items()
        if k in noise_scheduler_kwargs and v is not None
    }  # only overwrite if not None
    if noise_scheduler_kwargs_vals != {}:
        logger.warning(
            f"\033[1;33mWILL OVERWRITE THE FOLLOWING NOISE SCHEDULER KWARGS:\033[0m\n {noise_scheduler_kwargs_vals}\n"
        )

    # now load scheduler
    if args.pretrained_model_name_or_path is not None:
        noise_scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(
            initial_pipeline_save_path,
            subfolder="scheduler",
            local_files_only=True,
            **noise_scheduler_kwargs_vals,
        )
    else:
        noise_scheduler_config = DDIMScheduler.load_config(
            args.noise_scheduler_config_path,
        )
        if accelerator.is_main_process:
            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
            wandb_tracker.config["noise_scheduler_config"] = noise_scheduler_config
        noise_scheduler: DDIMScheduler = DDIMScheduler.from_config(
            noise_scheduler_config,
            **noise_scheduler_kwargs_vals,
        )

    return noise_scheduler
