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

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from warnings import warn

import numpy as np
import PIL
import torch
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromCkptMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
)
from packaging import version

from src.custom_embedding import CustomEmbedding

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class CustomStableDiffusionImg2ImgPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromCkptMixin
):
    r"""
    Pipeline for *class*-guided image to image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]
        - *LoRA*: [`loaders.LoraLoaderMixin.load_lora_weights`]
        - *Ckpt*: [`loaders.FromCkptMixin.from_ckpt`]

    as well as the following saving methods:
        - *LoRA*: [`loaders.LoraLoaderMixin.save_lora_weights`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        class_embedding (`CustomEmbedding`):
            The frozen embedding layer to use for the class conditioning.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        class_embedding: CustomEmbedding,
    ):
        super().__init__()

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(
            unet.config, "_diffusers_version"
        ) and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse(
            "0.9.0.dev0"
        )
        is_unet_sample_size_less_64 = (
            hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        )
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate(
                "sample_size<64", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            class_embedding=class_embedding,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher"
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_model_cpu_offload
    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                "`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher."
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook
            )

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_class(
        self,
        class_labels: Optional[Union[int, List[int], torch.Tensor]],
        device,
        do_classifier_free_guidance,
        class_labels_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ) -> torch.FloatTensor:
        r"""
        Encodes the class label into an embedding space.

        Args:
            class_labels (`int` or `List[int]` or `torch.Tensor`, *optional*):
                class indexes to be encoded; if Tensor should be 1D
            device: (`torch.device`):
                torch device
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            class_labels_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated class embeddings. Can be used to easily tweak inputs. If not
                provided, class embeddings will be generated from `class_labels` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if class_labels is not None:
            if isinstance(class_labels, int):
                batch_size = 1
                class_labels = torch.tensor([class_labels]).long()
            elif isinstance(class_labels, list):
                batch_size = len(class_labels)
                class_labels = torch.tensor(class_labels).long()
            elif isinstance(class_labels, torch.Tensor):
                class_labels = class_labels.long()
                batch_size = class_labels.shape[0]
        else:
            batch_size = class_labels_embeds.shape[0]

        if class_labels_embeds is None:
            class_labels_embeds = self.class_embedding(class_labels)

        class_labels_embeds = class_labels_embeds.to(
            dtype=self.class_embedding.dtype, device=device
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_embeds = torch.zeros(
                (batch_size, self.unet.config.cross_attention_dim)
            ).to(class_labels.device)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            class_labels_embeds = torch.cat([uncond_embeds, class_labels_embeds])

        return class_labels_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        class_labels,
        strength,
        callback_steps,
        class_labels_embeds,
        latent_shape,
        image,
    ):
        if image is None and latent_shape is None:
            raise ValueError(
                "Either `image` or `latent_shape` must be provided as input."
            )

        if strength < 0 or strength > 1:
            raise ValueError(
                f"The value of strength should be in [0, 1] but is {strength}"
            )

        if image is None and strength != 1:
            warn(
                "`image` is None so the generation will start from pure Gaussian noise, "
                "but `strength` is not set to 1 so the denoising process will not run for the full denoising trajectory. "
                "This will produce images that are not fully denoised."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if class_labels is not None and class_labels_embeds is not None:
            raise ValueError(
                f"Cannot forward both `class_labels`: {class_labels} and `class_labels_embeds`: {class_labels_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif class_labels is None and class_labels_embeds is None:
            raise ValueError(
                "Provide either `class_labels` or `class_labels_embeds`. Cannot leave both `class_labels` and `class_labels_embeds` undefined."
            )
        elif class_labels is not None and (
            not isinstance(class_labels, int)
            and not isinstance(class_labels, list)
            and not isinstance(class_labels, torch.Tensor)
        ):
            raise ValueError(
                f"`class_labels` has to be of type `int` or `list` or `torch.Tensor` but is {type(class_labels)}"
            )

        if isinstance(class_labels, torch.Tensor) and class_labels.ndim != 1:
            raise ValueError("If a Tensor `class_labels` should be 1D")

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start

    def prepare_latents(
        self,
        image: Union[
            torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image], np.ndarray, None
        ],
        timestep,
        batch_size,
        dtype,
        device,
        latent_shape,
        generator=None,
    ):
        if (
            not isinstance(image, (torch.Tensor, PIL.Image.Image, list))
            and image is not None
        ):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image`, list, or `None`, but is {type(image)}"
            )

        # image=None means generation not conditioned on any original image;
        # so just generate from Gaussian noise
        if image is None:
            init_latents = torch.randn(latent_shape, device=device, dtype=dtype)
            return init_latents

        image = image.to(device=device, dtype=dtype)

        if image.shape[1] == 4:
            # image is already a latent vector
            # ugly hardcoded test...
            init_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            elif isinstance(generator, list):
                init_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                    for i in range(batch_size)
                ]
                init_latents = torch.cat(init_latents, dim=0)
            else:
                init_latents = self.vae.encode(image).latent_dist.sample(generator)

            init_latents = self.vae.config.scaling_factor * init_latents

        init_latents = torch.cat([init_latents], dim=0)

        noise = randn_tensor(
            init_latents.shape, generator=generator, device=device, dtype=dtype
        )

        # get latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)

        return init_latents

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[
            Union[
                torch.FloatTensor,
                PIL.Image.Image,
                List[PIL.Image.Image],
                np.ndarray,
            ]
        ] = None,
        latent_shape: Optional[Tuple[int, ...]] = None,
        class_labels: Optional[Union[int, List[int], torch.Tensor]] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        eta: Optional[float] = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        class_labels_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Union[List[PIL.Image.Image], np.ndarray]:
        r"""
        Generation method. Returns a list of PIL images or a numpy array corresponding to the generated images from the
        given initial `image`. If `image` is `None` the generation happens from scratch (i.e. from a pure random Gaussian noise latent),
        corresponding to conditional image generation instead of img2img translation.

        Args:
            image (`torch.FloatTensor`, `PIL.Image.Image`, `List[PIL.Image.Image]`, `np.ndarray`, or `None`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Can also accept image latents as `image`: if passing latents directly, it will not be encoded
                again. Can also accept `None`, in which case the generation will start from scratch.
            latent_shape (`Tuple[int, ...]`, *optional*):
                The shape of the latent vector to be generated if `image` is `None`. Ignored otherwise.
            class_labels (`int` or `List[int]` or `torch.Tensor`, *optional*):
                The class labels to guide the image generation. If not defined, one has to pass `class_labels_embeds`.
                instead.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`. If `image` is `None`, strength
                should be set to 1 (and not noise will actually be added to the generated, allready pure Gaussian noise).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the class labels,
                usually at the expense of lower image quality.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            class_labels_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated class embeddings. Can be used to easily tweak class inputs, *e.g.* class weighting. If not
                provided, class embeddings will be generated from `class_labels` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between 'pil' for
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or 'np' for `np.array`.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        Examples:

        Returns:
            `List[PIL.Image.Image]` or `np.ndarray`
        """
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            class_labels=class_labels,
            strength=strength,
            callback_steps=callback_steps,
            class_labels_embeds=class_labels_embeds,
            latent_shape=latent_shape,
            image=image,
        )

        # 2. Define call parameters
        if class_labels is not None and isinstance(class_labels, int):
            batch_size = 1
        elif class_labels is not None and isinstance(class_labels, list):
            batch_size = len(class_labels)
        elif class_labels is not None and isinstance(class_labels, torch.Tensor):
            batch_size = class_labels.size()
        else:
            batch_size = class_labels_embeds.shape[0]

        # This call to self._execution_device sometimes fails unexpectedly...
        # -> override self._execution_device with user input if needed
        try:
            device = self._execution_device
        except AttributeError as e:
            warn(
                f"Call to self._execution_device failed: {e}\nUsing `device`={device} input argument instead."
            )
            device = device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance, and for HF guidance_scale <= 1
        # also means no CLF
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        class_labels_embeds = self._encode_class(
            class_labels=class_labels,
            device=device,
            do_classifier_free_guidance=do_classifier_free_guidance,
            class_labels_embeds=class_labels_embeds,
            lora_scale=lora_scale,
        )

        # hack to match the expected encoder_hidden_states shape
        (bs, ed) = class_labels_embeds.shape
        class_labels_embeds = class_labels_embeds.reshape(bs, 1, ed)
        padding = (
            torch.zeros_like(class_labels_embeds)
            .repeat(1, 76, 1)
            .to(class_labels_embeds.device)
        )
        class_labels_embeds = torch.cat([class_labels_embeds, padding], dim=1)

        # 4. Preprocess image
        if image is not None:
            image = self.image_processor.preprocess(image)

        # 5. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size)

        # 6. Prepare latent variables
        latents = self.prepare_latents(
            image=image,
            timestep=latent_timestep,
            batch_size=batch_size,
            dtype=class_labels_embeds.dtype,
            device=device,
            latent_shape=latent_shape,
            generator=generator,
        )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                # this does nothing for (Inverse)DDIM
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=class_labels_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
        else:
            image = latents

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(
            image, output_type=output_type, do_denormalize=do_denormalize
        )

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image
