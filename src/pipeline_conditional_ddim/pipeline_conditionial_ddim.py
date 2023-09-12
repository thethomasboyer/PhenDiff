# Copyright 2023 The HuggingFace Team & Thomas Boyer. All rights reserved.
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

from typing import List, Literal, Optional, Tuple, Union

import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DDIMScheduler
from diffusers.utils import is_accelerate_available, randn_tensor

DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5


class ConditionalDDIMPipeline(DiffusionPipeline):
    r"""
    A conditional version of the DDIM pipeline.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler: DDIMScheduler):
        super().__init__()

        # make sure scheduler can always be converted to DDIM
        scheduler: DDIMScheduler = DDIMScheduler.from_config(scheduler.config)  # type: ignore

        self.register_modules(unet=unet, scheduler=scheduler)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offload all models to CPU to reduce memory usage with a low impact on performance. Moves one whole model at a
        time to the GPU when its `forward` method is called, and the model remains in GPU until the next model runs.
        Memory savings are lower than using `enable_sequential_cpu_offload`, but performance is much better due to the
        iterative execution of the `unet`.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("Could not import accelerate.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        _, hook = cpu_offload_with_hook(self.unet, device, prev_module_hook=None)

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

    def check_inputs(
        self,
        class_labels: Optional[torch.Tensor] = None,
        class_emb: Optional[torch.Tensor] = None,
        w: Optional[Union[int, float, torch.Tensor]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        frac_diffusion_skipped: Optional[float] = None,
        start_image: Optional[torch.Tensor] = None,
    ) -> None:
        assert class_labels is None or (
            isinstance(class_labels, torch.Tensor) and class_labels.ndim == 1
        ), "class_labels must be a 1D tensor of shape (batch_size,) if not None."

        assert class_emb is None or (
            isinstance(class_emb, torch.Tensor) and class_emb.ndim == 2
        ), "class_emb must be a 2D tensor of shape (batch_size, emb_dim) if not None."

        assert (
            class_labels is None or class_emb is None
        ), "Cannot pass both class_labels and class_emb."

        batch_size = (
            class_labels.shape[0] if class_labels is not None else class_emb.shape[0]
        )

        assert (
            isinstance(w, float)
            or isinstance(w, int)
            or w is None
            or (w.ndim == 1 and batch_size == w.shape[0])
        ), "w must be a 1D tensor of shape (batch_size,) if not None and not a single int/float."

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size} through class conditioning. Make sure the batch size matches the length of the generators."
            )

        assert (frac_diffusion_skipped is not None and start_image is not None) or (
            frac_diffusion_skipped is None and start_image is None
        ), "Either pass both frac_diffusion_skipped and start_image or none of them."

        if frac_diffusion_skipped is not None:
            assert (
                isinstance(frac_diffusion_skipped, float)
                or isinstance(frac_diffusion_skipped, int)
            ) and 0 <= frac_diffusion_skipped <= 1, "frac_diffusion_skipped must be a float (or int) between 0 and 1; got {frac_diffusion_skipped}."

    @torch.no_grad()
    def __call__(
        self,
        class_labels: torch.Tensor | None,
        class_emb: torch.Tensor | None = None,
        w: int | float | torch.Tensor | None = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        start_image: Optional[torch.Tensor] = None,
        add_forward_noise_to_image: bool = True,
        frac_diffusion_skipped: Optional[float] = None,
        guidance_eqn: Literal["imagen", "CFG"] = "imagen",
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            class_labels (`torch.Tensor` or None):
                The class labels to condition on. Should be a tensor of shape `(batch_size,)` or `None` if `class_emb` is directly given.
            class_emb (`torch.Tensor` or None):
                The class embeddings to condition on. Should be None if class_labels are passed or a tensor of shape `(batch_size, emb_dim)` otherwise.
            w (`int` or `float` or `torch.Tensor` or None):
                The guidance factor. Should be None (then falls back to default), or a int/float or a tensor of shape `(batch_size,)` of postive value(s).
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            eta (`float`, *optional*, defaults to 0.0):
                The eta parameter which controls the scale of the variance (0 is DDIM and 1 is one type of DDPM).
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            use_clipped_model_output (`bool`, *optional*, defaults to `None`):
                if `True` or `False`, see documentation for `DDIMScheduler.step`. If `None`, nothing is passed
                downstream to the scheduler. So use `None` for schedulers which don't support this argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
            start_image (`torch.Tensor`, *optional*, defaults to `None`):
                A starting image to use for the diffusion process. If `None`, a random image is generated.
                `frac_diffusion_skipped` must be passed if `start_image` is not `None`.
            add_forward_noise_to_image (`bool`, *optional*, defaults to `True`)
                Whether to add noise to the input image or not. If `True`, the noising is done in accordance with
                `frac_diffusion_skipped` (so `1 - frac_diffusion_skipped` of the forward diffusion process is performed).
            frac_diffusion_skipped (`float`, *optional*, defaults to `None`):
                The fraction of the diffusion process to skip. Must be passed if `start_image` is not `None`.
                Should be between 0 and 1. 0 therefore means full diffusion trajectory.
            guidance_eqn (`Literal["imagen", "CFG"]`, *optional*, defaults to `"imagen"`):
                What guidance equation to use. Can be either that of the [Imagen](https://arxiv.org/pdf/2205.11487.pdf#subsection.2.2) paper
                or that of the original [Classifier-Free Diffusion Guidance](https://arxiv.org/pdf/2207.12598.pdf#equation.3.6) paper.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        # Checks
        self.check_inputs(
            class_labels,
            class_emb,
            w,
            generator,
            frac_diffusion_skipped,
            start_image,
        )

        # Misc.
        if num_inference_steps is None:
            # None means default value
            num_inference_steps = DEFAULT_NUM_INFERENCE_STEPS

        batch_size = (
            class_labels.shape[0] if class_labels is not None else class_emb.shape[0]
        )

        # Device
        device = self._execution_device

        # Sample Gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                *self.unet.config.sample_size,
            )

        # Starting point
        if start_image is not None:
            image = start_image
        else:
            image = randn_tensor(
                image_shape,
                generator=generator,
                device=device,
                dtype=self.unet.dtype,
            )

        # Set step values
        self.scheduler.set_timesteps(num_inference_steps)
        # use ony end of diffusion if asked
        if frac_diffusion_skipped is not None:
            # we keep the last (1 - frac_diffusion_skipped)*100% of timesteps
            init_timestep = self.scheduler.config.num_train_timesteps * (
                1 - frac_diffusion_skipped
            )
            # generation goes from T to 0, so we keep the timesteps *smaller* than init_timestep
            timesteps = self.scheduler.timesteps[
                self.scheduler.timesteps <= init_timestep
            ]
        else:
            timesteps = self.scheduler.timesteps

        # Noise images
        if add_forward_noise_to_image:
            noise = randn_tensor(
                image.shape, generator=generator, device=device, dtype=image.dtype
            )
            image = self.scheduler.add_noise(
                image, noise, timesteps[0].repeat(batch_size)
            )

        # CLF
        if w is None:
            w = DEFAULT_GUIDANCE_SCALE

        do_classifier_free_guidance = (
            isinstance(w, torch.Tensor)
            or (guidance_eqn == "imagen" and w > 1)
            or (guidance_eqn == "CFG" and w > 0)
        )

        for t in self.progress_bar(timesteps):
            # TODO: do the cond & uncond passes at once like for SD!
            # 1. predict noise model_output
            cond_output = self.unet(
                sample=image,
                timestep=t,
                class_labels=class_labels,
                class_emb=class_emb,
            ).sample

            # 2. Form the classifier-free guided score
            if do_classifier_free_guidance:
                # unconditionally predict noise model_output
                uncond_output = self.unet(
                    sample=image,
                    timestep=t,
                    class_labels=None,
                    class_emb=torch.zeros((batch_size, self.unet.time_embed_dim)).to(
                        device
                    ),
                ).sample

                # cast w to outputs shape if applicable
                if isinstance(w, torch.Tensor):
                    w = w.view(-1, 1, 1, 1)

                # combine predictions
                match guidance_eqn:
                    case "imagen":
                        guided_score = uncond_output + w * (cond_output - uncond_output)
                    case "CFG":
                        guided_score = cond_output + w * (cond_output - uncond_output)
                    case _:
                        raise ValueError(
                            f"Unknown guidance equation '{guidance_eqn}'; should be 'imagen' or 'CFG'"
                        )

            else:
                guided_score = cond_output

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                guided_score,
                t,
                image,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
            ).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
