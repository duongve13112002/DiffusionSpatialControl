import importlib
import inspect
import math
from pathlib import Path
import re
from collections import defaultdict
from typing import List, Optional, Union
import cv2
import time
import k_diffusion
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from modules.external_k_diffusion import CompVisDenoiser, CompVisVDenoiser
from torch import einsum
from torch.autograd.function import Function

from diffusers.utils import PIL_INTERPOLATION, is_accelerate_available
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor,is_compiled_module
from diffusers.image_processor import VaeImageProcessor,PipelineImageInput
import modules.safe as _
from safetensors.torch import load_file
from diffusers import ControlNetModel
from PIL import Image
import torchvision.transforms as transforms
from diffusers.models import AutoencoderKL, ImageProjection
from modules.ip_adapter import IPAdapterMixin
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
import gc
from modules.t2i_adapter import preprocessing_t2i_adapter,default_height_width
from modules.encoder_prompt_modify import encode_prompt_function
from modules.encode_region_map_function import encode_region_map
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.loaders import LoraLoaderMixin

def get_image_size(image):
    height, width = None, None
    if isinstance(image, Image.Image):
        return image.size  
    elif isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        return (width, height)  
    elif torch.is_tensor(image):
        #RGB image
        if len(image.shape) == 3:
            _, height, width = image.shape
        else:
            height, width = image.shape
        return (width, height) 
    else:
        raise TypeError("The image must be an instance of PIL.Image, numpy.ndarray, or torch.Tensor.")


def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

# from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class ModelWrapper:
    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def apply_model(self, *args, **kwargs):
        if len(args) == 3:
            encoder_hidden_states = args[-1]
            args = args[:2]
        if kwargs.get("cond", None) is not None:
            encoder_hidden_states = kwargs.pop("cond")
        return self.model(
            *args, encoder_hidden_states=encoder_hidden_states, **kwargs
        ).sample


class StableDiffusionPipeline(IPAdapterMixin,DiffusionPipeline,StableDiffusionMixin,LoraLoaderMixin):

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        feature_extractor,
        image_encoder = None,
    ):
        super().__init__()

        # get correct sigmas from LMS
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.controlnet = None
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        self.setup_unet(self.unet)

    def setup_unet(self, unet):
        unet = unet.to(self.device)
        model = ModelWrapper(unet, self.scheduler.alphas_cumprod)
        if self.scheduler.config.prediction_type == "v_prediction":
            self.k_diffusion_model = CompVisVDenoiser(model)
        else:
            self.k_diffusion_model = CompVisDenoiser(model)

    def get_scheduler(self, scheduler_type: str):
        library = importlib.import_module("k_diffusion")
        sampling = getattr(library, "sampling")
        return getattr(sampling, scheduler_type)

    def encode_image(self, image, device, num_images_per_prompt, output_hidden_states=None):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        if output_hidden_states:
            image_enc_hidden_states = self.image_encoder(image, output_hidden_states=True).hidden_states[-2]
            image_enc_hidden_states = image_enc_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_enc_hidden_states = self.image_encoder(
                torch.zeros_like(image), output_hidden_states=True
            ).hidden_states[-2]
            uncond_image_enc_hidden_states = uncond_image_enc_hidden_states.repeat_interleave(
                num_images_per_prompt, dim=0
            )
            return image_enc_hidden_states, uncond_image_enc_hidden_states
        else:
            image_embeds = self.image_encoder(image).image_embeds
            image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            uncond_image_embeds = torch.zeros_like(image_embeds)

            return image_embeds, uncond_image_embeds


    def prepare_ip_adapter_image_embeds(
        self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt, do_classifier_free_guidance
    ):
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            image_embeds = []
            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )
                single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
                single_negative_image_embeds = torch.stack(
                    [single_negative_image_embeds] * num_images_per_prompt, dim=0
                )

                if do_classifier_free_guidance:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                    single_image_embeds = single_image_embeds.to(device)

                image_embeds.append(single_image_embeds)
        else:
            repeat_dims = [1]
            image_embeds = []
            for single_image_embeds in ip_adapter_image_embeds:
                if do_classifier_free_guidance:
                    single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                    single_negative_image_embeds = single_negative_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_negative_image_embeds.shape[1:]))
                    )
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
                else:
                    single_image_embeds = single_image_embeds.repeat(
                        num_images_per_prompt, *(repeat_dims * len(single_image_embeds.shape[1:]))
                    )
                image_embeds.append(single_image_embeds)

        return image_embeds

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [
            self.unet,
            self.text_encoder,
            self.vae,
            self.safety_checker,
        ]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents.to(self.device, dtype=self.vae.dtype)
        #latents = 1 / 0.18215 * latents
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image


    def _default_height_width(self, height, width, image):
        if isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[3]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[2]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    @property
    def do_classifier_free_guidance(self):
        return self._do_classifier_free_guidance and self.unet.config.time_cond_proj_dim is None

    def setup_controlnet(self,controlnet):
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)
        self.register_modules(
            controlnet=controlnet,
        )

    def preprocess_controlnet(self,controlnet_conditioning_scale,control_guidance_start,control_guidance_end,image,width,height,num_inference_steps,batch_size,num_images_per_prompt):
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = False or global_pool_conditions

         # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                device=self._execution_device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    device=self._execution_device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 7.2 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(num_inference_steps):
            keeps = [
                1.0 - float(i / num_inference_steps < s or (i + 1) / num_inference_steps > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        return image,controlnet_keep,guess_mode,controlnet_conditioning_scale



    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if latents is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(
                    shape, generator=generator, device="cpu", dtype=dtype
                ).to(device)
            else:
                latents = torch.randn(
                    shape, generator=generator, device=device, dtype=dtype
                )
        else:
            # if latents.shape != shape:
            #     raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        return latents

    def preprocess(self, image):
        if isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, PIL.Image.Image):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            w, h = image[0].size
            w, h = map(lambda x: x - x % 8, (w, h))  # resize to integer multiple of 8

            image = [
                np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[
                    None, :
                ]
                for i in image
            ]
            image = np.concatenate(image, axis=0)
            image = np.array(image).astype(np.float32) / 255.0
            image = image.transpose(0, 3, 1, 2)
            image = 2.0 * image - 1.0
            image = torch.from_numpy(image)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, dim=0)
        return image

    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        image = self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        #image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image
    
    def numpy_to_pil(self,images):
        r"""
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        #images = (images * 255).round().astype("uint8")
        images = np.clip((images * 255).round(), 0, 255).astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def latent_to_image(self,latent,output_type):
        image = self.decode_latents(latent)
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        if len(image) > 1:
            return image
        return image[0]


    @torch.no_grad()
    def img2img(
        self,
        prompt: Union[str, List[str]],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[torch.Generator] = None,
        image: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        latents=None,
        strength=1.0,
        region_map_state=None,
        sampler_name="",
        sampler_opt={},
        start_time=-1,
        timeout=180,
        scale_ratio=8.0,
        latent_processing = 0,
        weight_func = lambda w, sigma, qk: w * sigma * qk.std(),
        upscale=False,
        upscale_x: float = 2.0,
        upscale_method: str = "bicubic",
        upscale_antialias: bool = False,
        upscale_denoising_strength: int = 0.7,
        width = None,
        height = None,
        seed = 0,
        sampler_name_hires="",
        sampler_opt_hires= {},
        latent_upscale_processing = False,
        ip_adapter_image = None,
        control_img = None,
        controlnet_conditioning_scale = None,
        control_guidance_start = None,
        control_guidance_end = None,
        image_t2i_adapter : Optional[PipelineImageInput] = None,
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,
        adapter_conditioning_factor: float = 1.0,
        guidance_rescale: float = 0.0,
        cross_attention_kwargs = None,
        clip_skip = None,
        long_encode = 0,
        num_images_per_prompt = 1,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
    ):
        if isinstance(sampler_name, str):
            sampler = self.get_scheduler(sampler_name)
        else:
            sampler = sampler_name
        if height is None:
            _,height = get_image_size(image)
            height = int((height // 8)*8) 
        if width is None:
            width,_ = get_image_size(image)
            width = int((width // 8)*8)  

        if image_t2i_adapter is not None:
            height, width = default_height_width(self,height, width, image_t2i_adapter)
        if image is not None:
            image = self.preprocess(image)
            image = image.to(self.vae.device, dtype=self.vae.dtype)

            init_latents = self.vae.encode(image).latent_dist.sample(generator)
            latents = 0.18215 * init_latents

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        latents = latents.to(device, dtype=self.unet.dtype)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        self._do_classifier_free_guidance = False if guidance_scale <= 1.0 else True
        # 3. Encode input prompt

        text_embeddings, negative_prompt_embeds, text_input_ids = encode_prompt_function(
            self,
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            lora_scale = lora_scale,
            clip_skip = clip_skip,
            long_encode = long_encode,
        )

        if self.do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, text_embeddings])

        text_embeddings = text_embeddings.to(self.unet.dtype)

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        sigmas = self.get_sigmas(num_inference_steps, sampler_opt).to(
            text_embeddings.device, dtype=text_embeddings.dtype
        )

        sigma_sched = sigmas[t_start:]

        noise = randn_tensor(
            latents.shape,
            generator=generator,
            device=device,
            dtype=text_embeddings.dtype,
        )
        latents = latents.to(device)
        latents = latents + noise * (sigma_sched[0]**2 + 1) ** 0.5
        steps_denoising = len(sigma_sched)
        # 5. Prepare latent variables
        self.k_diffusion_model.sigmas = self.k_diffusion_model.sigmas.to(latents.device)
        self.k_diffusion_model.log_sigmas = self.k_diffusion_model.log_sigmas.to(
            latents.device
        )

        region_state = encode_region_map(
            self,
            region_map_state,
            width = width,
            height = height,
            num_images_per_prompt = num_images_per_prompt,
            text_ids=text_input_ids,
        )
        if cross_attention_kwargs is None:
            cross_attention_kwargs ={}

        controlnet_conditioning_scale_copy = controlnet_conditioning_scale.copy() if isinstance(controlnet_conditioning_scale, list) else controlnet_conditioning_scale 
        control_guidance_start_copy =  control_guidance_start.copy() if isinstance(control_guidance_start, list) else control_guidance_start 
        control_guidance_end_copy =  control_guidance_end.copy() if isinstance(control_guidance_end, list) else control_guidance_end 
        guess_mode = False

        if self.controlnet is not None:
            img_control,controlnet_keep,guess_mode,controlnet_conditioning_scale = self.preprocess_controlnet(controlnet_conditioning_scale,control_guidance_start,control_guidance_end,control_img,width,height,len(sigma_sched),batch_size,num_images_per_prompt)

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )
        if latent_processing == 1:
            latents_process = [self.latent_to_image(latents,output_type)]
        lst_latent_sigma = []
        step_control = -1
        adapter_state = None
        adapter_sp_count = []
        if image_t2i_adapter is not None:
            adapter_state = preprocessing_t2i_adapter(self,image_t2i_adapter,width,height,adapter_conditioning_scale,1)
        def model_fn(x, sigma):
            nonlocal step_control,lst_latent_sigma,adapter_sp_count

            if start_time > 0 and timeout > 0:
                assert (time.time() - start_time) < timeout, "inference process timed out"

            latent_model_input = torch.cat([x] * 2) if self.do_classifier_free_guidance else x
            
            region_prompt = {
                "region_state": region_state,
                "sigma": sigma[0],
                "weight_func": weight_func,
              }
            cross_attention_kwargs["region_prompt"] = region_prompt

            if latent_model_input.dtype != text_embeddings.dtype:
                latent_model_input = latent_model_input.to(text_embeddings.dtype)
            ukwargs = {}

            down_intrablock_additional_residuals = None
            if adapter_state is not None:
                if len(adapter_sp_count) < int( steps_denoising* adapter_conditioning_factor):
                    down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
                else:
                    down_intrablock_additional_residuals = None
            sigma_string_t2i = str(sigma.item())
            if sigma_string_t2i not in adapter_sp_count:
                adapter_sp_count.append(sigma_string_t2i)

            if self.controlnet is not None :
                sigma_string = str(sigma.item())
                if sigma_string not in lst_latent_sigma:
                    step_control+=1
                    lst_latent_sigma.append(sigma_string)

                if isinstance(controlnet_keep[step_control], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[step_control])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[step_control]
                
                down_block_res_samples = None
                mid_block_res_sample = None
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input / ((sigma**2 + 1) ** 0.5),
                        self.k_diffusion_model.sigma_to_t(sigma),
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=img_control,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                ukwargs ={
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual":mid_block_res_sample,
                }
            
            noise_pred = self.k_diffusion_model(
                latent_model_input, sigma, cond=text_embeddings,cross_attention_kwargs = cross_attention_kwargs,down_intrablock_additional_residuals = down_intrablock_additional_residuals,added_cond_kwargs=added_cond_kwargs, **ukwargs
            )

            
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
            if latent_processing == 1:
                latents_process.append(self.latent_to_image(noise_pred,output_type))
            return noise_pred

        sampler_args = self.get_sampler_extra_args_i2i(sigma_sched,len(sigma_sched),sampler_opt,latents,seed, sampler)
        latents = sampler(model_fn, latents, **sampler_args)
        self.maybe_free_model_hooks()
        torch.cuda.empty_cache()
        gc.collect()
        if upscale:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            target_height = int(height * upscale_x // vae_scale_factor  )* 8
            target_width = int(width * upscale_x // vae_scale_factor)*8
            
            latents = torch.nn.functional.interpolate(
                latents,
                size=(
                    int(target_height // vae_scale_factor),
                    int(target_width // vae_scale_factor),
                ),
                mode=upscale_method,
                antialias=upscale_antialias,
            )
            latent_reisze= self.img2img(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                latents=latents,
                strength=upscale_denoising_strength,
                sampler_name=sampler_name_hires,
                sampler_opt=sampler_opt_hires,
                region_map_state=region_map_state,
                latent_processing = latent_upscale_processing,
                width = int(target_width),
                height = int(target_height),
                seed = seed,
                ip_adapter_image = ip_adapter_image,
                control_img = control_img,
                controlnet_conditioning_scale = controlnet_conditioning_scale_copy,
                control_guidance_start = control_guidance_start_copy,
                control_guidance_end = control_guidance_end_copy,
                image_t2i_adapter= image_t2i_adapter,
                adapter_conditioning_scale = adapter_conditioning_scale,
                adapter_conditioning_factor = adapter_conditioning_factor,
                guidance_rescale = guidance_rescale,
                cross_attention_kwargs = cross_attention_kwargs,
                clip_skip = clip_skip,
                long_encode = long_encode,
                num_images_per_prompt = num_images_per_prompt,
            )
            if latent_processing == 1:
                latents_process= latents_process+latent_reisze
                return latents_process
            torch.cuda.empty_cache()
            gc.collect()
            return latent_reisze
  
        if latent_processing == 1:
            return latents_process
        self.maybe_free_model_hooks()
        torch.cuda.empty_cache()
        gc.collect()
        return [self.latent_to_image(latents,output_type)]

    def get_sigmas(self, steps, params):
        discard_next_to_last_sigma = params.get("discard_next_to_last_sigma", False)
        steps += 1 if discard_next_to_last_sigma else 0

        if params.get("scheduler", None) == "karras":
            sigma_min, sigma_max = (
                self.k_diffusion_model.sigmas[0].item(),
                self.k_diffusion_model.sigmas[-1].item(),
            )
            sigmas = k_diffusion.sampling.get_sigmas_karras(
                n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=self.device
            )
        elif params.get("scheduler", None) == "exponential":
            sigma_min, sigma_max = (
                self.k_diffusion_model.sigmas[0].item(),
                self.k_diffusion_model.sigmas[-1].item(),
            )
            sigmas = k_diffusion.sampling.get_sigmas_exponential(
                n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=self.device
            )
        elif params.get("scheduler", None) == "polyexponential":
            sigma_min, sigma_max = (
                self.k_diffusion_model.sigmas[0].item(),
                self.k_diffusion_model.sigmas[-1].item(),
            )
            sigmas = k_diffusion.sampling.get_sigmas_polyexponential(
                n=steps, sigma_min=sigma_min, sigma_max=sigma_max, device=self.device
            )
        else:
            sigmas = self.k_diffusion_model.get_sigmas(steps)

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas

    def create_noise_sampler(self, x, sigmas, p,seed):
        """For DPM++ SDE: manually create noise sampler to enable deterministic results across different batch sizes"""

        from k_diffusion.sampling import BrownianTreeNoiseSampler
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        #current_iter_seeds = p.all_seeds[p.iteration * p.batch_size:(p.iteration + 1) * p.batch_size]
        return BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)

    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/48a15821de768fea76e66f26df83df3fddf18f4b/modules/sd_samplers.py#L454
    def get_sampler_extra_args_t2i(self, sigmas, eta, steps,sampler_opt,latents,seed, func):
        extra_params_kwargs = {}

        if "eta" in inspect.signature(func).parameters:
            extra_params_kwargs["eta"] = eta

        if "sigma_min" in inspect.signature(func).parameters:
            extra_params_kwargs["sigma_min"] = sigmas[0].item()
            extra_params_kwargs["sigma_max"] = sigmas[-1].item()

        if "n" in inspect.signature(func).parameters:
            extra_params_kwargs["n"] = steps
        else:
            extra_params_kwargs["sigmas"] = sigmas
        if sampler_opt.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(latents, sigmas, steps,seed)
            extra_params_kwargs['noise_sampler'] = noise_sampler
        if sampler_opt.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'
        
        return extra_params_kwargs

    # https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/48a15821de768fea76e66f26df83df3fddf18f4b/modules/sd_samplers.py#L454
    def get_sampler_extra_args_i2i(self, sigmas,steps,sampler_opt,latents,seed, func):
        extra_params_kwargs = {}

        if "sigma_min" in inspect.signature(func).parameters:
            ## last sigma is zero which isn't allowed by DPM Fast & Adaptive so taking value before last
            extra_params_kwargs["sigma_min"] = sigmas[-2]

        if "sigma_max" in inspect.signature(func).parameters:
            extra_params_kwargs["sigma_max"] = sigmas[0]

        if "n" in inspect.signature(func).parameters:
            extra_params_kwargs["n"] = len(sigmas) - 1

        if "sigma_sched" in inspect.signature(func).parameters:
            extra_params_kwargs["sigma_sched"] = sigmas

        if "sigmas" in inspect.signature(func).parameters:
            extra_params_kwargs["sigmas"] = sigmas
        if sampler_opt.get('brownian_noise', False):
            noise_sampler = self.create_noise_sampler(latents, sigmas, steps,seed)
            extra_params_kwargs['noise_sampler'] = noise_sampler
        if sampler_opt.get('solver_type', None) == 'heun':
            extra_params_kwargs['solver_type'] = 'heun'

        return extra_params_kwargs

    @torch.no_grad()
    def txt2img(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_steps: Optional[int] = 1,
        upscale=False,
        upscale_x: float = 2.0,
        upscale_method: str = "bicubic",
        upscale_antialias: bool = False,
        upscale_denoising_strength: int = 0.7,
        region_map_state=None,
        sampler_name="",
        sampler_opt={},
        start_time=-1,
        timeout=180,
        latent_processing = 0,
        weight_func = lambda w, sigma, qk: w * sigma * qk.std(),
        seed = 0,
        sampler_name_hires= "",
        sampler_opt_hires= {},
        latent_upscale_processing = False,
        ip_adapter_image = None,
        control_img = None,
        controlnet_conditioning_scale = None,
        control_guidance_start = None,
        control_guidance_end = None,
        image_t2i_adapter : Optional[PipelineImageInput] = None,
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,
        adapter_conditioning_factor: float = 1.0,
        guidance_rescale: float = 0.0,
        cross_attention_kwargs = None,
        clip_skip = None,
        long_encode = 0,
        num_images_per_prompt = 1,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
    ):
        height, width = self._default_height_width(height, width, None)
        if isinstance(sampler_name, str):
            sampler = self.get_scheduler(sampler_name)
        else:
            sampler = sampler_name
        # 1. Check inputs. Raise error if not correct
        if image_t2i_adapter is not None:
            height, width = default_height_width(self,height, width, image_t2i_adapter)
        self.check_inputs(prompt, height, width, callback_steps)
        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        self._do_classifier_free_guidance = False if guidance_scale <= 1.0 else True
        # 3. Encode input prompt

        text_embeddings, negative_prompt_embeds, text_input_ids = encode_prompt_function(
            self,
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            lora_scale = lora_scale,
            clip_skip = clip_skip,
            long_encode = long_encode,
        )
        if self.do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, text_embeddings])

        # 3. Encode input prompt
        text_embeddings = text_embeddings.to(self.unet.dtype)

        # 4. Prepare timesteps
        sigmas = self.get_sigmas(num_inference_steps, sampler_opt).to(
            text_embeddings.device, dtype=text_embeddings.dtype
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        latents = latents * (sigmas[0]**2 + 1) ** 0.5
        steps_denoising = len(sigmas)
        self.k_diffusion_model.sigmas = self.k_diffusion_model.sigmas.to(latents.device)
        self.k_diffusion_model.log_sigmas = self.k_diffusion_model.log_sigmas.to(
            latents.device
        )

        region_state = encode_region_map(
            self,
            region_map_state,
            width = width,
            height = height,
            num_images_per_prompt = num_images_per_prompt,
            text_ids=text_input_ids,
        )
        if cross_attention_kwargs is None:
            cross_attention_kwargs ={}
        controlnet_conditioning_scale_copy = controlnet_conditioning_scale.copy() if isinstance(controlnet_conditioning_scale, list) else controlnet_conditioning_scale 
        control_guidance_start_copy =  control_guidance_start.copy() if isinstance(control_guidance_start, list) else control_guidance_start 
        control_guidance_end_copy =  control_guidance_end.copy() if isinstance(control_guidance_end, list) else control_guidance_end 
        guess_mode = False

        if self.controlnet is not None:
            img_control,controlnet_keep,guess_mode,controlnet_conditioning_scale = self.preprocess_controlnet(controlnet_conditioning_scale,control_guidance_start,control_guidance_end,control_img,width,height,num_inference_steps,batch_size,num_images_per_prompt)
        

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )
        if latent_processing == 1:
            latents_process = [self.latent_to_image(latents,output_type)]
        lst_latent_sigma = []
        step_control = -1
        adapter_state = None
        adapter_sp_count = []
        if image_t2i_adapter is not None:
            adapter_state = preprocessing_t2i_adapter(self,image_t2i_adapter,width,height,adapter_conditioning_scale,1)
        def model_fn(x, sigma):
            nonlocal step_control,lst_latent_sigma,adapter_sp_count

            if start_time > 0 and timeout > 0:
                assert (time.time() - start_time) < timeout, "inference process timed out"

            latent_model_input = torch.cat([x] * 2) if self.do_classifier_free_guidance else x
            region_prompt = {
                "region_state": region_state,
                "sigma": sigma[0],
                "weight_func": weight_func,
              }
            cross_attention_kwargs["region_prompt"] = region_prompt

            if latent_model_input.dtype != text_embeddings.dtype:
                latent_model_input = latent_model_input.to(text_embeddings.dtype)
            ukwargs = {}

            down_intrablock_additional_residuals = None
            if adapter_state is not None:
                if len(adapter_sp_count) < int( steps_denoising* adapter_conditioning_factor):
                    down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
                else:
                    down_intrablock_additional_residuals = None
            sigma_string_t2i = str(sigma.item())
            if sigma_string_t2i not in adapter_sp_count:
                adapter_sp_count.append(sigma_string_t2i)

            if self.controlnet is not None :
                sigma_string = str(sigma.item())
                if sigma_string not in lst_latent_sigma:
                    #sigmas_sp = sigma.detach().clone()
                    step_control+=1
                    lst_latent_sigma.append(sigma_string)

                if isinstance(controlnet_keep[step_control], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[step_control])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[step_control]
                
                down_block_res_samples = None
                mid_block_res_sample = None
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input / ((sigma**2 + 1) ** 0.5),
                        self.k_diffusion_model.sigma_to_t(sigma),
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=img_control,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                ukwargs ={
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual":mid_block_res_sample,
                }

            
            noise_pred = self.k_diffusion_model(
                latent_model_input, sigma, cond=text_embeddings,cross_attention_kwargs=cross_attention_kwargs,down_intrablock_additional_residuals=down_intrablock_additional_residuals,added_cond_kwargs=added_cond_kwargs, **ukwargs
            )

            
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
            if latent_processing == 1:
                latents_process.append(self.latent_to_image(noise_pred,output_type))
            return noise_pred
        extra_args = self.get_sampler_extra_args_t2i(
            sigmas, eta, num_inference_steps,sampler_opt,latents,seed, sampler
        )
        latents = sampler(model_fn, latents, **extra_args)
        self.maybe_free_model_hooks()
        torch.cuda.empty_cache()
        gc.collect()
        if upscale:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            target_height = int(height * upscale_x // vae_scale_factor  )* 8
            target_width = int(width * upscale_x // vae_scale_factor)*8
            latents = torch.nn.functional.interpolate(
                latents,
                size=(
                    int(target_height // vae_scale_factor),
                    int(target_width // vae_scale_factor),
                ),
                mode=upscale_method,
                antialias=upscale_antialias,
            )
            latent_reisze= self.img2img(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                latents=latents,
                strength=upscale_denoising_strength,
                sampler_name=sampler_name_hires,
                sampler_opt=sampler_opt_hires,
                region_map_state = region_map_state,
                latent_processing = latent_upscale_processing,
                width = int(target_width),
                height = int(target_height),
                seed = seed,
                ip_adapter_image = ip_adapter_image,
                control_img = control_img,
                controlnet_conditioning_scale = controlnet_conditioning_scale_copy,
                control_guidance_start = control_guidance_start_copy,
                control_guidance_end = control_guidance_end_copy,
                image_t2i_adapter= image_t2i_adapter,
                adapter_conditioning_scale = adapter_conditioning_scale,
                adapter_conditioning_factor = adapter_conditioning_factor,
                guidance_rescale = guidance_rescale,
                cross_attention_kwargs = cross_attention_kwargs,
                clip_skip = clip_skip,
                long_encode = long_encode,
                num_images_per_prompt = num_images_per_prompt,
            )
            if latent_processing == 1:
                latents_process= latents_process+latent_reisze
                return latents_process
            torch.cuda.empty_cache()
            gc.collect()
            return latent_reisze

        # 8. Post-processing
        if latent_processing == 1:
            return latents_process
        return [self.latent_to_image(latents,output_type)]


    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = self.vae.config.scaling_factor * image_latents

        return image_latents
    
    def prepare_mask_latents(
        self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        mask = mask.to(device=device, dtype=dtype)

        masked_image = masked_image.to(device=device, dtype=dtype)

        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self._encode_vae_image(masked_image, generator=generator)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(
                    "The passed mask and the required batch size don't match. Masks are supposed to be duplicated to"
                    f" a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number"
                    " of masks that you pass is divisible by the total requested batch size."
                )
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(
                    "The passed images and the required batch size don't match. Images are supposed to be duplicated"
                    f" to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed."
                    " Make sure the number of images that you pass is divisible by the total requested batch size."
                )
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return mask, masked_image_latents

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
        sigma_t = sigma * alpha_t

        return alpha_t, sigma_t

    def add_noise(self,init_latents_proper,noise,sigma):
        if isinstance(sigma, torch.Tensor) and sigma.numel() > 1:
            sigma,_ = sigma.sort(descending=True)
            sigma = sigma[0].item()
        init_latents_proper = init_latents_proper + sigma * noise
        return init_latents_proper

    def prepare_latents_inpating(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        sigma=None,
        is_strength_max=True,
        return_noise=False,
        return_image_latents=False,
    ):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if (image is None or sigma is None) and not is_strength_max:
            raise ValueError(
                "Since strength < 1. initial latents are to be initialised as a combination of Image + Noise."
                "However, either the image or the noise sigma has not been provided."
            )

        if return_image_latents or (latents is None and not is_strength_max):
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)

        if latents is None:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            # if strength is 1. then initialise the latents to noise, else initial to image + noise
            latents = noise if is_strength_max else self.add_noise(image_latents, noise, sigma)
            # if pure noise then scale the initial latents by the  Scheduler's init sigma
            latents = latents * (sigma.item()**2 + 1) ** 0.5 if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * (sigma.item()**2 + 1) ** 0.5

        outputs = (latents,)

        if return_noise:
            outputs += (noise,)

        if return_image_latents:
            outputs += (image_latents,)

        return outputs

    @torch.no_grad()
    def inpaiting(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_steps: Optional[int] = 1,
        upscale=False,
        upscale_x: float = 2.0,
        upscale_method: str = "bicubic",
        upscale_antialias: bool = False,
        upscale_denoising_strength: int = 0.7,
        region_map_state=None,
        sampler_name="",
        sampler_opt={},
        start_time=-1,
        timeout=180,
        latent_processing = 0,
        weight_func = lambda w, sigma, qk: w * sigma * qk.std(),
        seed = 0,
        sampler_name_hires= "",
        sampler_opt_hires= {},
        latent_upscale_processing = False,
        ip_adapter_image = None,
        control_img = None,
        controlnet_conditioning_scale = None,
        control_guidance_start = None,
        control_guidance_end = None,
        image_t2i_adapter : Optional[PipelineImageInput] = None,
        adapter_conditioning_scale: Union[float, List[float]] = 1.0,
        adapter_conditioning_factor: float = 1.0,
        guidance_rescale: float = 0.0,
        cross_attention_kwargs = None,
        clip_skip = None,
        long_encode = 0,
        num_images_per_prompt = 1,
        image: Union[torch.Tensor, PIL.Image.Image] = None,
        mask_image: Union[torch.Tensor, PIL.Image.Image] = None,
        masked_image_latents: torch.FloatTensor = None,
        padding_mask_crop: Optional[int] = None,
        strength: float = 1.0,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
    ):
        height, width = self._default_height_width(height, width, None)
        if isinstance(sampler_name, str):
            sampler = self.get_scheduler(sampler_name)
        else:
            sampler = sampler_name
        # 1. Check inputs. Raise error if not correct
        if image_t2i_adapter is not None:
            height, width = default_height_width(self,height, width, image_t2i_adapter)
        self.check_inputs(prompt, height, width, callback_steps)
        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.

        lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        self._do_classifier_free_guidance = False if guidance_scale <= 1.0 else True
        # 3. Encode input prompt

        text_embeddings, negative_prompt_embeds, text_input_ids = encode_prompt_function(
            self,
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            lora_scale = lora_scale,
            clip_skip = clip_skip,
            long_encode = long_encode,
        )
        if self.do_classifier_free_guidance:
            text_embeddings = torch.cat([negative_prompt_embeds, text_embeddings])

        text_embeddings = text_embeddings.to(self.unet.dtype)

        # 4. Prepare timesteps
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        sigmas = self.get_sigmas(num_inference_steps, sampler_opt).to(
            text_embeddings.device, dtype=text_embeddings.dtype
        )
        sigmas = sigmas[t_start:] if strength >= 0 and strength < 1.0 else sigmas
        is_strength_max = strength == 1.0


        # 5. Prepare mask, image, 
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = "fill"
        else:
            crops_coords = None
            resize_mode = "default"

        original_image = image
        init_image = self.image_processor.preprocess(
            image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode
        )
        init_image = init_image.to(dtype=torch.float32)

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        image_latents = None
        noise_inpaiting = None

        latents_outputs = self.prepare_latents_inpating(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
            image=init_image,
            sigma=sigmas[0],
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise_inpaiting, image_latents = latents_outputs
        else:
            latents, noise_inpaiting = latents_outputs

         # 7. Prepare mask latent variables
        mask_condition = self.mask_processor.preprocess(
            mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )

        if masked_image_latents is None:
            masked_image = init_image * (mask_condition < 0.5)
        else:
            masked_image = masked_image_latents

        mask, masked_image_latents = self.prepare_mask_latents(
            mask_condition,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            self.do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents+num_channels_masked_image+num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        steps_denoising = len(sigmas)
        self.k_diffusion_model.sigmas = self.k_diffusion_model.sigmas.to(latents.device)
        self.k_diffusion_model.log_sigmas = self.k_diffusion_model.log_sigmas.to(
            latents.device
        )

        region_state = encode_region_map(
            self,
            region_map_state,
            width = width,
            height = height,
            num_images_per_prompt = num_images_per_prompt,
            text_ids=text_input_ids,
        )
        if cross_attention_kwargs is None:
            cross_attention_kwargs ={}
        controlnet_conditioning_scale_copy = controlnet_conditioning_scale.copy() if isinstance(controlnet_conditioning_scale, list) else controlnet_conditioning_scale 
        control_guidance_start_copy =  control_guidance_start.copy() if isinstance(control_guidance_start, list) else control_guidance_start 
        control_guidance_end_copy =  control_guidance_end.copy() if isinstance(control_guidance_end, list) else control_guidance_end 
        guess_mode = False

        if self.controlnet is not None:
            img_control,controlnet_keep,guess_mode,controlnet_conditioning_scale = self.preprocess_controlnet(controlnet_conditioning_scale,control_guidance_start,control_guidance_end,control_img,width,height,num_inference_steps,batch_size,num_images_per_prompt)
        

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )
        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )
        if latent_processing == 1:
            latents_process = [self.latent_to_image(latents,output_type)]
        lst_latent_sigma = []
        step_control = -1
        adapter_state = None
        adapter_sp_count = []
        flag_add_noise_inpaiting = 0
        if image_t2i_adapter is not None:
            adapter_state = preprocessing_t2i_adapter(self,image_t2i_adapter,width,height,adapter_conditioning_scale,1)
        def model_fn(x, sigma):
            nonlocal step_control,lst_latent_sigma,adapter_sp_count,flag_add_noise_inpaiting

            if start_time > 0 and timeout > 0:
                assert (time.time() - start_time) < timeout, "inference process timed out"

            if num_channels_unet == 4 and flag_add_noise_inpaiting:
                init_latents_proper = image_latents
                if self.do_classifier_free_guidance:
                    init_mask, _ = mask.chunk(2)
                else:
                    init_mask = mask

                if sigma.item() > sigmas[-1].item():
                    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma.item())
                    init_latents_proper = alpha_t * init_latents_proper + sigma_t * noise_inpaiting

                rate_latent_timestep_sigma = (sigma**2 + 1) ** 0.5

                x  = ((1 - init_mask) * init_latents_proper + init_mask * x/ rate_latent_timestep_sigma ) * rate_latent_timestep_sigma

            non_inpainting_latent_model_input = (
                    torch.cat([x] * 2) if self.do_classifier_free_guidance else x
                )

            inpainting_latent_model_input = torch.cat(
                    [non_inpainting_latent_model_input,mask, masked_image_latents], dim=1
            ) if num_channels_unet == 9 else non_inpainting_latent_model_input
            region_prompt = {
                "region_state": region_state,
                "sigma": sigma[0],
                "weight_func": weight_func,
              }
            cross_attention_kwargs["region_prompt"] = region_prompt


            if non_inpainting_latent_model_input.dtype != text_embeddings.dtype:
                non_inpainting_latent_model_input = non_inpainting_latent_model_input.to(text_embeddings.dtype)

            if inpainting_latent_model_input.dtype != text_embeddings.dtype:
                inpainting_latent_model_input = inpainting_latent_model_input.to(text_embeddings.dtype)
            ukwargs = {}

            down_intrablock_additional_residuals = None
            if adapter_state is not None:
                if len(adapter_sp_count) < int( steps_denoising* adapter_conditioning_factor):
                    down_intrablock_additional_residuals = [state.clone() for state in adapter_state]
                else:
                    down_intrablock_additional_residuals = None
            sigma_string_t2i = str(sigma.item())
            if sigma_string_t2i not in adapter_sp_count:
                adapter_sp_count.append(sigma_string_t2i)

            if self.controlnet is not None :
                sigma_string = str(sigma.item())
                if sigma_string not in lst_latent_sigma:
                    step_control+=1
                    lst_latent_sigma.append(sigma_string)

                if isinstance(controlnet_keep[step_control], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[step_control])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[step_control]
                
                down_block_res_samples = None
                mid_block_res_sample = None
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                        non_inpainting_latent_model_input / ((sigma**2 + 1) ** 0.5),
                        self.k_diffusion_model.sigma_to_t(sigma),
                        encoder_hidden_states=text_embeddings,
                        controlnet_cond=img_control,
                        conditioning_scale=cond_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                if guess_mode and self.do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])
                ukwargs ={
                "down_block_additional_residuals": down_block_res_samples,
                "mid_block_additional_residual":mid_block_res_sample,
                }

            
            noise_pred = self.k_diffusion_model(
                inpainting_latent_model_input, sigma, cond=text_embeddings,cross_attention_kwargs=cross_attention_kwargs,down_intrablock_additional_residuals=down_intrablock_additional_residuals,added_cond_kwargs=added_cond_kwargs, **ukwargs
            )

            
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
             

            if latent_processing == 1:
                latents_process.append(self.latent_to_image(noise_pred,output_type))
            flag_add_noise_inpaiting = 1
            return noise_pred
        extra_args = self.get_sampler_extra_args_t2i(
            sigmas, eta, num_inference_steps,sampler_opt,latents,seed, sampler
        )
        latents = sampler(model_fn, latents, **extra_args)
        self.maybe_free_model_hooks()
        torch.cuda.empty_cache()
        gc.collect()
        if upscale:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            target_height = int(height * upscale_x // vae_scale_factor  )* 8
            target_width = int(width * upscale_x // vae_scale_factor)*8
            latents = torch.nn.functional.interpolate(
                latents,
                size=(
                    int(target_height // vae_scale_factor),
                    int(target_width // vae_scale_factor),
                ),
                mode=upscale_method,
                antialias=upscale_antialias,
            )
            
            latent_reisze= self.img2img(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                latents=latents,
                strength=upscale_denoising_strength,
                sampler_name=sampler_name_hires,
                sampler_opt=sampler_opt_hires,
                region_map_state = region_map_state,
                latent_processing = latent_upscale_processing,
                width = int(target_width),
                height = int(target_height),
                seed = seed,
                ip_adapter_image = ip_adapter_image,
                control_img = control_img,
                controlnet_conditioning_scale = controlnet_conditioning_scale_copy,
                control_guidance_start = control_guidance_start_copy,
                control_guidance_end = control_guidance_end_copy,
                image_t2i_adapter= image_t2i_adapter,
                adapter_conditioning_scale = adapter_conditioning_scale,
                adapter_conditioning_factor = adapter_conditioning_factor,
                guidance_rescale = guidance_rescale,
                cross_attention_kwargs = cross_attention_kwargs,
                clip_skip = clip_skip,
                long_encode = long_encode,
                num_images_per_prompt = num_images_per_prompt,
            )
            if latent_processing == 1:
                latents_process= latents_process+latent_reisze
                return latents_process
            torch.cuda.empty_cache()
            gc.collect()
            return latent_reisze

        # 8. Post-processing
        if latent_processing == 1:
            return latents_process
        return [self.latent_to_image(latents,output_type)]



