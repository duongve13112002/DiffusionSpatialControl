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
from modules.prompt_parser import FrozenCLIPEmbedderWithCustomWords
from torch import einsum
from torch.autograd.function import Function

from diffusers import DiffusionPipeline
from diffusers.utils import PIL_INTERPOLATION, is_accelerate_available
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor,is_compiled_module,is_torch_version
from diffusers.image_processor import VaeImageProcessor,PipelineImageInput
import modules.safe as _
from safetensors.torch import load_file
from diffusers import ControlNetModel
from PIL import Image
import torchvision.transforms as transforms
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, LMSDiscreteScheduler
from modules.u_net_condition_modify import UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.models import AutoencoderKL, ImageProjection, MultiAdapter, T2IAdapter
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    BaseOutput,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from  diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from packaging import version
from diffusers.configuration_utils import FrozenDict

def _preprocess_adapter_image(image, height, width):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        image = [np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])) for i in image]
        image = [
            i[None, ..., None] if i.ndim == 2 else i[None, ...] for i in image
        ]  # expand [h, w] or [h, w, c] to [b, h, w, c]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        if image[0].ndim == 3:
            image = torch.stack(image, dim=0)
        elif image[0].ndim == 4:
            image = torch.cat(image, dim=0)
        else:
            raise ValueError(
                f"Invalid image tensor! Expecting image tensor with 3 or 4 dimension, but recive: {image[0].ndim}"
            )
    return image

#t2i_adapter setup
def setup_model_t2i_adapter(class_name,adapter = None):
    if isinstance(adapter, (list, tuple)):
        adapter = MultiAdapter(adapter)
    class_name.adapter = adapter



def preprocessing_t2i_adapter(class_name,image,width,height,adapter_conditioning_scale,num_images_per_prompt = 1):
    if isinstance(class_name.adapter, MultiAdapter):
        adapter_input = []
        for one_image in image:
            one_image = _preprocess_adapter_image(one_image, height, width)
            one_image = one_image.to(device=class_name.device, dtype=class_name.adapter.dtype)
            adapter_input.append(one_image)
    else:
        adapter_input = _preprocess_adapter_image(image, height, width)
        adapter_input = adapter_input.to(device=class_name.device, dtype=class_name.adapter.dtype)

    if isinstance(class_name.adapter, MultiAdapter):
        adapter_state = class_name.adapter(adapter_input, adapter_conditioning_scale)
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v
    else:
        adapter_state = class_name.adapter(adapter_input)
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v * adapter_conditioning_scale


    if num_images_per_prompt > 1:
        for k, v in enumerate(adapter_state):
            adapter_state[k] = v.repeat(num_images_per_prompt, 1, 1, 1)
    if class_name.do_classifier_free_guidance:
        for k, v in enumerate(adapter_state):
            adapter_state[k] = torch.cat([v] * 2, dim=0)
    return adapter_state


def default_height_width(class_name, height, width, image):
    # NOTE: It is possible that a list of images have different
    # dimensions for each image, so just checking the first image
    # is not _exactly_ correct, but it is simple.
    while isinstance(image, list):
        image = image[0]

    if height is None:
        if isinstance(image, PIL.Image.Image):
            height = image.height
        elif isinstance(image, torch.Tensor):
            height = image.shape[-2]

        # round down to nearest multiple of `self.adapter.downscale_factor`
        height = (height // class_name.adapter.downscale_factor) * class_name.adapter.downscale_factor

    if width is None:
        if isinstance(image, PIL.Image.Image):
            width = image.width
        elif isinstance(image, torch.Tensor):
            width = image.shape[-1]

        # round down to nearest multiple of `self.adapter.downscale_factor`
        width = (width // class_name.adapter.downscale_factor) * class_name.adapter.downscale_factor

    return height, width