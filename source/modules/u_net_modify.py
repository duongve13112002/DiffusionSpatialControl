from diffusers.loaders.unet import UNet2DConditionLoadersMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    delete_adapter_layers,
    is_accelerate_available,
    is_torch_version,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)

from modules.attention_modify import AttnProcessor,IPAdapterAttnProcessor,AttnProcessor2_0,IPAdapterAttnProcessor2_0
import torch
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.nn as nn
import inspect
import os
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
from diffusers.models.embeddings import (
    ImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)
from modules.embedding_ipadapter import IPAdapterFullImageProjection
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta
from huggingface_hub.utils import validate_hf_hub_args
from diffusers.loaders.single_file_utils import (
    convert_stable_cascade_unet_single_file_to_diffusers,
    infer_stable_cascade_single_file_config,
    load_single_file_model_checkpoint,
)
from diffusers.loaders.utils import AttnProcsLayers
if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)



class UNet2DConditionLoadersMixin_modify(UNet2DConditionLoadersMixin):
    def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict, low_cpu_mem_usage=False):
        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        updated_state_dict = {}
        image_projection = None
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext

        if "proj.weight" in state_dict:
            # IP-Adapter
            num_image_text_embeds = 4
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // 4

            with init_context():
                image_projection = ImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=clip_embeddings_dim,
                    num_image_text_embeds=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        elif "proj.0.weight" in state_dict:
            # IP-Adapter Full and Face ID
            clip_embeddings_dim_in = state_dict["proj.0.weight"].shape[1]
            clip_embeddings_dim_out = state_dict["proj.0.weight"].shape[0]
            multiplier = clip_embeddings_dim_out // clip_embeddings_dim_in
            norm_layer = "norm.weight" if "norm.weight" in state_dict else "proj.3.weight"
            cross_attention_dim = state_dict[norm_layer].shape[0]
            num_tokens = state_dict["proj.2.weight"].shape[0] // cross_attention_dim

            with init_context():
                image_projection = IPAdapterFullImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=clip_embeddings_dim_in,
                    mult=multiplier,
                    num_tokens=num_tokens,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                diffusers_name = diffusers_name.replace("proj.3", "norm")
                updated_state_dict[diffusers_name] = value

        else:
            # IP-Adapter Plus
            num_image_text_embeds = state_dict["latents"].shape[1]
            embed_dims = state_dict["proj_in.weight"].shape[1]
            output_dims = state_dict["proj_out.weight"].shape[0]
            hidden_dims = state_dict["latents"].shape[2]
            heads = state_dict["layers.0.0.to_q.weight"].shape[0] // 64

            with init_context():
                image_projection = IPAdapterPlusImageProjection(
                    embed_dims=embed_dims,
                    output_dims=output_dims,
                    hidden_dims=hidden_dims,
                    heads=heads,
                    num_queries=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("0.to", "2.to")
                diffusers_name = diffusers_name.replace("1.0.weight", "3.0.weight")
                diffusers_name = diffusers_name.replace("1.0.bias", "3.0.bias")
                diffusers_name = diffusers_name.replace("1.1.weight", "3.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.3.weight", "3.1.net.2.weight")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                else:
                    updated_state_dict[diffusers_name] = value

        if not low_cpu_mem_usage:
            image_projection.load_state_dict(updated_state_dict)
        else:
            load_model_dict_into_meta(image_projection, updated_state_dict, device=self.device, dtype=self.dtype)

        return image_projection

    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts, low_cpu_mem_usage=False):

        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
        for name in self.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = (
                    AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
                )
                attn_procs[name] = attn_processor_class()
            else:
                attn_processor_class = (
                    IPAdapterAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else IPAdapterAttnProcessor
                )
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        # IP-Adapter
                        num_image_text_embeds += [4]
                    elif "proj.0.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face and Face ID
                        if f"{key_id}.to_k_lora.down.weight" in state_dict["ip_adapter"]:
                            num_image_text_embeds += [4]
                        else:
                            num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

                with init_context():
                    attn_procs[name] = attn_processor_class(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=num_image_text_embeds,
                    )

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                if not low_cpu_mem_usage:
                    attn_procs[name].load_state_dict(value_dict)
                else:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(attn_procs[name], value_dict, device=device, dtype=dtype)

                key_id += 2

        lora_dicts = {}
        for key_id, name in enumerate(self.attn_processors.keys()):
            for i, state_dict in enumerate(state_dicts):
                if f"{key_id}.to_k_lora.down.weight" in state_dict["ip_adapter"]:
                    if i not in lora_dicts:
                        lora_dicts[i] = {}
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_k_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_k_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_q_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_q_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_v_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_v_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_k_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_k_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_q_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_q_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_v_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_v_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.up.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.up.weight"
                            ]
                        }
                    )

        return attn_procs, lora_dicts

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=False):
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]
        # Set encoder_hid_proj after loading ip_adapter weights,
        # because `IPAdapterPlusImageProjection` also has `attn_processors`.
        self.encoder_hid_proj = None

        attn_procs, lora_dict = self._convert_ip_adapter_attn_to_diffusers(
            state_dicts, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.set_attn_processor(attn_procs)

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
                state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"

        self.to(dtype=self.dtype, device=self.device)
        return lora_dict


class FromOriginalUNetMixin:
    """
    Load pretrained UNet model weights saved in the `.ckpt` or `.safetensors` format into a [`StableCascadeUNet`].
    """

    @classmethod
    @validate_hf_hub_args
    def from_single_file(cls, pretrained_model_link_or_path, **kwargs):
        r"""
        Instantiate a [`StableCascadeUNet`] from pretrained StableCascadeUNet weights saved in the original `.ckpt` or
        `.safetensors` format. The pipeline is set in evaluation mode (`model.eval()`) by default.

        Parameters:
            pretrained_model_link_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A link to the `.ckpt` file (for example
                      `"https://huggingface.co/<repo_id>/blob/main/<path_to_file>.ckpt"`) on the Hub.
                    - A path to a *file* containing all pipeline weights.
            config: (`dict`, *optional*):
                Dictionary containing the configuration of the model:
            torch_dtype (`str` or `torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                dtype is automatically derived from the model's weights.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to True, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to overwrite load and saveable variables of the model.

        """
        class_name = cls.__name__
        if class_name != "StableCascadeUNet":
            raise ValueError("FromOriginalUNetMixin is currently only compatible with StableCascadeUNet")

        config = kwargs.pop("config", None)
        resume_download = kwargs.pop("resume_download", False)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        cache_dir = kwargs.pop("cache_dir", None)
        local_files_only = kwargs.pop("local_files_only", None)
        revision = kwargs.pop("revision", None)
        torch_dtype = kwargs.pop("torch_dtype", None)

        checkpoint = load_single_file_model_checkpoint(
            pretrained_model_link_or_path,
            resume_download=resume_download,
            force_download=force_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
        )

        if config is None:
            config = infer_stable_cascade_single_file_config(checkpoint)
            model_config = cls.load_config(**config, **kwargs)
        else:
            model_config = config

        ctx = init_empty_weights if is_accelerate_available() else nullcontext
        with ctx():
            model = cls.from_config(model_config, **kwargs)

        diffusers_format_checkpoint = convert_stable_cascade_unet_single_file_to_diffusers(checkpoint)
        if is_accelerate_available():
            unexpected_keys = load_model_dict_into_meta(model, diffusers_format_checkpoint, dtype=torch_dtype)
            if len(unexpected_keys) > 0:
                logger.warn(
                    f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                )

        else:
            model.load_state_dict(diffusers_format_checkpoint)

        if torch_dtype is not None:
            model.to(torch_dtype)

        return model
