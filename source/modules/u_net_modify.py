from diffusers.loaders.unet import UNet2DConditionLoadersMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    delete_adapter_layers,
    is_accelerate_available,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)

from modules.attention_modify import AttnProcessor,IPAdapterAttnProcessor,AttnProcessor2_0,IPAdapterAttnProcessor2_0
import torch
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.nn as nn
import os
from typing import Callable, Dict, List, Optional, Union
from diffusers.models.embeddings import (
    ImageProjection,
    IPAdapterFullImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta




class UNet2DConditionLoadersMixin_modify(UNet2DConditionLoadersMixin):
    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts):

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
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
                    elif "proj.3.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face
                        num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

                attn_procs[name] = attn_processor_class(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=1.0,
                    num_tokens=num_image_text_embeds,
                ).to(dtype=self.dtype, device=self.device)

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                attn_procs[name].load_state_dict(value_dict)
                key_id += 2

        return attn_procs

    def _load_ip_adapter_weights(self, state_dicts):
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]
        # Set encoder_hid_proj after loading ip_adapter weights,
        # because `IPAdapterPlusImageProjection` also has `attn_processors`.
        self.encoder_hid_proj = None

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts)
        self.set_attn_processor(attn_procs)

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(state_dict["image_proj"])
            image_projection_layer.to(device=self.device, dtype=self.dtype)
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"
