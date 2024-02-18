from diffusers.utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    delete_adapter_layers,
    is_accelerate_available,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)

#from modules.model_diffusers import CrossAttnProcessor
#from modules.ip_adapter import IPAdapterAttnProcessor
import torch
import torch.nn.functional as F
from torch.autograd.function import Function
import torch.nn as nn
from torch import einsum
import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Union
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.embeddings import ImageProjection
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta
import math
from einops import rearrange
from modules.ip_adapter_processor import IPAdapterMaskProcessor

xformers_available = False
try:
    import xformers

    xformers_available = True
except ImportError:
    pass

EPSILON = 1e-6
exists = lambda val: val is not None
default = lambda val, d: val if exists(val) else d
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
def get_attention_scores(attn, query, key, attention_mask=None):

    if attn.upcast_attention:
        query = query.float()
        key = key.float()
    if attention_mask is None:
        baddbmm_input = torch.empty(
            query.shape[0],
            query.shape[1],
            key.shape[1],
            dtype=query.dtype,
            device=query.device,
        )
        beta = 0
    else:
        baddbmm_input = attention_mask
        beta = 1

    attention_scores = torch.baddbmm(
        baddbmm_input,
        query,
        key.transpose(-1, -2),
        beta=beta,
        alpha=attn.scale,
    )

    del baddbmm_input

    if attn.upcast_softmax:
        attention_scores = attention_scores.float()

    return attention_scores.to(query.dtype)


# Get attention_score with this:
def scaled_dot_product_attention_regionstate(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,weight_func =None, region_state = None, sigma = None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype,device = query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias

    batch_size, num_heads, sequence_length, embed_dim = attn_weight.shape
    attn_weight = attn_weight.reshape((-1,sequence_length,embed_dim))
    cross_attention_weight = weight_func(region_state, sigma, attn_weight)
    repeat_time = attn_weight.shape[0]//cross_attention_weight.shape[0]
    attn_weight += torch.repeat_interleave(
        cross_attention_weight, repeats=repeat_time, dim=0
    )
    attn_weight = attn_weight.reshape((-1,num_heads,sequence_length,embed_dim))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value



class AttnProcessor(nn.Module):
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        region_prompt = None,
        ip_adapter_masks = None,
    ):
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        _,img_sequence_length,_ = hidden_states.shape

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)


        is_xattn = False
        if encoder_hidden_states is not None and region_prompt is not None:
            is_xattn = True
            region_state = region_prompt["region_state"]
            weight_func = region_prompt["weight_func"]
            sigma = region_prompt["sigma"]

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        key = attn.to_k(encoder_hidden_states,*args)
        value = attn.to_v(encoder_hidden_states,*args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_xattn and isinstance(region_state, dict):
            # use torch.baddbmm method (slow)
            attention_scores = get_attention_scores(attn, query, key, attention_mask)
            cross_attention_weight = weight_func(region_state[img_sequence_length].to(query.device), sigma, attention_scores)
            attention_scores += torch.repeat_interleave(
                cross_attention_weight, repeats=attention_scores.shape[0] // cross_attention_weight.shape[0], dim=0
            )

            # calc probs
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(query.dtype)
            hidden_states = torch.bmm(attention_probs, value)

        elif xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                attn_bias=attention_mask,
            )
            hidden_states = hidden_states.to(query.dtype)

        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
class IPAdapterAttnProcessor(nn.Module):
    r"""
    Attention processor for Multiple IP-Adapater.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or List[`float`], defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        region_prompt = None,
        ip_adapter_masks = None,
    ):

        _,img_sequence_length,_ = hidden_states.shape
        residual = hidden_states

        is_xattn = False
        if encoder_hidden_states is not None and region_prompt is not None:
            is_xattn = True
            region_state = region_prompt["region_state"]
            weight_func = region_prompt["weight_func"]
            sigma = region_prompt["sigma"]

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`.This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to supress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)


        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if is_xattn and isinstance(region_state, dict):
            # use torch.baddbmm method (slow)
            attention_scores = get_attention_scores(attn, query, key, attention_mask)
            cross_attention_weight = weight_func(region_state[img_sequence_length].to(query.device), sigma, attention_scores)
            attention_scores += torch.repeat_interleave(
                cross_attention_weight, repeats=attention_scores.shape[0] // cross_attention_weight.shape[0], dim=0
            )

            # calc probs
            attention_probs = attention_scores.softmax(dim=-1)
            attention_probs = attention_probs.to(query.dtype)
            hidden_states = torch.bmm(attention_probs, value)

        elif xformers_available:
            hidden_states = xformers.ops.memory_efficient_attention(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                attn_bias=attention_mask,
            )
            hidden_states = hidden_states.to(query.dtype)

        else:
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.batch_to_head_dim(hidden_states)


        '''# for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
        ):
            ip_key = to_k_ip(current_ip_hidden_states)
            ip_value = to_v_ip(current_ip_hidden_states)

            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            if xformers_available:
                current_ip_hidden_states = xformers.ops.memory_efficient_attention(
                    query.contiguous(),
                    ip_key.contiguous(),
                    ip_value.contiguous(),
                    attn_bias=None,
                )
                current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)
            else:
                ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
                current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)
            
            current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)
            hidden_states = hidden_states + scale * current_ip_hidden_states'''

        #control region apply ip-adapter
        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, torch.Tensor) or ip_adapter_masks.ndim != 4:
                raise ValueError(
                    " ip_adapter_mask should be a tensor with shape [num_ip_adapter, 1, height, width]."
                    " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                )
            if len(ip_adapter_masks) != len(self.scale):
                raise ValueError(
                    f"Number of ip_adapter_masks ({len(ip_adapter_masks)}) must match number of IP-Adapters ({len(self.scale)})"
                )
        else:
            ip_adapter_masks = [None] * len(ip_hidden_states)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            ip_key = to_k_ip(current_ip_hidden_states)
            ip_value = to_v_ip(current_ip_hidden_states)
            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)
            ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)

            if mask is not None:
                mask_downsample = IPAdapterMaskProcessor.downsample(
                    mask, batch_size, current_ip_hidden_states.shape[1], current_ip_hidden_states.shape[2]
                )

                mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)

                current_ip_hidden_states = current_ip_hidden_states * mask_downsample

            hidden_states = hidden_states + scale * current_ip_hidden_states



        # linear proj
        hidden_states = attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        region_prompt = None,
        ip_adapter_masks = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        _,img_sequence_length,_ = hidden_states.shape
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        is_xattn = False
        if encoder_hidden_states is not None and region_prompt is not None:
            is_xattn = True
            region_state = region_prompt["region_state"]
            weight_func = region_prompt["weight_func"]
            sigma = region_prompt["sigma"]

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        if is_xattn and isinstance(region_state, dict):
            #w = attn.head_to_batch_dim(w,out_dim = 4).transpose(1, 2)
            hidden_states = scaled_dot_product_attention_regionstate(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,weight_func = weight_func,region_state=region_state[img_sequence_length].to(query.device),sigma = sigma)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAdapterAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or `List[float]`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        region_prompt = None,
        ip_adapter_masks = None,
    ):
        residual = hidden_states

        _,img_sequence_length,_ = hidden_states.shape

        is_xattn = False
        if encoder_hidden_states is not None and region_prompt is not None:
            is_xattn = True
            region_state = region_prompt["region_state"]
            weight_func = region_prompt["weight_func"]
            sigma = region_prompt["sigma"]

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`.This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to supress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)


        

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)


        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1

        if is_xattn and isinstance(region_state, dict):
            #w = attn.head_to_batch_dim(w,out_dim = 4).transpose(1, 2)
            hidden_states = scaled_dot_product_attention_regionstate(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,weight_func = weight_func,region_state=region_state[img_sequence_length].to(query.device),sigma = sigma)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        ''''# for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip
        ):
            ip_key = to_k_ip(current_ip_hidden_states)
            ip_value = to_v_ip(current_ip_hidden_states)

            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            current_ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

            hidden_states = hidden_states + scale * current_ip_hidden_states'''


        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, torch.Tensor) or ip_adapter_masks.ndim != 4:
                raise ValueError(
                    " ip_adapter_mask should be a tensor with shape [num_ip_adapter, 1, height, width]."
                    " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                )
            if len(ip_adapter_masks) != len(self.scale):
                raise ValueError(
                    f"Number of ip_adapter_masks ({len(ip_adapter_masks)}) must match number of IP-Adapters ({len(self.scale)})"
                )
        else:
            ip_adapter_masks = [None] * len(ip_hidden_states)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            ip_key = to_k_ip(current_ip_hidden_states)
            ip_value = to_v_ip(current_ip_hidden_states)
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            current_ip_hidden_states = F.scaled_dot_product_attention(
                query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

            if mask is not None:
                mask_downsample = IPAdapterMaskProcessor.downsample(
                    mask, batch_size, current_ip_hidden_states.shape[1], current_ip_hidden_states.shape[2]
                )

                mask_downsample = mask_downsample.to(query.dtype).to(current_ip_hidden_states.device)

                current_ip_hidden_states = current_ip_hidden_states * mask_downsample

            hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

