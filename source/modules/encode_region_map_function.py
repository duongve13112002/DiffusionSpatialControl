from typing import Any, Callable, Dict, List, Optional, Union
import importlib
import inspect
import math
from pathlib import Path
import re
from collections import defaultdict
import cv2
import time
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.autograd.function import Function
from diffusers import DiffusionPipeline


#Support for find the region of object
def encode_region_map_sp(state,tokenizer,unet,width,height, scale_ratio=8, text_ids=None,do_classifier_free_guidance = True):
        if text_ids is None:
            return torch.FloatTensor(0)
        uncond, cond = text_ids[0], text_ids[1]
        w_tensors = dict()
        cond = cond.reshape(-1,).tolist() if isinstance(cond,np.ndarray) or isinstance(cond, torch.Tensor) else None
        uncond = uncond.reshape(-1,).tolist() if isinstance(uncond,np.ndarray) or isinstance(uncond, torch.Tensor) else None
        for layer in unet.down_blocks:
            c = int(len(cond))
            w_r, h_r = int(math.ceil(width / scale_ratio)), int(math.ceil(height / scale_ratio))

            ret_cond_tensor = torch.zeros((1, int(w_r * h_r), c), dtype=torch.float32)
            ret_uncond_tensor = torch.zeros((1, int(w_r * h_r), c), dtype=torch.float32)
            if state is not None:
                for k, v in state.items():
                    if v["map"] is None:
                        continue
                    is_in = 0

                    k_as_tokens = tokenizer(
                        k,
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        add_special_tokens=False,
                    ).input_ids

                    region_map_resize = np.array(v["map"] < 255 ,dtype = np.uint8)
                    region_map_resize = cv2.resize(region_map_resize,(w_r,h_r),interpolation = cv2.INTER_CUBIC)
                    region_map_resize = (region_map_resize == np.max(region_map_resize)).astype(float)
                    region_map_resize = region_map_resize * float(v["weight"]) 
                    region_map_resize[region_map_resize==0] = -1 * float(v["mask_outsides"]) 
                    ret = torch.from_numpy(
                        region_map_resize
                    )
                    ret = ret.reshape(-1, 1).repeat(1, len(k_as_tokens))

                    if cond is not None:
                        for idx, tok in enumerate(cond):
                            if cond[idx : idx + len(k_as_tokens)] == k_as_tokens:
                                is_in = 1
                                ret_cond_tensor[0, :, idx : idx + len(k_as_tokens)] += ret

                    if uncond is not None:
                        for idx, tok in enumerate(uncond):
                            if uncond[idx : idx + len(k_as_tokens)] == k_as_tokens:
                                is_in = 1
                                ret_uncond_tensor[0, :, idx : idx + len(k_as_tokens)] += ret

                    if not is_in == 1:
                        print(f"tokens {k_as_tokens} not found in text")

            w_tensors[w_r * h_r] = torch.cat([ret_uncond_tensor, ret_cond_tensor]) if do_classifier_free_guidance else ret_cond_tensor
            scale_ratio *= 2

        return w_tensors

def encode_region_map(
    pipe : DiffusionPipeline,
    state,
    width,
    height,
    num_images_per_prompt,
    text_ids = None,
):
    negative_prompt_tokens_id, prompt_tokens_id = text_ids[0] , text_ids[1]
    if prompt_tokens_id is None:
        return  torch.FloatTensor(0)
    prompt_tokens_id = np.array(prompt_tokens_id)
    negative_prompt_tokens_id = np.array(prompt_tokens_id) if negative_prompt_tokens_id is not None else None

    #Spilit to each prompt
    number_prompt = prompt_tokens_id.shape[0]
    prompt_tokens_id = np.split(prompt_tokens_id,number_prompt)
    negative_prompt_tokens_id = np.split(negative_prompt_tokens_id,number_prompt) if negative_prompt_tokens_id is not None else None
    lst_prompt_map = []
    if not isinstance(state,list):
        state = [state]
    if len(state) < number_prompt:
        state = [state] + [None] * int(number_prompt - len(state))
    for i in range(0,number_prompt):
        text_ids = [negative_prompt_tokens_id[i],prompt_tokens_id[i]] if negative_prompt_tokens_id is not None else [None,prompt_tokens_id[i]] 
        region_map = encode_region_map_sp(state[i],pipe.tokenizer,pipe.unet,width,height,scale_ratio = pipe.vae_scale_factor,text_ids = text_ids,do_classifier_free_guidance = pipe.do_classifier_free_guidance)
        lst_prompt_map.append(region_map)

    region_state_sp = {}
    for d in lst_prompt_map:
        for key, tensor in d.items():
            if key in region_state_sp:
                #If key exist, concat
                region_state_sp[key] = torch.cat((region_state_sp[key], tensor))
            else:
                # if key doesnt exist, add
                region_state_sp[key] = tensor

    #add_when_apply num_images_per_prompt
    region_state = {}

    for key, tensor in region_state_sp.items():
        # Repeant accoding to axis = 0 
        region_state[key] = tensor.repeat(num_images_per_prompt,1,1)

    return region_state


