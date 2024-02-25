import transformers
transformers.utils.move_cache()
import random
import tempfile
import time
import gradio as gr
import numpy as np
import torch
import math
import re
import sys
from gradio import inputs
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    DEISMultistepScheduler,
    DDPMScheduler,
    DDIMScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    T2IAdapter,
    SASolverScheduler,
)
from modules.u_net_condition_modify import UNet2DConditionModel
from modules.model_diffusers import (
    StableDiffusionPipeline_finetune,
    StableDiffusionControlNetPipeline_finetune,
    StableDiffusionControlNetImg2ImgPipeline_finetune,
    StableDiffusionImg2ImgPipeline_finetune
)
from modules.attention_modify import AttnProcessor,IPAdapterAttnProcessor,AttnProcessor2_0,IPAdapterAttnProcessor2_0
from modules.model_k_diffusion import StableDiffusionPipeline
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel,CLIPImageProcessor
from PIL import Image,ImageOps
from pathlib import Path
from safetensors.torch import load_file
import modules.safe as _
import os
import cv2
from controlnet_aux import PidiNetDetector, HEDdetector,LineartAnimeDetector,LineartDetector,MLSDdetector,OpenposeDetector,MidasDetector,NormalBaeDetector,ContentShuffleDetector,ZoeDetector
from transformers import pipeline
from modules import samplers_extra_k_diffusion
import gc
import copy
from modules.preprocessing_segmentation import preprocessing_segmentation
import torch.nn.functional as F
from modules.t2i_adapter import setup_model_t2i_adapter
from modules.ip_adapter_processor import IPAdapterMaskProcessor
from typing import Callable, Dict, List, Optional, Union
embeddings_dict = dict()
lora_dict = {'Not using Lora':None,}
lora_lst = ['Not using Lora']
formula = [
    ['w = token_weight_martix * sigma * std(qk)',0],
]

encoding_type ={
    "Automatic111 Encoding": 0,
    "Long Prompt Encoding": 1,
    "Short Prompt Encoding": 2,
}
model_ip_adapter_lst = ['IP-Adapter','IP-Adapter VIT-G','IP-Adapter Light','IP-Adapter Face','IP-Adapter Plus','IP-Adapter Plus Face']

model_ip_adapter_type = {
    "IP-Adapter": "ip-adapter_sd15.bin",
    "IP-Adapter VIT-G": "ip-adapter_sd15_vit-G.bin",
    "IP-Adapter Light": "ip-adapter_sd15_light.bin",
    "IP-Adapter Face":"ip-adapter-full-face_sd15.bin",
    "IP-Adapter Plus": "ip-adapter-plus_sd15.bin",
    "IP-Adapter Plus Face": "ip-adapter-plus-face_sd15.bin",
}

controlnet_lst = ["Canny","Depth","Openpose","Soft Edge","Lineart","Lineart (anime)","Scribble","MLSD","Semantic Segmentation","Normal Map","Shuffle","Instruct Pix2Pix"]
adapter_lst = ["Canny","Sketch","Color","Depth","Openpose","Semantic Segmentation","Zoedepth"]
controlnet_type ={
    "Canny": "lllyasviel/control_v11p_sd15_canny",
    "Depth": "lllyasviel/control_v11f1p_sd15_depth",
    "Openpose": "lllyasviel/control_v11p_sd15_openpose",
    "Soft Edge": "lllyasviel/control_v11p_sd15_softedge",
    "Lineart":"ControlNet-1-1-preview/control_v11p_sd15_lineart",
    "Lineart (anime)":"lllyasviel/control_v11p_sd15s2_lineart_anime",
    "Scribble":"lllyasviel/control_v11p_sd15_scribble",
    "MLSD":"lllyasviel/control_v11p_sd15_mlsd",
    "Semantic Segmentation":"lllyasviel/control_v11p_sd15_seg",
    "Normal Map":"lllyasviel/control_v11p_sd15_normalbae",
    "Shuffle":"lllyasviel/control_v11e_sd15_shuffle",
    "Instruct Pix2Pix":"lllyasviel/control_v11e_sd15_ip2p",
}
adapter_type ={
    "Canny": "TencentARC/t2iadapter_canny_sd15v2",
    "Sketch": "TencentARC/t2iadapter_sketch_sd15v2",
    "Color": "TencentARC/t2iadapter_color_sd14v1",
    "Depth": "TencentARC/t2iadapter_depth_sd15v2",
    "Openpose":"TencentARC/t2iadapter_openpose_sd14v1",
    "Semantic Segmentation":"TencentARC/t2iadapter_seg_sd14v1",
    "Zoedepth":"TencentARC/t2iadapter_zoedepth_sd15v1",
}
models = [
    ("AbyssOrangeMix2", "Korakoe/AbyssOrangeMix2-HF"),
    ("BloodOrangeMix", "WarriorMama777/BloodOrangeMix"),
    ("ElyOrangeMix", "WarriorMama777/ElyOrangeMix"),
    ("Pastal Mix", "JamesFlare/pastel-mix"),
    ("Basil Mix", "nuigurumi/basil_mix"),
    ("Stable Diffusion v1.5", "runwayml/stable-diffusion-v1-5"),
    ("Stable Diffusion v2.1", "stabilityai/stable-diffusion-2-1-base"),
    ("Realistic Vision v1.4", "SG161222/Realistic_Vision_V1.4"),
    ("Dreamlike Photoreal v2.0", "dreamlike-art/dreamlike-photoreal-2.0"),
    ("Waifu-diffusion v1.4", "hakurei/waifu-diffusion"),
    ("Stable diffusion PixelArt v1.4", "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"),
    ("Anything v3", "Linaqruf/anything-v3.0"),
    ("Sketch style", "Cosk/sketchstyle-cutesexyrobutts"),
    ("Anything v5", "stablediffusionapi/anything-v5"),
    ("Counterfeit v2.5", "gsdf/Counterfeit-V2.5"),
    ("Edge of realism", "stablediffusionapi/edge-of-realism"),
    ("Photorealistic fuen", "claudfuen/photorealistic-fuen-v1"),
    ("Protogen x5.8 (Scifi-Anime)", "darkstorm2150/Protogen_x5.8_Official_Release"),
    ("Dreamlike Anime", "dreamlike-art/dreamlike-anime-1.0"),
    ("Something V2.2", "NoCrypt/SomethingV2_2"),
    ("Realistic Vision v3.0", "SG161222/Realistic_Vision_V3.0_VAE"),
    ("Noosphere v3.0", "digiplay/Noosphere_v3"),
    ("Beauty Fool v1.2", "digiplay/BeautyFool_v1.2VAE_pruned"),
    ("Prefix RealisticMix v1.0", "digiplay/PrefixRealisticMix_v1"),
    ("Prefix FantasyMix v1.0", "digiplay/PrefixFantasyMix_v1"),
    ("Unstable Diffusers YamerMIX v3.0", "digiplay/unstableDiffusersYamerMIX_v3"),
    ("GTA5 Artwork Diffusion", "ItsJayQz/GTA5_Artwork_Diffusion"),
    ("Open Journey", "prompthero/openjourney"),
    ("SoapMix2.5D v2.0", "digiplay/SoapMix2.5D_v2"),
    ("CoffeeMix v2.0", "digiplay/CoffeeMix_v2"),
    ("helloworld v3.0", "digiplay/helloworld_v3"),
    ("ARRealVX v1.1", "digiplay/ARRealVX1.1"),
    ("Fishmix v1.0", "digiplay/fishmix_other_v1"),
    ("DiamondCoalMix v2.0", "digiplay/DiamondCoalMix_v2_pruned_diffusers"),
    ("ISOMix v3.22", "digiplay/ISOmix_v3.22"),
    ("Pika v2", "digiplay/Pika_v2"),
    ("BluePencil v0.9b", "digiplay/bluePencil_v09b"),
    ("MeinaPastel v6", "Meina/MeinaPastel_V6"),
    ("Realistic Vision v4", "SG161222/Realistic_Vision_V4.0"),
    ("Revanimated v1.2.2", "stablediffusionapi/revanimated"),
    ("NeverEnding Dream v1.2.2", "Lykon/NeverEnding-Dream"),
    ("CetusMixCoda", "Stax124/CetusMixCoda"),
    ("NewMarsMix R11", "digiplay/NewMarsMix_R11"),
    ("Juggernaut Final", "digiplay/Juggernaut_final"),
    ("BlankCanvas v1.0", "digiplay/BlankCanvas_v1"),
    ("FumizukiMix v1.0", "digiplay/FumizukiMix_v1"),
    ("CampurSari v1.0", "digiplay/CampurSari_Gen1"),
    ("Realisian v1.0", "digiplay/Realisian_v5"),
    ("Real Epic Majic Revolution v1.0", "digiplay/RealEpicMajicRevolution_v1"),
    ("QuinceMix v2.0", "digiplay/quincemix_v2"),
    ("Counterfeit v3.0", "stablediffusionapi/counterfeit-v30"),
    ("MeinaMix v11.0", "Meina/MeinaMix_V11"),
]

keep_vram = ["Korakoe/AbyssOrangeMix2-HF","WarriorMama777/BloodOrangeMix","WarriorMama777/ElyOrangeMix","JamesFlare/pastel-mix","nuigurumi/basil_mix","runwayml/stable-diffusion-v1-5","stabilityai/stable-diffusion-2-1-base","SG161222/Realistic_Vision_V1.4","dreamlike-art/dreamlike-photoreal-2.0","hakurei/waifu-diffusion","Onodofthenorth/SD_PixelArt_SpriteSheet_Generator","Linaqruf/anything-v3.0","Cosk/sketchstyle-cutesexyrobutts","stablediffusionapi/anything-v5","gsdf/Counterfeit-V2.5","stablediffusionapi/edge-of-realism","claudfuen/photorealistic-fuen-v1","darkstorm2150/Protogen_x5.8_Official_Release","dreamlike-art/dreamlike-anime-1.0","NoCrypt/SomethingV2_2","SG161222/Realistic_Vision_V3.0_VAE","digiplay/Noosphere_v3","digiplay/BeautyFool_v1.2VAE_pruned","digiplay/PrefixRealisticMix_v1","digiplay/PrefixFantasyMix_v1","digiplay/unstableDiffusersYamerMIX_v3","ItsJayQz/GTA5_Artwork_Diffusion","prompthero/openjourney","digiplay/SoapMix2.5D_v2","digiplay/CoffeeMix_v2","digiplay/helloworld_v3","digiplay/ARRealVX1.1","digiplay/fishmix_other_v1","digiplay/DiamondCoalMix_v2_pruned_diffusers","digiplay/ISOmix_v3.22","digiplay/Pika_v2","digiplay/bluePencil_v09b","Meina/MeinaPastel_V6","SG161222/Realistic_Vision_V4.0","stablediffusionapi/revanimated","Lykon/NeverEnding-Dream","Stax124/CetusMixCoda","digiplay/NewMarsMix_R11","digiplay/Juggernaut_final","digiplay/BlankCanvas_v1","digiplay/FumizukiMix_v1","digiplay/CampurSari_Gen1","digiplay/Realisian_v5","digiplay/RealEpicMajicRevolution_v1","stablediffusionapi/counterfeit-v30","Meina/MeinaMix_V11"]
base_name, base_model = models[0]

samplers_k_diffusion = [
    ('Euler', 'sample_euler', {}),
    ('Euler a', 'sample_euler_ancestral', {"uses_ensd": True}),
    ('LMS', 'sample_lms', {}),
    ('LCM', samplers_extra_k_diffusion.sample_lcm, {"second_order": True}),
    ('Heun', 'sample_heun', {"second_order": True}),
    ('Heun++', samplers_extra_k_diffusion.sample_heunpp2, {"second_order": True}),
    ('DDPM', samplers_extra_k_diffusion.sample_ddpm, {"second_order": True}),
    ('DPM2', 'sample_dpm_2', {'discard_next_to_last_sigma': True}),
    ('DPM2 a', 'sample_dpm_2_ancestral', {'discard_next_to_last_sigma': True, "uses_ensd": True}),
    ('DPM++ 2S a', 'sample_dpmpp_2s_ancestral', {"uses_ensd": True, "second_order": True}),
    ('DPM++ 2M', 'sample_dpmpp_2m', {}),
    ('DPM++ SDE', 'sample_dpmpp_sde', {"second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE', 'sample_dpmpp_2m_sde', {"brownian_noise": True}),
    ('DPM++ 3M SDE', 'sample_dpmpp_3m_sde', {'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('DPM fast (img-to-img)', 'sample_dpm_fast', {"uses_ensd": True}),
    ('DPM adaptive (img-to-img)', 'sample_dpm_adaptive', {"uses_ensd": True}),
    ('DPM++ 2M SDE Heun', 'sample_dpmpp_2m_sde', {"brownian_noise": True, "solver_type": "heun"}),
    ('Restart', samplers_extra_k_diffusion.restart_sampler, {"second_order": True}),
    ('Euler Karras', 'sample_euler', {'scheduler': 'karras'}),
    ('Euler a Karras', 'sample_euler_ancestral', {'scheduler': 'karras',"uses_ensd": True}),
    ('LMS Karras', 'sample_lms', {'scheduler': 'karras'}),
    ('LCM Karras', samplers_extra_k_diffusion.sample_lcm, {'scheduler': 'karras',"second_order": True}),
    ('Heun Karras', 'sample_heun', {'scheduler': 'karras',"second_order": True}),
    ('Heun++ Karras', samplers_extra_k_diffusion.sample_heunpp2, {'scheduler': 'karras',"second_order": True}),
    ('DDPM Karras', samplers_extra_k_diffusion.sample_ddpm, {'scheduler': 'karras', "second_order": True}),
    ('DPM2 Karras', 'sample_dpm_2', {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM2 a Karras', 'sample_dpm_2_ancestral', {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "uses_ensd": True, "second_order": True}),
    ('DPM++ 2S a Karras', 'sample_dpmpp_2s_ancestral', {'scheduler': 'karras', "uses_ensd": True, "second_order": True}),
    ('DPM++ 2M Karras', 'sample_dpmpp_2m', {'scheduler': 'karras'}),
    ('DPM++ SDE Karras', 'sample_dpmpp_sde', {'scheduler': 'karras', "second_order": True, "brownian_noise": True}),
    ('DPM++ 2M SDE Karras', 'sample_dpmpp_2m_sde', {'scheduler': 'karras', "brownian_noise": True}),
    ('DPM++ 2M SDE Heun Karras', 'sample_dpmpp_2m_sde', {'scheduler': 'karras', "brownian_noise": True, "solver_type": "heun"}),
    ('DPM++ 3M SDE Karras', 'sample_dpmpp_3m_sde', {'scheduler': 'karras', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('Restart Karras', samplers_extra_k_diffusion.restart_sampler, {'scheduler': 'karras', "second_order": True}),
    ('Euler Exponential', 'sample_euler', {'scheduler': 'exponential'}),
    ('Euler a Exponential', 'sample_euler_ancestral', {'scheduler': 'exponential',"uses_ensd": True}),
    ('LMS Exponential', 'sample_lms', {'scheduler': 'exponential'}),
    ('LCM Exponential', samplers_extra_k_diffusion.sample_lcm, {'scheduler': 'exponential',"second_order": True}),
    ('Heun Exponential', 'sample_heun', {'scheduler': 'exponential',"second_order": True}),
    ('Heun++ Exponential', samplers_extra_k_diffusion.sample_heunpp2, {'scheduler': 'exponential',"second_order": True}),
    ('DDPM Exponential', samplers_extra_k_diffusion.sample_ddpm, {'scheduler': 'exponential', "second_order": True}),
    ('DPM++ 2M Exponential', 'sample_dpmpp_2m', {'scheduler': 'exponential'}),
    ('DPM++ 2M SDE Exponential', 'sample_dpmpp_2m_sde', {'scheduler': 'exponential', "brownian_noise": True}),
    ('DPM++ 2M SDE Heun Exponential', 'sample_dpmpp_2m_sde', {'scheduler': 'exponential', "brownian_noise": True, "solver_type": "heun"}),
    ('DPM++ 3M SDE Exponential', 'sample_dpmpp_3m_sde', {'scheduler': 'exponential', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('Restart Exponential', samplers_extra_k_diffusion.restart_sampler, {'scheduler': 'exponential', "second_order": True}),
    ('Euler Polyexponential', 'sample_euler', {'scheduler': 'polyexponential'}),
    ('Euler a Polyexponential', 'sample_euler_ancestral', {'scheduler': 'polyexponential',"uses_ensd": True}),
    ('LMS Polyexponential', 'sample_lms', {'scheduler': 'polyexponential'}),
    ('LCM Polyexponential', samplers_extra_k_diffusion.sample_lcm, {'scheduler': 'polyexponential',"second_order": True}),
    ('Heun Polyexponential', 'sample_heun', {'scheduler': 'polyexponential',"second_order": True}),
    ('Heun++ Polyexponential', samplers_extra_k_diffusion.sample_heunpp2, {'scheduler': 'polyexponential',"second_order": True}),
    ('DDPM Polyexponential', samplers_extra_k_diffusion.sample_ddpm, {'scheduler': 'polyexponential', "second_order": True}), 
    ('DPM++ 2M Polyexponential', 'sample_dpmpp_2m', {'scheduler': 'polyexponential'}),
    ('DPM++ 2M SDE Heun Polyexponential', 'sample_dpmpp_2m_sde', {'scheduler': 'polyexponential', "brownian_noise": True, "solver_type": "heun"}),
    ('DPM++ 3M SDE Polyexponential', 'sample_dpmpp_3m_sde', {'scheduler': 'polyexponential', 'discard_next_to_last_sigma': True, "brownian_noise": True}),
    ('Restart Polyexponential', samplers_extra_k_diffusion.restart_sampler, {'scheduler': 'polyexponential', "second_order": True}),
]


samplers_diffusers = [
    ('Euler a', lambda ddim_scheduler_config: EulerAncestralDiscreteScheduler.from_config(ddim_scheduler_config), {}),
    ('Euler', lambda ddim_scheduler_config: EulerDiscreteScheduler.from_config(ddim_scheduler_config), {}),
    ('LMS', lambda ddim_scheduler_config: LMSDiscreteScheduler.from_config(ddim_scheduler_config), {}),
    ('Heun',lambda ddim_scheduler_config: HeunDiscreteScheduler.from_config(ddim_scheduler_config), {}),
    ('DPM2',lambda ddim_scheduler_config: KDPM2DiscreteScheduler.from_config(ddim_scheduler_config), {}),
    ('DPM2 a',lambda ddim_scheduler_config: KDPM2AncestralDiscreteScheduler.from_config(ddim_scheduler_config), {}),
    ('DPM++ 2S a',lambda ddim_scheduler_config: DPMSolverSinglestepScheduler.from_config(ddim_scheduler_config), {}),
    ('DPM++ 2M',lambda ddim_scheduler_config: DPMSolverMultistepScheduler.from_config(ddim_scheduler_config), {}),
    ('DPM++ SDE',lambda ddim_scheduler_config: DPMSolverSDEScheduler.from_config(ddim_scheduler_config), {}),
    ('DPM++ 2M SDE',lambda ddim_scheduler_config: DPMSolverMultistepScheduler.from_config(ddim_scheduler_config,algorithm_type="sde-dpmsolver++"), {}),
    ('DEIS',lambda ddim_scheduler_config: DEISMultistepScheduler.from_config(ddim_scheduler_config), {}),
    ('UniPC Time Uniform 1',lambda ddim_scheduler_config: UniPCMultistepScheduler.from_config(ddim_scheduler_config,solver_type = "bh1"), {}),
    ('UniPC Time Uniform 2',lambda ddim_scheduler_config: UniPCMultistepScheduler.from_config(ddim_scheduler_config,solver_type = "bh2"), {}),
    ('SA-Solver',lambda ddim_scheduler_config: SASolverScheduler.from_config(ddim_scheduler_config), {}),
    ('Euler Karras', lambda ddim_scheduler_config: EulerDiscreteScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
    ('LMS Karras',lambda ddim_scheduler_config: LMSDiscreteScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),   
    ('Heun Karras',lambda ddim_scheduler_config: HeunDiscreteScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
    ('DPM2 Karras',lambda ddim_scheduler_config: KDPM2DiscreteScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
    ('DPM2 a Karras',lambda ddim_scheduler_config: KDPM2AncestralDiscreteScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
    ('DPM++ 2S a Karras',lambda ddim_scheduler_config: DPMSolverSinglestepScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
    ('DPM++ 2M Karras',lambda ddim_scheduler_config: DPMSolverMultistepScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
    ('DPM++ SDE Karras',lambda ddim_scheduler_config: DPMSolverSDEScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
    ('DPM++ 2M SDE Karras',lambda ddim_scheduler_config: DPMSolverMultistepScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True,algorithm_type="sde-dpmsolver++"), {}),
    ('DEIS Karras',lambda ddim_scheduler_config: DEISMultistepScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
    ('UniPC Time Uniform 1 Karras',lambda ddim_scheduler_config: UniPCMultistepScheduler.from_config(ddim_scheduler_config,solver_type = "bh1",use_karras_sigmas=True), {}),
    ('UniPC Time Uniform 2 Karras',lambda ddim_scheduler_config: UniPCMultistepScheduler.from_config(ddim_scheduler_config,solver_type = "bh2",use_karras_sigmas=True), {}),
    ('SA-Solver Karras',lambda ddim_scheduler_config: SASolverScheduler.from_config(ddim_scheduler_config,use_karras_sigmas=True), {}),
]


start_time = time.time()
timeout = 360

scheduler = DDIMScheduler.from_pretrained(
    base_model,
    subfolder="scheduler",
)
vae = AutoencoderKL.from_pretrained(base_model,
    subfolder="vae",
    torch_dtype=torch.float16
)
if vae is None:
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse", 
        torch_dtype=torch.float16
    )
text_encoder = CLIPTextModel.from_pretrained(
    base_model,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
tokenizer = CLIPTokenizer.from_pretrained(
    base_model,
    subfolder="tokenizer",
    torch_dtype=torch.float16,
)
unet = UNet2DConditionModel.from_pretrained(
    base_model,
    subfolder="unet",
    torch_dtype=torch.float16,
)
feature_extract = CLIPImageProcessor.from_pretrained(
    base_model,
    subfolder="feature_extractor",
)
pipe = StableDiffusionPipeline(
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    vae=vae,
    scheduler=scheduler,
    feature_extractor = feature_extract,
)

if torch.cuda.is_available():
    pipe = pipe.to("cuda")

def get_model_list():
    return models

scheduler_cache ={
    base_model: scheduler
}
te_cache = {
    base_model: text_encoder
}
vae_cache = {
    base_model: vae
}
unet_cache = {
    base_model: unet
}

lora_cache = {
    base_model: LoRANetwork(text_encoder, unet)
}
tokenizer_cache ={
    base_model: tokenizer
}
feature_cache ={
    base_model: feature_extract
}
controlnetmodel_cache ={
    
}
adapter_cache ={
    
}
te_base_weight_length = text_encoder.get_input_embeddings().weight.data.shape[0]
original_prepare_for_tokenization = tokenizer.prepare_for_tokenization
current_model = base_model

def setup_controlnet(name_control,device):
    global controlnet_type,controlnetmodel_cache
    if name_control not in controlnetmodel_cache:
        model_control = ControlNetModel.from_pretrained(name_control, torch_dtype=torch.float16).to(device)
        controlnetmodel_cache[name_control] = model_control
    return controlnetmodel_cache[name_control]

def setup_adapter(adapter_sp,device):
    global model_ip_adapter_type,adapter_cache
    if adapter_sp not in adapter_cache:
        model_control = T2IAdapter.from_pretrained(adapter_sp, torch_dtype=torch.float16).to(device)
        adapter_cache[adapter_sp] = model_control
    return adapter_cache[adapter_sp]



def setup_model(name,clip_skip, lora_state=None, lora_scale=1.0,diffuser_pipeline = False ,control_net_model = None,img_input = None,device = "cpu"):
    global current_model

    keys = [k[0] for k in models]
    model = models[keys.index(name)][1]
    if model not in unet_cache:
        vae_model = AutoencoderKL.from_pretrained(model,subfolder="vae",torch_dtype=torch.float16)
        unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", torch_dtype=torch.float16)
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=torch.float16)
        tokenizer = CLIPTokenizer.from_pretrained(model,subfolder="tokenizer",torch_dtype=torch.float16)
        scheduler = DDIMScheduler.from_pretrained(model,subfolder="scheduler")
        feature_extract = CLIPImageProcessor.from_pretrained(base_model,subfolder="feature_extractor")

        if vae_model is None:
            vae_model = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        scheduler_cache[model] = scheduler
        unet_cache[model] = unet
        te_cache[model] = text_encoder
        vae_cache[model] = vae_model
        tokenizer_cache[model] = tokenizer
        feature_cache[model] = feature_extract
    if current_model != model:
        unet_cache[current_model].to(device)
        te_cache[current_model].to(device)
        vae_cache[current_model].to(device)
        current_model = model

    local_te, local_unet,local_sche,local_vae,local_token,local_feature = copy.deepcopy(te_cache[model]), copy.deepcopy(unet_cache[model]),scheduler_cache[model],vae_cache[model], copy.deepcopy(tokenizer_cache[model]),feature_cache[model]
    if torch.cuda.is_available():
        local_unet.to("cuda")
        local_te.to("cuda")
        local_vae.to("cuda")

    
    if diffuser_pipeline:
        if control_net_model is not None:
            if img_input is not None:
                pipe = StableDiffusionControlNetImg2ImgPipeline_finetune(
                    vae= local_vae,
                    text_encoder= local_te,
                    tokenizer=local_token,
                    unet=local_unet,
                    controlnet = control_net_model,
                    safety_checker= None,
                    scheduler = local_sche,
                    feature_extractor=local_feature,
                    requires_safety_checker = False,
                ).to(device)
            else:
                pipe = StableDiffusionControlNetPipeline_finetune(
                    vae= local_vae,
                    text_encoder= local_te,
                    tokenizer=local_token,
                    unet=local_unet,
                    controlnet = control_net_model,
                    scheduler = local_sche,
                    safety_checker= None,
                    feature_extractor=local_feature,
                    requires_safety_checker = False,
                ).to(device)
        else:
            if img_input is not None:
                pipe = StableDiffusionImg2ImgPipeline_finetune(
                    vae= local_vae,
                    text_encoder= local_te,
                    tokenizer=local_token,
                    unet=local_unet,
                    scheduler = local_sche,
                    safety_checker= None,
                    feature_extractor=local_feature,
                    requires_safety_checker = False,
                ).to(device)
            else:
                pipe = StableDiffusionPipeline_finetune(
                    vae= local_vae,
                    text_encoder= local_te,
                    tokenizer=local_token,
                    unet=local_unet,
                    scheduler = local_sche,
                    safety_checker= None,
                    feature_extractor=local_feature,
                    requires_safety_checker = False,
                ).to(device)
    else:        
        pipe = StableDiffusionPipeline(
            text_encoder=local_te,
            tokenizer=local_token,
            unet=local_unet,
            vae=local_vae,
            scheduler=local_sche,
            feature_extractor=local_feature,
        ).to(device)
        

    if lora_state is not None and lora_state != "":
        pipe = load_lora_control_pipeline(pipe,lora_state,lora_scale,device)
    
    pipe.unet.set_attn_processor(AttnProcessor())
    if hasattr(F, "scaled_dot_product_attention"):
        pipe.unet.set_attn_processor(AttnProcessor2_0())

    if diffuser_pipeline == False:
        pipe.setup_unet(pipe.unet)
        pipe.tokenizer.prepare_for_tokenization = local_token.prepare_for_tokenization
    torch.cuda.empty_cache()
    gc.collect()
    return pipe


def error_str(error, title="Error"):
    return (
        f"""#### {title}
            {error}"""
        if error
        else ""
    )

def make_token_names(embs):
    all_tokens = []
    for name, vec in embs.items():
        tokens = [f'emb-{name}-{i}' for i in range(len(vec))]
        all_tokens.append(tokens)
    return all_tokens

def setup_tokenizer(tokenizer, embs):
    reg_match = [re.compile(fr"(?:^|(?<=\s|,)){k}(?=,|\s|$)") for k in embs.keys()]
    clip_keywords = [' '.join(s) for s in make_token_names(embs)]

    def parse_prompt(prompt: str):
        for m, v in zip(reg_match, clip_keywords):
            prompt = m.sub(v, prompt)
        return prompt

    def prepare_for_tokenization(self, text: str, is_split_into_words: bool = False, **kwargs):
        text = parse_prompt(text)
        r = original_prepare_for_tokenization(text, is_split_into_words, **kwargs)
        return r
        tokenizer.prepare_for_tokenization = prepare_for_tokenization.__get__(tokenizer, CLIPTokenizer)
    return [t for sublist in make_token_names(embs) for t in sublist]


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def load_lora_control_pipeline(pipeline_control,file_path,lora_scale,device):
    state_dict = load_file(file_path,device=device)

    LORA_PREFIX_UNET = 'lora_unet'
    LORA_PREFIX_TEXT_ENCODER = 'lora_te'
    alpha = lora_scale

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
    
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
    
        # as we have set the alpha beforehand, so just skip
        if '.alpha' in key or key in visited:
            continue
        
        if 'text' in key:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_TEXT_ENCODER+'_')[-1].split('_')
            curr_layer = pipeline_control.text_encoder
        else:
            layer_infos = key.split('.')[0].split(LORA_PREFIX_UNET+'_')[-1].split('_')
            curr_layer = pipeline_control.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += '_'+layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)
    
        # org_forward(x) + lora_up(lora_down(x)) * multiplier
        pair_keys = []
        if 'lora_down' in key:
            pair_keys.append(key.replace('lora_down', 'lora_up'))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace('lora_up', 'lora_down'))
    
        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
        
        # update visited list
        for item in pair_keys:
            visited.append(item)
    torch.cuda.empty_cache()
    gc.collect()
    return pipeline_control


def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[...]
    img[invalid_mask] = background_color

    if gamma_corrected:
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def adapter_preprocessing(model_adapter,img_control,low_threshold_adapter = None,high_threshold_adapter=None,has_body=False,has_hand=False,has_face=False,preprocessor_adapter=None,disable_preprocessing_adapter=False):
    if disable_preprocessing_adapter == True :
        return img_control.copy()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if model_adapter == 'Canny':
        img_control = np.array(img_control)
        img_control = cv2.Canny(img_control, low_threshold_adapter, high_threshold_adapter)
        img_control = Image.fromarray(img_control)
    elif model_adapter == 'Openpose':
        processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to(device)
        img_control = processor(img_control, include_body=has_body, include_hand=has_hand, include_face=has_face)
    elif model_adapter == 'Depth':
        if preprocessor_adapter == 'DPT':
            processor = pipeline('depth-estimation')
            img_control = processor(img_control)['depth']
            img_control = np.array(img_control)
            img_control = img_control[:, :, None]
            img_control = np.concatenate([img_control, img_control, img_control], axis=2)
            img_control = Image.fromarray(img_control)
        else:
            processor = MidasDetector.from_pretrained("lllyasviel/Annotators").to(device)
            img_control = processor(img_control)
    elif model_adapter == 'Semantic Segmentation':
        img_control = preprocessing_segmentation(preprocessor_adapter,img_control)
    elif model_adapter == 'Color':
        img_control = img_control.resize((8, 8))
        img_control = img_control.resize((512, 512), resample=Image.Resampling.NEAREST)
    elif model_adapter == 'Zoedepth':
        processor = ZoeDetector.from_pretrained("valhalla/t2iadapter-aux-models", filename="zoed_nk.pth", model_type="zoedepth_nk").to(device)
        img_control = processor(img_control, gamma_corrected=True)
    else:
        active_model = False
        if model_adapter == 'Sketch':
            active_model = True
        if preprocessor_name == 'HED':
            processor = HEDdetector.from_pretrained('lllyasviel/Annotators').to(device)
        else:
            processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators').to(device)
        img_control = processor(img_control,scribble=active_model)
    if model_adapter != 'Canny' and model_adapter != 'Semantic Segmentation' and model_adapter != 'Color':
        del processor
    torch.cuda.empty_cache()
    gc.collect()
    return img_control

def control_net_preprocessing(control_net_model,img_control,low_threshold = None,high_threshold=None,has_body=False,has_hand=False,has_face=False,preprocessor_name=None,disable_preprocessing=False):
    if disable_preprocessing == True or control_net_model == 'Instruct Pix2Pix':
        return img_control.copy()
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if control_net_model == 'Canny':
        img_control = np.array(img_control)
        img_control = cv2.Canny(img_control, low_threshold, high_threshold)
        img_control = img_control[:, :, None]
        img_control = np.concatenate([img_control, img_control, img_control], axis=2)
        img_control = Image.fromarray(img_control)
    elif control_net_model == 'Openpose':
        processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet').to(device)
        img_control = processor(img_control, include_body=has_body, include_hand=has_hand, include_face=has_face)
    elif control_net_model == 'Depth':
        if preprocessor_name == 'DPT':
            processor = pipeline('depth-estimation')
            img_control = processor(img_control)['depth']
            img_control = np.array(img_control)
            img_control = img_control[:, :, None]
            img_control = np.concatenate([img_control, img_control, img_control], axis=2)
            img_control = Image.fromarray(img_control)
        else:
            processor = MidasDetector.from_pretrained("lllyasviel/Annotators").to(device)
            img_control = processor(img_control)      
    elif control_net_model == 'Lineart (anime)':
        processor = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators").to(device)
        img_control = processor(img_control)
    elif control_net_model == 'Lineart':
        processor = LineartDetector.from_pretrained("lllyasviel/Annotators").to(device)
        img_control = processor(img_control)
    elif control_net_model == 'MLSD':
        processor = MLSDdetector.from_pretrained("lllyasviel/ControlNet").to(device)
        img_control = processor(img_control)
        #img_control = np.array(img_control)
    elif control_net_model == 'Semantic Segmentation':
        img_control = preprocessing_segmentation(preprocessor_name,img_control)
    elif control_net_model == 'Normal Map':
        processor = NormalBaeDetector.from_pretrained("lllyasviel/Annotators").to(device)
        img_control = processor(img_control)
    elif control_net_model == 'Shuffle':
        processor = ContentShuffleDetector()
        img_control = processor(img_control)
    else:
        active_model = False
        if control_net_model == 'Scribble':
            active_model = True
        if preprocessor_name == 'HED':
            processor = HEDdetector.from_pretrained('lllyasviel/Annotators').to(device)
        else:
            processor = PidiNetDetector.from_pretrained('lllyasviel/Annotators').to(device)
        img_control = processor(img_control,scribble=active_model)
    if control_net_model != 'Canny' and control_net_model != 'Semantic Segmentation':
        del processor
    torch.cuda.empty_cache()
    gc.collect()
    return img_control

def add_embedding(pipe_model,embs):
    tokenizer, text_encoder = pipe_model.tokenizer, pipe_model.text_encoder
    if embs is not None and len(embs) > 0:
        ti_embs = {}
        for name, file in embs.items():
            if str(file).endswith(".pt"):
                loaded_learned_embeds = torch.load(file, map_location="cpu")
            else:
                loaded_learned_embeds = load_file(file, device="cpu")
            loaded_learned_embeds = loaded_learned_embeds["string_to_param"]["*"] if "string_to_param" in loaded_learned_embeds else loaded_learned_embeds
            if isinstance(loaded_learned_embeds, dict):
                ti_embs.update(loaded_learned_embeds)
            else:
                ti_embs[name] = loaded_learned_embeds

        if len(ti_embs) > 0:
            tokens = setup_tokenizer(tokenizer, ti_embs)
            added_tokens = tokenizer.add_tokens(tokens)
            delta_weight = torch.cat([val for val in ti_embs.values()], dim=0)

            assert added_tokens == delta_weight.shape[0]
            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            token_embeds[-delta_weight.shape[0]:] = delta_weight
    torch.cuda.empty_cache()
    gc.collect()
    return pipe_model

def mask_region_apply_ip_adapter(mask):
    if mask is None:
        return None
    #define black is region masked
    if not isinstance(mask,List):
        mask = [mask]
    mask = [ImageOps.invert(i).convert('RGB') for i in mask]
    processor = IPAdapterMaskProcessor()
    masks = processor.preprocess(mask)
    return masks

lst_control = []
lst_adapter =[]
lst_ip_adapter = []
current_number_ip_adapter = 0
current_number_control = 0
current_number_adapter = 0
def inference(
    prompt,
    guidance,
    steps,
    width=512,
    height=512,
    clip_skip =2,
    seed=0,
    neg_prompt="",
    state=None,
    img_input=None,
    i2i_scale=0.5,
    hr_enabled=False,
    hr_method="Latent",
    hr_scale=1.5,
    hr_denoise=0.8,
    sampler="DPM++ 2M Karras",
    embs=None,
    model=None,
    lora_state=None,
    lora_scale=None,
    formula_setting = None,
    controlnet_enabled = False,
    control_net_model = None,
    low_threshold = None,
    high_threshold = None,
    has_body = False,
    has_hand = False,
    has_face = False,
    img_control = None,
    image_condition = None,
    controlnet_scale = 0,
    preprocessor_name = None,
    diffuser_pipeline = False,
    sampler_hires="DPM++ 2M Karras",
    latent_processing = 0,
    control_guidance_start = 0.0,
    control_guidance_end = 1.0,
    multi_controlnet = False,
    disable_preprocessing = False,
    region_condition = False,
    hr_process_enabled = False,
    ip_adapter = False,
    model_ip_adapter = None,
    inf_adapt_image = None,
    inf_adapt_image_strength = 1.0,
    hr_region_condition = False,
    adapter_enabled = False,
    model_adapter = None,
    low_threshold_adapter = None,
    high_threshold_adapter = None,
    has_body_openpose_adapter = False,
    has_hand_openpose_adapter = False,
    has_face_openpose_adapter = False,
    adapter_img = None,
    image_condition_adapter = None,
    preprocessor_adapter = None,
    adapter_conditioning_scale = 0,
    adapter_conditioning_factor = None,
    multi_adapter = False,
    disable_preprocessing_adapter = False,
    ip_adapter_multi = False,
    guidance_rescale = 0,
    inf_control_adapt_image = None,
    long_encode = 0,
):
    global formula,controlnet_type,lst_control,lst_adapter,model_ip_adapter_type,adapter_type,lst_ip_adapter,current_number_ip_adapter,encoding_type
    img_control_input = None
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if region_condition == False:
        state = None

    if adapter_enabled:
        if len(lst_adapter) > 0 and multi_adapter:
            adapter_img = []
            model_adapter = []
            adapter_conditioning_scale = []
            adapter_conditioning_factor = []
            for i in range( len(lst_adapter)):
                setting_processing = list(lst_adapter[i].items())
                setting_processing = setting_processing[:-2]
                setting_processing = dict(setting_processing)
                image_sp_adapter = adapter_preprocessing(**setting_processing)
                adapter_img.append(image_sp_adapter)
                adapter_sp = adapter_type[lst_adapter[i]["model_adapter"]]
                model_adapter.append(setup_adapter(adapter_sp,device))
                adapter_conditioning_scale.append(float(lst_adapter[i]["adapter_conditioning_scale"]))
                adapter_conditioning_factor.append(float(lst_adapter[i]["adapter_conditioning_factor"]))
            adapter_conditioning_factor = adapter_conditioning_factor[-1]
            torch.cuda.empty_cache()
            gc.collect()
        elif adapter_img is not None and multi_adapter ==False:
            adapter_img = adapter_preprocessing(model_adapter,adapter_img,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter,disable_preprocessing_adapter)
            model_adapter = adapter_type[model_adapter]
            adapter_conditioning_scale = float(adapter_conditioning_scale)
            adapter_conditioning_factor = float(adapter_conditioning_factor)
            torch.cuda.empty_cache()
            gc.collect()
            model_adapter=setup_adapter(model_adapter,device)
            torch.cuda.empty_cache()
            gc.collect()
        else:
            model_adapter = None 
            adapter_img = None
    else:
        model_adapter = None 
        adapter_img = None


    if controlnet_enabled:
        if len(lst_control) > 0 and multi_controlnet:
            img_control = []
            control_net_model = []
            controlnet_scale = []
            control_guidance_start = []
            control_guidance_end = []
            for i in range( len(lst_control)):
                setting_processing = list(lst_control[i].items())
                setting_processing = setting_processing[:-3]
                setting_processing = dict(setting_processing)
                image_sp_control = control_net_preprocessing(**setting_processing)
                img_control.append(image_sp_control)
                conrol_net_sp = controlnet_type[lst_control[i]["control_net_model"]]
                control_net_model.append(setup_controlnet(conrol_net_sp,device))
                controlnet_scale.append(float(lst_control[i]["controlnet_scale"]))
                control_guidance_start.append(float(lst_control[i]["control_guidance_start"]))
                control_guidance_end.append(float(lst_control[i]["control_guidance_end"]))
            torch.cuda.empty_cache()
            gc.collect()
        elif img_control is not None and multi_controlnet ==False:
            img_control = control_net_preprocessing(control_net_model,img_control,low_threshold,high_threshold,has_body,has_hand,has_face,preprocessor_name,disable_preprocessing)
            control_net_model = controlnet_type[control_net_model]
            controlnet_scale = float(controlnet_scale)
            control_guidance_start = float(control_guidance_start)
            control_guidance_end = float(control_guidance_end)
            torch.cuda.empty_cache()
            gc.collect()
            control_net_model=setup_controlnet(control_net_model,device)
            torch.cuda.empty_cache()
            gc.collect()
        else:
            control_net_model = None 
            img_control = None
    else:
        control_net_model = None 
        img_control = None
    keys_f = [k[0] for k in formula]
    formula_setting = formula[keys_f.index(formula_setting)][1]
    if seed is None or seed < 0:
        seed = random.randint(0, sys.maxsize)

    lora_state = lora_dict[lora_state]
    pipe = setup_model(model,clip_skip, lora_state, lora_scale,diffuser_pipeline,control_net_model,img_input,device)
    generator = torch.Generator(device).manual_seed(int(seed))
    
    weight_func = lambda w, sigma, qk: w * sigma * qk.std()

    start_time = time.time()

    sampler_name, sampler_opt = None, None
    pipe = add_embedding(pipe,embs)
    width_resize_mask_ipadapter = width
    height_resize_mask_ipadapter = height
    if img_input is not None:
        width_resize_mask_ipadapter = img_input.width
        height_resize_mask_ipadapter = img_input.height    
    setup_model_t2i_adapter(pipe,model_adapter)
    cross_attention_kwargs = {}

    #Get type encoding
    long_encode = encoding_type[long_encode]

    if ip_adapter == True:
        if ip_adapter_multi and len(lst_ip_adapter) > 0:
            ip_adapter_image_lst =[]
            model_ip_adapter_lst = []
            scale_ip_adapter_lst = []
            region_aplly_lst = []
            for i in lst_ip_adapter:
                ip_adapter_image_lst.append(i["image"])
                model_ip_adapter_lst.append(model_ip_adapter_type[i["model"]])
                scale_ip_adapter_lst.append(float(i["scale"]))
                if i["region_apply"] is not None:
                    region_aplly_lst.append(i["region_apply"])
            if len(region_aplly_lst) == 0:
                region_aplly_lst = None
            inf_adapt_image = ip_adapter_image_lst
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=model_ip_adapter_lst)
            pipe.set_ip_adapter_scale(scale_ip_adapter_lst)
            cross_attention_kwargs = {"ip_adapter_masks":mask_region_apply_ip_adapter(region_aplly_lst)}
        elif inf_adapt_image is not None and ip_adapter_multi == False:
            model_ip_adapter = model_ip_adapter_type[model_ip_adapter]
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=model_ip_adapter)
            pipe.set_ip_adapter_scale(float(inf_adapt_image_strength))
            cross_attention_kwargs = {"ip_adapter_masks":mask_region_apply_ip_adapter(inf_control_adapt_image)}
        else:
            inf_adapt_image = None
    else:
        inf_adapt_image = None
    if diffuser_pipeline:
        for label, funcname, options in samplers_diffusers:
            if label == sampler:
                sampler_name, sampler_opt = funcname, options
            if label == sampler_hires:
                sampler_name_hires, sampler_opt_hires = funcname, options
        pipe.scheduler = sampler_name(pipe.scheduler.config)
        output_type = 'pil'
        if hr_enabled and img_input is None:
            output_type = 'latent'
        config = {
            "prompt": prompt,
            "negative_prompt": neg_prompt,
            "num_inference_steps": int(steps),
            "guidance_scale": guidance,
            "generator": generator,
            "region_map_state": state,
            "latent_processing": latent_processing,
            'weight_func':weight_func,
            'clip_skip' :int(clip_skip),
            "output_type" : output_type,
            "image_t2i_adapter":adapter_img,
            "adapter_conditioning_scale":adapter_conditioning_scale,
            "adapter_conditioning_factor":adapter_conditioning_factor,
            "guidance_rescale":guidance_rescale,
            "long_encode" : int(long_encode),
            "cross_attention_kwargs": cross_attention_kwargs
        }
        
        if control_net_model is not None and img_input is not None:
            result = pipe(controlnet_conditioning_scale = controlnet_scale,inf_adapt_image=inf_adapt_image,image =img_input , control_image=img_control,strength =  i2i_scale,control_guidance_start=control_guidance_start,control_guidance_end=control_guidance_end,**config)
        else:
            if control_net_model is not None:
                result = pipe(width = width,height = height,controlnet_conditioning_scale = controlnet_scale, image=img_control,control_guidance_start=control_guidance_start,control_guidance_end=control_guidance_end,ip_adapter_image=inf_adapt_image,**config)
            elif img_input is not None:
                result = pipe(image =img_input,strength =  i2i_scale,ip_adapter_image=inf_adapt_image,**config)
            else:
                result = pipe(height = height, width = width,ip_adapter_image=inf_adapt_image,**config)            
        if hr_enabled and img_input is None:
            del pipe
            torch.cuda.empty_cache()
            gc.collect()
            pipe = setup_model(model,clip_skip, lora_state, lora_scale,diffuser_pipeline,control_net_model,True,device)
            pipe = add_embedding(pipe,embs)
            pipe.scheduler = sampler_name_hires(pipe.scheduler.config)
            vae_scale_factor = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
            target_height = int(height * upscale_x // vae_scale_factor  )* 8
            target_width = int(width * upscale_x // vae_scale_factor)*8
            latents = result[-1].unsqueeze(0)
            latents = torch.nn.functional.interpolate(
                latents,
                size=(
                    int(target_height // vae_scale_factor),
                    int(target_width // vae_scale_factor),
                ),
                mode=latent_upscale_modes[hr_method]["upscale_method"],
                antialias=latent_upscale_modes[hr_method]["upscale_antialias"],
            )
                
            config = {
                "prompt": prompt,
                "negative_prompt": neg_prompt,
                "num_inference_steps": int(steps),
                "guidance_scale": guidance,
                "generator": generator,
                "region_map_state": state,
                "latent_processing": hr_process_enabled,
                'weight_func':weight_func,
                'clip_skip' :int(clip_skip),
                "image_t2i_adapter":adapter_img,
                "adapter_conditioning_scale":adapter_conditioning_scale,
                "adapter_conditioning_factor":adapter_conditioning_factor,
                "guidance_rescale":guidance_rescale,
                "long_encode" : int(long_encode),
                "cross_attention_kwargs":cross_attention_kwargs,
            }
            if control_net_model is not None:
                upscale_result = pipe(width=int(target_width),height=int(target_height),controlnet_conditioning_scale = controlnet_scale,image = latents, control_image=img_control,strength = hr_denoise,control_guidance_start=control_guidance_start,control_guidance_end=control_guidance_end,**config)
            else:
                upscale_result = pipe(width=int(target_width),height=int(target_height),image = latents,strength = hr_denoise,**config)
            result = result[:-1] + upscale_result
    else:
        for label, funcname, options in samplers_k_diffusion:
            if label == sampler:
                sampler_name, sampler_opt = funcname, options
            if label == sampler_hires:
                sampler_name_hires, sampler_opt_hires = funcname, options
        config = {
            "negative_prompt": neg_prompt,
            "num_inference_steps": int(steps),
            "guidance_scale": guidance,
            "generator": generator,
            "sampler_name": sampler_name,
            "sampler_opt": sampler_opt,
            "region_map_state": state,
            "start_time": start_time,
            "timeout": timeout,
            "latent_processing": latent_processing,
            'weight_func':weight_func,
            'seed': int(seed),
            'sampler_name_hires': sampler_name_hires,
            'sampler_opt_hires': sampler_opt_hires,
            "latent_upscale_processing": hr_process_enabled,
            "ip_adapter_image":inf_adapt_image,
            "controlnet_conditioning_scale":controlnet_scale,
            "control_img": img_control,
            "control_guidance_start":control_guidance_start,
            "control_guidance_end":control_guidance_end,
            "image_t2i_adapter":adapter_img,
            "adapter_conditioning_scale":adapter_conditioning_scale,
            "adapter_conditioning_factor":adapter_conditioning_factor,
            "guidance_rescale":guidance_rescale,
            'clip_skip' :int(clip_skip),
            "long_encode" : int(long_encode),
            "cross_attention_kwargs":cross_attention_kwargs,
        }
        pipe.setup_controlnet(control_net_model)
        if img_input is not None:
            result = pipe.img2img(prompt, image=img_input, strength=i2i_scale,width=img_input.width,height=img_input.height, **config)
        elif hr_enabled:
            result = pipe.txt2img(
                prompt,
                width=width,
                height=height,
                upscale=True,
                upscale_x=hr_scale,
                upscale_denoising_strength=hr_denoise,
                **config,
                **latent_upscale_modes[hr_method],
            )
        else:
            result = pipe.txt2img(prompt, width=width, height=height, **config)
    

    end_time = time.time()
                  
    vram_free, vram_total = torch.cuda.mem_get_info()
    if inf_adapt_image is not None:
        pipe.unload_ip_adapter()
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    print(f"done: model={model}, res={result[-1].width}x{result[-1].height}, step={steps}, time={round(end_time-start_time, 2)}s, vram_alloc={convert_size(vram_total-vram_free)}/{convert_size(vram_total)}")
    return gr.Image.update(result[-1], label=f"Initial Seed: {seed}"),result
    


color_list = []

def get_color(n):
    for _ in range(n - len(color_list)):
        color_list.append(tuple(np.random.random(size=3) * 256))
    return color_list


def create_mixed_img(current, state, w=512, h=512):
    w, h = int(w), int(h)
    image_np = np.full([h, w, 4], 255)
    if state is None:
        state = {}

    colors = get_color(len(state))
    idx = 0

    for key, item in state.items():
        if item["map"] is not None:
            m = item["map"] < 255
            alpha = 150
            if current == key:
                alpha = 200
            image_np[m] = colors[idx] + (alpha,)
        idx += 1

    return image_np

def apply_size_sketch(width,height,state,inf_image):
    if inf_image is not None:
        w_change = inf_image.width
        h_change = inf_image.height 
    else:
        w_change = int(width)
        h_change = int(height)

    if state is not None:
        for key, item in state.items():
            if item["map"] is not None:
                item["map"] = resize(item["map"], w_change, h_change)
    
    update_img = gr.Image.update(value=create_mixed_img("", state, w_change, h_change))
    return state, update_img,gr.Image.update(width=w_change,height = h_change)



def detect_text(text, state, width, height,formula_button,inf_image):
    global formula
    if text is None or text == "":
        return None, None, gr.Radio.update(value=None,visible = False), None,gr.Dropdown.update(value = formula_button)

    if inf_image is not None:
        w_change = inf_image.width
        h_change = inf_image.height 
    else:
        w_change = int(width)
        h_change = int(height)


    t = text.split(",")
    new_state = {}

    for item in t:
        item = item.strip()
        if item == "":
            continue
        if state is not None and item in state:
            new_state[item] = {
                "map": state[item]["map"],
                "weight": state[item]["weight"],
                "mask_outsides": state[item]["mask_outsides"],
            }
        else:
            new_state[item] = {
                "map": None,
                "weight": 0.5,
                "mask_outsides": 0
            }
    update = gr.Radio.update(choices=[key for key in new_state.keys()], value=None,visible = True)
    update_img = gr.update(value=create_mixed_img("", new_state, w_change, h_change))
    update_sketch = gr.update(value=None, interactive=False)
    return new_state, update_sketch, update, update_img,gr.Dropdown.update(value = formula_button)

def detect_text1(text, state, width, height,formula_button,inf_image):
    global formula
    if text is None or text == "":
        return None, None, gr.Radio.update(value=None,visible = False), None,gr.Dropdown.update(value = formula_button)

    if inf_image is not None:
        w_change = inf_image.width
        h_change = inf_image.height 
    else:
        w_change = int(width)
        h_change = int(height)

    t = text.split(",")
    new_state = {}

    for item in t:
        item = item.strip()
        if item == "":
            continue
        if state is not None and item in state:
            new_state[item] = {
                "map": state[item]["map"],
                "weight": state[item]["weight"],
                "mask_outsides": state[item]["mask_outsides"],
            }
        else:
            new_state[item] = {
                "map": None,
                "weight": 0.5,
                "mask_outsides": False
            }
    update = gr.Radio.update(choices=[key for key in new_state.keys()], value=None,visible = True)
    update_img = gr.update(value=create_mixed_img("", new_state, w_change, h_change))
    return new_state, update, update_img,gr.Dropdown.update(value = formula_button)


def resize(img, w, h):
    trs = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((h, w),interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((h, w)),
        ]
    )
    result = np.array(trs(img), dtype=np.uint8)
    return result


def switch_canvas(entry, state, width, height,inf_image):
    if inf_image is not None:
        w_change = inf_image.width
        h_change = inf_image.height 
    else:
        w_change = int(width)
        h_change = int(height)

    if entry is None or state is None:
        return None, 0.5, False, create_mixed_img("", state, w_change, h_change)

    return (
        gr.update(value=None, interactive=True),
        gr.update(value=state[entry]["weight"] if entry in state else 0.5),
        gr.update(value=state[entry]["mask_outsides"] if entry in state else False),
        create_mixed_img(entry, state, w_change, h_change),
    )


def apply_canvas(selected, draw, state, w, h,inf_image):
    if inf_image is not None:
        w_change = inf_image.width
        h_change = inf_image.height 
    else:
        w_change = int(w)
        h_change = int(h)


    if state is not None and selected in state and draw is not None:
        w, h = int(w_change), int(h_change)
        state[selected]["map"] = resize(draw, w, h)
    return state, gr.Image.update(value=create_mixed_img(selected, state, w, h))


def apply_weight(selected, weight, state):
    if state is not None and selected in state:
        state[selected]["weight"] = weight
    return state


def apply_option(selected, mask, state):
    if state is not None and selected in state:
        state[selected]["mask_outsides"] = mask
    return state

clustering_image =[]
number_clustering = 0 
def is_image_black(image):

    average_intensity = image.mean()

    if average_intensity < 10:
        return True
    else:
        return False
def change_diferent_black_to_white(image):

    width, height = image.size

    for x in range(width):
        for y in range(height):
            r, g, b = image.getpixel((x, y))

            if r != 0 and g != 0 and b != 0:
                image.putpixel((x, y), (255, 255, 255))
    return image

def change_black_to_other_color(image,color_list):

    width, height = image.size
    new_pixel = (random.randrange(1,256), random.randrange(1,256), random.randrange(1,256))
    while new_pixel in color_list:
        new_pixel = (random.randrange(1,256), random.randrange(1,256), random.randrange(1,256))
    for x in range(width):
        for y in range(height):
            pixel = image.getpixel((x, y))

            if pixel == (0, 0, 0):
                image.putpixel((x, y), new_pixel)
    return image

def get_color_mask(color, image, threshold=30):
    """
    Returns a color mask for the given color in the given image.
    """
    img_array = np.array(image, dtype=np.uint8)
    color_diff = np.sum((img_array - color) ** 2, axis=-1)
    img_array[color_diff > threshold] = img_array[color_diff > threshold] * 0
    return Image.fromarray(img_array)

def unique_colors(image, threshold=0.01):
    colors = image.getcolors(image.size[0] * image.size[1])
    total_pixels = image.size[0] * image.size[1]
    unique_colors = []
    for count, color in colors:
        if count / total_pixels > threshold:
            unique_colors.append(color)
    return unique_colors

def extract_color_textboxes(color_map_image,MAX_NUM_COLORS):
    color_map_image= Image.fromarray(color_map_image.astype('uint8'), 'RGB')
    # Get unique colors in color_map_image
    colors = unique_colors(color_map_image)
    color_map_image = change_black_to_other_color(color_map_image,colors)
    colors = unique_colors(color_map_image)
    color_masks = [get_color_mask(color, color_map_image) for color in colors]
    # Append white blocks to color_masks to fill up to MAX_NUM_COLORS
    num_missing_masks = MAX_NUM_COLORS - len(color_masks)
    white_mask = Image.new("RGB", color_map_image.size, color=(32, 32, 32))
    color_masks += [white_mask] * num_missing_masks
    color_output =[]
    for i in range(0,len(color_masks)) :
      color_masks[i] = change_diferent_black_to_white(color_masks[i])
      color_masks[i] = np.array(color_masks[i])
      color_masks[i] = cv2.cvtColor(color_masks[i], cv2.COLOR_RGB2GRAY)
      color_masks[i] = 255.0 - color_masks[i]
      if is_image_black(color_masks[i]) == False:
        color_masks[i] = color_masks[i].astype(np.uint8)
        color_output.append(color_masks[i])
    return color_output



def apply_image_clustering(image, selected, w, h, strength, mask, state,inf_image):
    if inf_image is not None:
        w_change = inf_image.width
        h_change = inf_image.height 
    else:
        w_change = int(w)
        h_change = int(h)

    if state is not None and selected in state:
        state[selected] = {
            "map": resize(image, w_change, h_change), 
            "weight": strength, 
            "mask_outsides": mask
        }   
    return state, gr.Image.update(value=create_mixed_img(selected, state, w_change, h_change))


# sp2, radio, width, height, global_stats
def apply_image(image, selected, w, h, strength, mask, state,inf_image):
    if inf_image is not None:
        w_change = inf_image.width
        h_change = inf_image.height 
    else:
        w_change = int(w)
        h_change = int(h)


    if state is not None and selected in state:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        state[selected] = {
            "map": resize(image, w_change, h_change), 
            "weight": strength, 
            "mask_outsides": mask 
        }
    elif state is not None:
        key_state = list(state.keys())
        global number_clustering,clustering_image
        number_clustering = 0
        clustering_image = []
        clustering_image = extract_color_textboxes(image,len(state)+1)
        number_clustering = len(clustering_image)
        if len(state) > len(clustering_image):
            amount_add = len(clustering_image)
        else:
            amount_add = len(state)
        for i in range(0,amount_add):
            state[key_state[i]] = {
                "map": resize(clustering_image[i], w_change, h_change), 
                "weight": strength, 
                "mask_outsides": mask
            }
    return state, gr.Image.update(value=create_mixed_img(selected, state, w_change, h_change))
#rendered, apply_style, apply_clustering_style,Previous,Next,Completed,sp2,sp3
def apply_base_on_color(sp2,state, width, height,inf_image):
    global number_clustering,clustering_image
    if inf_image is not None:
        w_change = inf_image.width
        h_change = inf_image.height 
    else:
        w_change = int(width)
        h_change = int(height)

    number_clustering = 0
    clustering_image = []
    clustering_image = extract_color_textboxes(sp2,len(state)+1)
    new_state = {}
    for i in state:
        new_state[i] = {
            "map": None,
            "weight": 0.5,
            "mask_outsides": False
        }
    return gr.Image.update(value = create_mixed_img("", new_state, w_change, h_change)),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Image.update(visible = False),gr.Image.update(value=clustering_image[0],visible = True),gr.Button.update(visible = True),new_state
def completing_clustering(sp2):
    return gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Image.update(visible = True),gr.Image.update(visible = False),gr.Button.update(visible = False)
def previous_image_page(sp3):
    global clustering_image,number_clustering
    number_clustering = number_clustering - 1
    if number_clustering < 0:
        number_clustering = len(clustering_image)-1
    return gr.Image.update(value = clustering_image[number_clustering])

def next_image_page(sp3):
    global clustering_image,number_clustering
    number_clustering = number_clustering + 1
    if number_clustering >= len(clustering_image):
        number_clustering = 0
    return gr.Image.update(value = clustering_image[number_clustering])
# [ti_state, lora_state, ti_vals, lora_vals, uploads]
def add_net(files):
    if files is None:
        return gr.CheckboxGroup.update(choices=list(embeddings_dict.keys())),gr.Dropdown.update(choices=[k for k in lora_lst],value=lora_lst[0],),gr.File.update(value=None),

    for file in files:
        item = Path(file.name)
        stripedname = str(item.stem).strip()
        stripedname = stripedname.replace('_',' ')
        stripedname = stripedname.replace('-',' ')
        stripedname = stripedname.title()
        if item.suffix == ".pt":
            state_dict = torch.load(file.name, map_location="cpu")
        else:
            state_dict = load_file(file.name, device="cpu")
        if any("lora" in k for k in state_dict.keys()):
            #lora_state = file.name
            if stripedname not in lora_dict:
                lora_lst.append(stripedname)
                lora_dict[stripedname] = file.name 
        else:
            #ti_state[stripedname] = file.name
            if stripedname not in embeddings_dict:
                embeddings_dict[stripedname] = file.name 
    return gr.CheckboxGroup.update(choices=list(embeddings_dict.keys())),gr.Dropdown.update(choices=[k for k in lora_lst],value=lora_lst[0],),gr.File.update(value=None),


# [ti_state, lora_state, ti_vals, lora_vals, uploads]
def clean_states(ti_state):
    global lora_dict
    global embeddings_dict
    global lora_lst
    delete_lora = list(lora_dict.values())[1:]
    for i in delete_lora:
        os.remove(i)
    delete_embed_lst = list(embeddings_dict.values())
    for i in delete_embed_lst:
        os.remove(i)
    embeddings_dict = dict()
    lora_dict = {'Not using Lora':None,}
    lora_lst = ['Not using Lora']
    return dict(),gr.CheckboxGroup.update(choices=list(embeddings_dict.keys()),value = None),gr.Dropdown.update(choices=[k for k in lora_lst],value=lora_lst[0],),gr.File.update(value=None),gr.Text.update(f""),

def add_model(insert_model):
    insert_model=insert_model.replace(" ", "")
    if len(insert_model) == 0:
        return gr.Dropdown.update(choices=[k[0] for k in get_model_list()],value=base_name),gr.Textbox.update(value = '')
    author,name = insert_model.split('/')
    name = name.replace('_',' ')
    name = name.replace('-',' ')
    name = name.title()
    for i in models:
        if name in i or insert_model in i:
            return gr.Dropdown.update(choices=[k[0] for k in get_model_list()],value=base_name),gr.Textbox.update(value = '')
    models.append((name,insert_model))
    keep_vram.append(insert_model)
    return gr.Dropdown.update(choices=[k[0] for k in get_model_list()],value=base_name),gr.Textbox.update(value = '')

def reset_model_button(insert_model):
    return gr.Textbox.update(value = '')

def choose_tistate(ti_vals):
    if len(ti_vals) == 0:
        return dict(),gr.Text.update(""),gr.CheckboxGroup.update(value = None)
    dict_copy = dict()
    for key, value in embeddings_dict.items():
        if key in ti_vals:
            dict_copy[key] = value
    lst_key = [key for key in dict_copy.keys()]
    lst_key = '; '.join(map(str, lst_key))
    return dict_copy,gr.Text.update(lst_key),gr.CheckboxGroup.update(value = None)

def delete_embed(ti_vals,ti_state,embs_choose):
    if len(ti_vals) == 0:
        return gr.CheckboxGroup.update(choices=list(embeddings_dict.keys())),ti_state,gr.Text.update(embs_choose)
    for key in ti_vals:
        if key in ti_state:
            ti_state.pop(key)
        if key in embeddings_dict:
            os.remove(embeddings_dict[key])
            embeddings_dict.pop(key)
    if len(ti_state) >= 1:
        lst_key = [key for key in ti_state.keys()]
        lst_key = '; '.join(map(str, lst_key))
    else:
        lst_key =""
    return gr.CheckboxGroup.update(choices=list(embeddings_dict.keys()),value = None),ti_state,gr.Text.update(lst_key)

def lora_delete(lora_vals):
    global lora_dict
    global lora_lst
    if lora_vals == 'Not using Lora':
        return gr.Dropdown.update(choices=[k for k in lora_lst],value=lora_lst[0],)
    os.remove(lora_dict[lora_vals])
    lora_dict.pop(lora_vals)
    lora_lst.remove(lora_vals)
    return gr.Dropdown.update(choices=[k for k in lora_lst],value=lora_lst[0],)
#diffuser_pipeline,sampler,gallery,hr_enabled
def mode_diffuser_pipeline( controlnet_enabled):
    if controlnet_enabled == True:
        return gr.Checkbox.update(value = True),gr.Checkbox.update()
    return gr.Checkbox.update(value = False),gr.Checkbox.update(value = False)

def res_cap(g, w, h, x):
    if g:
        return f"Enable upscaler: {w}x{h} to {int(w*x)}x{int(h*x)}"
    else:
        return "Enable upscaler"
#diffuser_pipeline,hr_enabled,sampler,gallery,controlnet_enabled
def mode_upscale(diffuser_pipeline, hr_scale, width, height,hr_enabled):
    if hr_enabled == True:
        return gr.Checkbox.update(value = False),gr.Checkbox.update(value = True,label=res_cap(True, width, height, hr_scale)),gr.Dropdown.update(value="DPM++ 2M Karras",choices=[s[0] for s in samplers_k_diffusion]),gr.Checkbox.update(value = False)
    return gr.Checkbox.update(value = False),gr.Checkbox.update(value = False,label=res_cap(False, width, height, hr_scale)),gr.Dropdown.update(value="DPM++ 2M Karras",choices=[s[0] for s in samplers_k_diffusion]),gr.Checkbox.update()

def change_control_net(model_control_net, low_threshold, high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose):
    if model_control_net == 'Canny':
        return gr.Slider.update(visible = True),gr.Slider.update(visible = True),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Radio.update(visible = False)
    if model_control_net == 'Depth':
        return gr.Slider.update(visible = False),gr.Slider.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Radio.update(visible = True,choices=["Midas","DPT"])
    if model_control_net == 'Openpose':
        return gr.Slider.update(visible = False),gr.Slider.update(visible = False),gr.Checkbox.update(visible = True),gr.Checkbox.update(visible = True),gr.Checkbox.update(visible = True),gr.Radio.update(visible = False)
    if model_control_net == 'Semantic Segmentation':
        return gr.Slider.update(visible = False),gr.Slider.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Radio.update(visible = True,choices=["Convnet tiny","Convnet small","Convnet base","Convnet large","Convnet xlarge","Swin tiny","Swin small","Swin base","Swin large"]) 
    if model_control_net =='Soft Edge' or model_control_net == 'Scribble' or model_control_net == 'Sketch':
        return gr.Slider.update(visible = False),gr.Slider.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Radio.update(visible = True,choices=["HED","PidiNet"])
    return gr.Slider.update(visible = False),gr.Slider.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Checkbox.update(visible = False),gr.Radio.update(visible = False)

previous_sampler = 'DPM++ 2M Karras'
previous_sampler_hires = 'DPM++ 2M Karras'
#sampler,gallery,hr_enabled,controlnet_enabled
def mode_diffuser_pipeline_sampler(diffuser_pipeline, sampler,sampler_hires):
    global previous_sampler, previous_sampler_hires
    sample_now = previous_sampler
    sampler_hires_now = previous_sampler_hires
    previous_sampler = sampler
    previous_sampler_hires = sampler_hires
    if diffuser_pipeline == False:
        return gr.Checkbox.update(value = False), gr.Dropdown.update(value=sample_now,choices=[s[0] for s in samplers_k_diffusion]),gr.Dropdown.update(value=sampler_hires_now,choices=[s[0] for s in samplers_k_diffusion])
    return gr.Checkbox.update(value = True),gr.Dropdown.update(value=sample_now,choices=[s[0] for s in samplers_diffusers]),gr.Dropdown.update(value=sampler_hires_now,choices=[s[0] for s in samplers_diffusers])

def change_gallery(latent_processing,hr_process_enabled):
    if latent_processing or hr_process_enabled:
        return gr.Gallery.update(visible = True)
    return gr.Gallery.update(visible = False)


in_edit_mode = False
in_edit_mode_adapter = False
def preview_image(model_control_net,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,img_control,preprocessor_name,multi_controlnet,disable_preprocessing):
    global in_edit_mode
    if multi_controlnet == True and in_edit_mode == True:
        global lst_control,current_number_control
        if model_control_net == lst_control[current_number_control]["control_net_model"]:
            setting_processing = list(lst_control[current_number_control].items())
            setting_processing = setting_processing[:-3]
            setting_processing = dict(setting_processing)
        else:
            setting_processing = {
            "control_net_model": model_control_net,
            "img_control": img_control,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "has_body": has_body_openpose,
            "has_face": has_face_openpose,
            "has_hand": has_hand_openpose,
            "preprocessor_name": preprocessor_name,
            "disable_preprocessing":disable_preprocessing,
        }
        image_sp_control = control_net_preprocessing(**setting_processing)
        return gr.Image.update(image_sp_control)
    elif img_control is not None:
        image_show = control_net_preprocessing(model_control_net,img_control,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,preprocessor_name,disable_preprocessing)
        return gr.Image.update(image_show)
    return gr.Image.update(value = None)



def change_image_condition(image_condition):
    if image_condition is None:
        return gr.Image.update()
    return gr.Image.update(value= None)


#control_net_model,img_control,low_threshold = None,high_threshold=None,has_hand=None,preprocessor_name=None
def control_net_muti(control_net_model,img_control,low_threshold ,high_threshold,has_body,has_hand,has_face,preprocessor_name,controlnet_scale,control_guidance_start,control_guidance_end,disable_preprocessing):
    global lst_control
    if img_control is not None:
        config = {
            "control_net_model": control_net_model,
            "img_control": img_control,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "has_body": has_body,
            "has_face": has_face,
            "has_hand": has_hand,
            "preprocessor_name": preprocessor_name,
            "disable_preprocessing":disable_preprocessing,
            "controlnet_scale": controlnet_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
        }
        lst_control.append(config)
    return gr.Image.update(value = None)

def previous_view_control():
    global lst_control,current_number_control
    if current_number_control <= 0:
        current_number_control = len(lst_control)-1
    else:
        current_number_control -= 1
    return gr.Dropdown.update(value = lst_control[current_number_control]["control_net_model"]),gr.Image.update(value = lst_control[current_number_control]["img_control"]),gr.Slider.update(value = lst_control[current_number_control]["low_threshold"]),gr.Slider.update(value = lst_control[current_number_control]["high_threshold"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_body"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_hand"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_face"]),gr.Radio.update(value = lst_control[current_number_control]["preprocessor_name"]),gr.Slider.update(value= lst_control[current_number_control]["controlnet_scale"]),gr.Slider.update(value= lst_control[current_number_control]["control_guidance_start"]),gr.Slider.update(value= lst_control[current_number_control]["control_guidance_end"]),gr.Checkbox.update(value = lst_control[current_number_control]["disable_preprocessing"])

def next_view_control():
    global lst_control,current_number_control
    if current_number_control >= len(lst_control)-1:
        current_number_control = 0
    else:
        current_number_control += 1
    return gr.Dropdown.update(value = lst_control[current_number_control]["control_net_model"]),gr.Image.update(value = lst_control[current_number_control]["img_control"]),gr.Slider.update(value = lst_control[current_number_control]["low_threshold"]),gr.Slider.update(value = lst_control[current_number_control]["high_threshold"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_body"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_hand"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_face"]),gr.Radio.update(value = lst_control[current_number_control]["preprocessor_name"]),gr.Slider.update(value= lst_control[current_number_control]["controlnet_scale"]),gr.Slider.update(value= lst_control[current_number_control]["control_guidance_start"]),gr.Slider.update(value= lst_control[current_number_control]["control_guidance_end"]),gr.Checkbox.update(value = lst_control[current_number_control]["disable_preprocessing"])

def apply_edit_control_net(control_net_model,img_control,low_threshold ,high_threshold,has_body,has_hand,has_face,preprocessor_name,controlnet_scale,control_guidance_start,control_guidance_end,disable_preprocessing):
    global lst_control,current_number_control,in_edit_mode
    if img_control is not None:
        config = {
            "control_net_model": control_net_model,
            "img_control": img_control,
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "has_body": has_body,
            "has_face": has_face,
            "has_hand": has_hand,
            "preprocessor_name": preprocessor_name,
            "disable_preprocessing":disable_preprocessing,
            "controlnet_scale": controlnet_scale,
            "control_guidance_start": control_guidance_start,
            "control_guidance_end": control_guidance_end,
        }
        lst_control[current_number_control] = config
        return gr.Dropdown.update(),gr.Image.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Radio.update(),gr.Checkbox.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Slider.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update()
    else:
        lst_control.pop(current_number_control)
        current_number_control -=1
        if current_number_control == -1:
            current_number_control = len(lst_control)-1
        if len(lst_control) == 0:
            in_edit_mode = False
            return gr.Dropdown.update(),gr.Image.update(value = None),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Radio.update(),gr.Checkbox.update(value = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Slider.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update()
        return gr.Dropdown.update(value = lst_control[current_number_control]["control_net_model"]),gr.Image.update(value = lst_control[current_number_control]["img_control"]),gr.Slider.update(value = lst_control[current_number_control]["low_threshold"]),gr.Slider.update(value = lst_control[current_number_control]["high_threshold"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_body"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_hand"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_face"]),gr.Radio.update(value = lst_control[current_number_control]["preprocessor_name"]),gr.Checkbox.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Slider.update(value= lst_control[current_number_control]["controlnet_scale"]),gr.Slider.update(value= lst_control[current_number_control]["control_guidance_start"]),gr.Slider.update(value= lst_control[current_number_control]["control_guidance_end"]),gr.Checkbox.update(value = lst_control[current_number_control]["disable_preprocessing"])

def complete_edit_multi():
    global current_number_control,in_edit_mode
    current_number_control = 0
    in_edit_mode = False
    return gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Image.update(value= None),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False)

def multi_controlnet_function(multi_controlnet):
    if multi_controlnet:
        return gr.Checkbox.update(value = True),gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update()
    return gr.Checkbox.update(),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False)

def edit_multi_control_image_function():
    global lst_control,current_number_control,in_edit_mode
    if len(lst_control) > 0:
        in_edit_mode = True
        return gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Dropdown.update(value = lst_control[current_number_control]["control_net_model"]),gr.Image.update(value = lst_control[current_number_control]["img_control"]),gr.Slider.update(value = lst_control[current_number_control]["low_threshold"]),gr.Slider.update(value = lst_control[current_number_control]["high_threshold"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_body"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_hand"]),gr.Checkbox.update(value = lst_control[current_number_control]["has_face"]),gr.Radio.update(value = lst_control[current_number_control]["preprocessor_name"]),gr.Slider.update(value= lst_control[current_number_control]["controlnet_scale"]),gr.Slider.update(value= lst_control[current_number_control]["control_guidance_start"]),gr.Slider.update(value= lst_control[current_number_control]["control_guidance_end"]),gr.Checkbox.update(value = lst_control[current_number_control]["disable_preprocessing"])
    in_edit_mode = False
    return gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Dropdown.update(),gr.Image.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Radio.update(),gr.Slider.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update()

def ip_adapter_work(ip_adapter):
    if ip_adapter:
        return gr.Checkbox.update(value = True)
    return gr.Checkbox.update()


def preview_image_adapter(model_adapter,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,img_control,preprocessor_adapter,multi_adapter,disable_preprocessing_adapter):
    global in_edit_mode_adapter
    if multi_adapter == True and in_edit_mode_adapter == True:
        global lst_adapter,current_number_adapter
        if model_adapter == lst_adapter[current_number_adapter]["model_adapter"]:
            setting_processing = list(lst_adapter[current_number_adapter].items())
            setting_processing = setting_processing[:-3]
            setting_processing = dict(setting_processing)
        else:
            setting_processing = {
            "model_adapter": model_adapter,
            "img_control": img_control,
            "low_threshold_adapter": low_threshold_adapter,
            "high_threshold_adapter": high_threshold_adapter,
            "has_body": has_body_openpose_adapter,
            "has_face": has_face_openpose_adapter,
            "has_hand": has_hand_openpose_adapter,
            "preprocessor_adapter": preprocessor_adapter,
            "disable_preprocessing_adapter":disable_preprocessing_adapter,
        }
        image_sp_control = adapter_preprocessing(**setting_processing)
        return gr.Image.update(image_sp_control)
    elif img_control is not None:
        image_show = adapter_preprocessing(model_adapter,img_control,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter,disable_preprocessing_adapter)
        return gr.Image.update(image_show)
    return gr.Image.update(value = None)



def change_image_condition_adapter(image_condition_adapter):
    if image_condition_adapter is None:
        return gr.Image.update()
    return gr.Image.update(value= None)


#control_net_model,img_control,low_threshold_adapter = None,high_threshold_adapter=None,has_hand=None,preprocessor_adapter=None
def adapter_muti(model_adapter,img_control,low_threshold_adapter ,high_threshold_adapter,has_body,has_hand,has_face,preprocessor_adapter,adapter_conditioning_scale,adapter_conditioning_factor,disable_preprocessing_adapter):
    global lst_adapter
    if img_control is not None:
        config = {
            "model_adapter": model_adapter,
            "img_control": img_control,
            "low_threshold_adapter": low_threshold_adapter,
            "high_threshold_adapter": high_threshold_adapter,
            "has_body": has_body,
            "has_face": has_face,
            "has_hand": has_hand,
            "preprocessor_adapter": preprocessor_adapter,
            "disable_preprocessing_adapter":disable_preprocessing_adapter,
            "adapter_conditioning_scale": adapter_conditioning_scale,
            "adapter_conditioning_factor": adapter_conditioning_factor,
        }
        lst_adapter.append(config)
    return gr.Image.update(value = None)

def previous_view_adapter():
    global lst_adapter,current_number_adapter
    if current_number_adapter <= 0:
        current_number_adapter = len(lst_adapter)-1
    else:
        current_number_adapter -= 1
    return gr.Dropdown.update(value = lst_adapter[current_number_adapter]["model_adapter"]),gr.Image.update(value = lst_adapter[current_number_adapter]["img_control"]),gr.Slider.update(value = lst_adapter[current_number_adapter]["low_threshold_adapter"]),gr.Slider.update(value = lst_adapter[current_number_adapter]["high_threshold_adapter"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_body"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_hand"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_face"]),gr.Radio.update(value = lst_adapter[current_number_adapter]["preprocessor_adapter"]),gr.Slider.update(value= lst_adapter[current_number_adapter]["adapter_conditioning_scale"]),gr.Slider.update(value= lst_adapter[current_number_adapter]["adapter_conditioning_factor"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["disable_preprocessing_adapter"])

def next_view_adapter():
    global lst_adapter,current_number_adapter
    if current_number_adapter >= len(lst_adapter)-1:
        current_number_adapter = 0
    else:
        current_number_adapter += 1
    return gr.Dropdown.update(value = lst_adapter[current_number_adapter]["model_adapter"]),gr.Image.update(value = lst_adapter[current_number_adapter]["img_control"]),gr.Slider.update(value = lst_adapter[current_number_adapter]["low_threshold_adapter"]),gr.Slider.update(value = lst_adapter[current_number_adapter]["high_threshold_adapter"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_body"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_hand"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_face"]),gr.Radio.update(value = lst_adapter[current_number_adapter]["preprocessor_adapter"]),gr.Slider.update(value= lst_adapter[current_number_adapter]["adapter_conditioning_scale"]),gr.Slider.update(value= lst_adapter[current_number_adapter]["adapter_conditioning_factor"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["disable_preprocessing_adapter"])

def apply_edit_adapter(model_adapter,img_control,low_threshold_adapter ,high_threshold_adapter,has_body,has_hand,has_face,preprocessor_adapter,adapter_conditioning_scale,adapter_conditioning_factor,disable_preprocessing_adapter):
    global lst_adapter,current_number_adapter,in_edit_mode_adapter
    if img_control is not None:
        config = {
            "model_adapter": model_adapter,
            "img_control": img_control,
            "low_threshold_adapter": low_threshold_adapter,
            "high_threshold_adapter": high_threshold_adapter,
            "has_body": has_body,
            "has_face": has_face,
            "has_hand": has_hand,
            "preprocessor_adapter": preprocessor_adapter,
            "disable_preprocessing_adapter":disable_preprocessing_adapter,
            "adapter_conditioning_scale": adapter_conditioning_scale,
            "adapter_conditioning_factor": adapter_conditioning_factor,
        }
        lst_adapter[current_number_adapter] = config
        return gr.Dropdown.update(),gr.Image.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Radio.update(),gr.Checkbox.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update()
    else:
        lst_adapter.pop(current_number_adapter)
        current_number_adapter -=1
        if current_number_adapter == -1:
            current_number_adapter = len(lst_adapter)-1
        if len(lst_adapter) == 0:
            in_edit_mode_adapter = False
            return gr.Dropdown.update(),gr.Image.update(value = None),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Radio.update(),gr.Checkbox.update(value = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update()
        return gr.Dropdown.update(value = lst_adapter[current_number_adapter]["model_adapter"]),gr.Image.update(value = lst_adapter[current_number_adapter]["img_control"]),gr.Slider.update(value = lst_adapter[current_number_adapter]["low_threshold_adapter"]),gr.Slider.update(value = lst_adapter[current_number_adapter]["high_threshold_adapter"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_body"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_hand"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_face"]),gr.Radio.update(value = lst_adapter[current_number_adapter]["preprocessor_adapter"]),gr.Checkbox.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Slider.update(value= lst_adapter[current_number_adapter]["adapter_conditioning_scale"]),gr.Slider.update(value= lst_adapter[current_number_adapter]["adapter_conditioning_factor"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["disable_preprocessing_adapter"])

def complete_edit_multi_adapter():
    global current_number_adapter,in_edit_mode_adapter
    current_number_adapter = 0
    in_edit_mode_adapter = False
    return gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Image.update(value= None),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False)

def multi_adapter_function(multi_adapter):
    if multi_adapter:
        return gr.Checkbox.update(value = True),gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update()
    return gr.Checkbox.update(),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False)

def edit_multi_adapter_image_function():
    global lst_adapter,current_number_adapter,in_edit_mode_adapter
    if len(lst_adapter) > 0:
        in_edit_mode_adapter = True
        return gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = False),gr.Button.update(visible = False),gr.Dropdown.update(value = lst_adapter[current_number_adapter]["model_adapter"]),gr.Image.update(value = lst_adapter[current_number_adapter]["img_control"]),gr.Slider.update(value = lst_adapter[current_number_adapter]["low_threshold_adapter"]),gr.Slider.update(value = lst_adapter[current_number_adapter]["high_threshold_adapter"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_body"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_hand"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["has_face"]),gr.Radio.update(value = lst_adapter[current_number_adapter]["preprocessor_adapter"]),gr.Slider.update(value= lst_adapter[current_number_adapter]["adapter_conditioning_scale"]),gr.Slider.update(value= lst_adapter[current_number_adapter]["adapter_conditioning_factor"]),gr.Checkbox.update(value = lst_adapter[current_number_adapter]["disable_preprocessing_adapter"])
    in_edit_mode_adapter = False
    return gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Button.update(),gr.Dropdown.update(),gr.Image.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Checkbox.update(),gr.Radio.update(),gr.Slider.update(),gr.Slider.update(),gr.Checkbox.update()


def ip_adpater_function(ip_adapter):
    if ip_adapter:
        return gr.Checkbox.update()
    return gr.Checkbox.update(value = False)

#ip_adapter,inf_adapt_image,inf_adapt_image_multi,inf_adapt_image_strength,inf_adapt_image_strength_multi,edit_ip_adapter_setting,apply_ip_adapter_setting
def ip_adpater_multi_function(ip_adapter_multi):
    if ip_adapter_multi:
        return gr.Checkbox.update(value = True), gr.Image.update(visible = False), gr.Image.update(visible = True), gr.Slider.update(visible = False), gr.Slider.update(visible = True),gr.Button.update(visible = True),gr.Button.update(visible = True), gr.Image.update(visible = False), gr.Image.update(visible = True)
    return gr.Checkbox.update(), gr.Image.update(visible = True), gr.Image.update(visible = False), gr.Slider.update(visible = True), gr.Slider.update(visible = False),gr.Button.update(visible = False),gr.Button.update(visible = False), gr.Image.update(visible = True), gr.Image.update(visible = False)

def apply_ip_adapter_setting_function(model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,inf_control_adapt_image_multi):
    global lst_ip_adapter,current_number_ip_adapter
    if inf_adapt_image_multi is not None:
        config ={
        "model" : model_ip_adapter,
        "image" : inf_adapt_image_multi,
        "region_apply": inf_control_adapt_image_multi,
        "scale" : float(inf_adapt_image_strength_multi),
        }
        lst_ip_adapter.append(config)
        return gr.Image.update(value = None),gr.Image.update(value = None)
    return gr.Image.update(value = None),gr.Image.update(value = None)

#model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,previous_ip_adapter_setting,next_ip_adapter_setting,apply_edit_ip_adapter_setting,complete_cip_adapter_setting,edit_ip_adapter_setting,apply_ip_adapter_setting
def edit_ip_adapter_setting_function():
    global lst_ip_adapter,current_number_ip_adapter
    if len(lst_ip_adapter) == 0:
        return (
            gr.Dropdown.update(),
            gr.Image.update(),
            gr.Slider.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Image.update(),
        )
    return (
        gr.Dropdown.update(value = lst_ip_adapter[current_number_ip_adapter]["model"]),
        gr.Image.update(value = lst_ip_adapter[current_number_ip_adapter]["image"]),
        gr.Slider.update(value = lst_ip_adapter[current_number_ip_adapter]["scale"]),
        gr.Button.update(visible = True),
        gr.Button.update(visible = True),
        gr.Button.update(visible = True),
        gr.Button.update(visible = True),
        gr.Button.update(visible = False),
        gr.Button.update(visible = False),
        gr.Image.update(value = lst_ip_adapter[current_number_ip_adapter]["region_apply"]),
    )

def previous_ip_adapter_setting_function():
    global lst_ip_adapter,current_number_ip_adapter
    current_number_ip_adapter -= 1
    if current_number_ip_adapter < 0:
        current_number_ip_adapter = len(lst_ip_adapter) -1
    return (
        gr.Dropdown.update(value = lst_ip_adapter[current_number_ip_adapter]["model"]),
        gr.Image.update(value = lst_ip_adapter[current_number_ip_adapter]["image"]),
        gr.Slider.update(value = lst_ip_adapter[current_number_ip_adapter]["scale"]),
        gr.Image.update(value = lst_ip_adapter[current_number_ip_adapter]["region_apply"]),
        )

def next_ip_adapter_setting_function():
    global lst_ip_adapter,current_number_ip_adapter
    current_number_ip_adapter += 1
    if current_number_ip_adapter == len(lst_ip_adapter):
        current_number_ip_adapter = 0
    return (
        gr.Dropdown.update(value = lst_ip_adapter[current_number_ip_adapter]["model"]),
        gr.Image.update(value = lst_ip_adapter[current_number_ip_adapter]["image"]),
        gr.Slider.update(value = lst_ip_adapter[current_number_ip_adapter]["scale"]),
        gr.Image.update(value = lst_ip_adapter[current_number_ip_adapter]["region_apply"]),
    )

#inf_adapt_image_multi,previous_ip_adapter_setting,next_ip_adapter_setting,edit_ip_adapter_setting,apply_ip_adapter_setting,apply_edit_ip_adapter_setting,complete_cip_adapter_setting
def complete_cip_adapter_setting_function():
    return (
        gr.Image.update(value = None),
        gr.Button.update(visible = False),
        gr.Button.update(visible = False),
        gr.Button.update(visible = True),
        gr.Button.update(visible = True),
        gr.Button.update(visible = False),
        gr.Button.update(visible = False),
        gr.Image.update(value = None),
    )


#model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,previous_ip_adapter_setting,next_ip_adapter_setting,edit_ip_adapter_setting,apply_ip_adapter_setting,apply_edit_ip_adapter_setting,complete_cip_adapter_setting
def apply_edit_ip_adapter_setting_function(model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,inf_control_adapt_image_multi):
    global lst_ip_adapter,current_number_ip_adapter
    if inf_adapt_image_multi is not None:
        config_change = lst_ip_adapter[current_number_ip_adapter]
        config_change["model"] = model_ip_adapter
        config_change["image"] = inf_adapt_image_multi
        config_change["scale"] = float(inf_adapt_image_strength_multi)
        config_change["region_apply"] = inf_control_adapt_image_multi
        return (
            gr.Dropdown.update(),
            gr.Image.update(),
            gr.Slider.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Button.update(),
            gr.Image.update(),
            )
    #Delete 
    lst_ip_adapter.pop(current_number_ip_adapter)
    current_number_ip_adapter -= 1
    if len(lst_ip_adapter) == 0:
        return (
            gr.Dropdown.update(),
            gr.Image.update(value = None),
            gr.Slider.update(),
            gr.Button.update(visible = False),
            gr.Button.update(visible = False),
            gr.Button.update(visible = True),
            gr.Button.update(visible = True),
            gr.Button.update(visible = False),
            gr.Button.update(visible = False),
            gr.Image.update(value = None),
        )
    if current_number_ip_adapter == -1:
        current_number_ip_adapter = len(lst_ip_adapter)-1
    return (
        gr.Dropdown.update(value = lst_ip_adapter[current_number_ip_adapter]["model"]),
        gr.Image.update(value = lst_ip_adapter[current_number_ip_adapter]["image"]),
        gr.Slider.update(value = lst_ip_adapter[current_number_ip_adapter]["scale"]),
        gr.Button.update(),
        gr.Button.update(),
        gr.Button.update(),
        gr.Button.update(),
        gr.Button.update(),
        gr.Button.update(),
        gr.Image.update(value = lst_ip_adapter[current_number_ip_adapter]["region_apply"]),
    )




latent_upscale_modes = {
    "Latent (bilinear)": {"upscale_method": "bilinear", "upscale_antialias": False},
    "Latent (bilinear antialiased)": {"upscale_method": "bilinear", "upscale_antialias": True},
    "Latent (bicubic)": {"upscale_method": "bicubic", "upscale_antialias": False},
    "Latent (bicubic antialiased)": {
        "upscale_method": "bicubic",
        "upscale_antialias": True,
    },
    "Latent (nearest)": {"upscale_method": "nearest", "upscale_antialias": False},
    "Latent (nearest-exact)": {
        "upscale_method": "nearest-exact",
        "upscale_antialias": False,
    },
    "Latent (area)": {"upscale_method": "area", "upscale_antialias": False},
}

css = """
.finetuned-diffusion-div div{
    display:inline-flex;
    align-items:center;
    gap:.8rem;
    font-size:1.75rem;
    padding-top:2rem;
}
.finetuned-diffusion-div div h1{
    font-weight:900;
    margin-bottom:7px
}
.finetuned-diffusion-div p{
    margin-bottom:10px;
    font-size:94%
}
.box {
  float: left;
  height: 20px;
  width: 20px;
  margin-bottom: 15px;
  border: 1px solid black;
  clear: both;
}
a{
    text-decoration:underline
}
.tabs{
    margin-top:0;
    margin-bottom:0
}
#gallery{
    min-height:20rem
}
.no-border {
    border: none !important;
}
 """
with gr.Blocks(css=css) as demo:
    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Demo for diffusion models</h1>
              </div>
              <p>Running on CPU 🥶 This demo does not work on CPU.</p>
            </div>
        """
    )
    global_stats = gr.State(value={})

    with gr.Row():

        with gr.Column(scale=55):
            model = gr.Dropdown(
                choices=[k[0] for k in get_model_list()],
                label="Model",
                value=base_name,
            )
            with gr.Row():
                image_out = gr.Image()
                gallery = gr.Gallery(label="Generated images", show_label=True, elem_id="gallery",visible = False).style(grid=[1], height="auto")

        with gr.Column(scale=45):

            with gr.Group():

                with gr.Row():
                    with gr.Column(scale=70):

                        prompt = gr.Textbox(
                            label="Prompt",
                            value="A lovely girl sitting on a bridge",
                            show_label=True,
                            placeholder="Enter prompt.",
                        )
                        neg_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="bad quality",
                            show_label=True,
                            placeholder="Enter negative prompt.",
                        )

                    generate = gr.Button(value="Generate").style(
                        rounded=(False, True, True, False)
                    )

            with gr.Tab("Options"):

                with gr.Group():

                    with gr.Row():
                        diffuser_pipeline = gr.Checkbox(label="Using diffusers pipeline", value=False)
                        latent_processing = gr.Checkbox(label="Show processing", value=False)
                        region_condition = gr.Checkbox(label="Enable region condition", value=False)
                    with gr.Row():
                        guidance = gr.Slider(
                            label="Guidance scale", value=7.5, maximum=20
                        )
                        guidance_rescale = gr.Slider(
                            label="Guidance rescale", value=0, maximum=20
                        )
                    with gr.Row():
                        width = gr.Slider(
                            label="Width", value=512, minimum=64, maximum=1920, step=8
                        )
                        height = gr.Slider(
                            label="Height", value=512, minimum=64, maximum=1920, step=8
                        )
                    with gr.Row():
                        clip_skip = gr.Slider(
                            label="Clip Skip", value=2, minimum=1, maximum=12, step=1
                        )
                        steps = gr.Slider(
                            label="Steps", value=25, minimum=2, maximum=100, step=1
                        )
                    with gr.Row():   
                        long_encode = sampler = gr.Dropdown(
                            value="Automatic111 Encoding",
                            label="Encoding prompt type",
                            choices=[s for s in encoding_type],
                        )
                        sampler = gr.Dropdown(
                            value="DPM++ 2M Karras",
                            label="Sampler",
                            choices=[s[0] for s in samplers_k_diffusion],
                        )
                    with gr.Row():
                        seed = gr.Number(label="Seed (Lower than 0 = random)", value=-1)
                        Insert_model = gr.Textbox(
                            label="Insert model",
                            show_label=True,
                            placeholder="Enter a hugging face's link model.",
                        )
                        insert_model = gr.Button(value="Insert")
                    
                    insert_model.click(
                        add_model,
                        inputs=[Insert_model],
                        outputs=[model, Insert_model],
                        queue=False,
                    )
                    

            with gr.Tab("Image to image"):
                with gr.Group():
                    inf_image = gr.Image(
                        label="Image", source="upload", type="pil"
                    )
                    inf_strength = gr.Slider(
                        label="Transformation strength",
                        minimum=0,
                        maximum=1,
                        step=0.01,
                        value=0.5,
                    )
            with gr.Tab("Hires fix"):
                with gr.Group():
                    with gr.Row():
                        hr_enabled = gr.Checkbox(label="Enable upscaler", value=False)
                        hr_process_enabled = gr.Checkbox(label="Show processing upscaler", value=False)
                        hr_region_condition = gr.Checkbox(label="Enable region condition upscaler", value=False)
                    with gr.Row():
                        hr_method = gr.Dropdown(
                            [key for key in latent_upscale_modes.keys()],
                            value="Latent (bilinear)",
                            label="Upscale method",
                        )
                        sampler_hires = gr.Dropdown(
                            value="DPM++ 2M Karras",
                            label="Sampler",
                            choices=[s[0] for s in samplers_k_diffusion],
                        )
                    
                    hr_scale = gr.Slider(
                        label="Upscale factor",
                        minimum=1.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.2,
                    )
                    hr_denoise = gr.Slider(
                        label="Denoising strength",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.8,
                    )

                    hr_scale.change(
                        lambda g, x, w, h: gr.Checkbox.update(
                            label=res_cap(g, w, h, x)
                        ),
                        inputs=[hr_enabled, hr_scale, width, height],
                        outputs=hr_enabled,
                        queue=False,
                    )
                    hr_process_enabled.change(
                        change_gallery,
                        inputs=[latent_processing,hr_process_enabled],
                        outputs=[gallery],
                        queue=False,
                    )
            latent_processing.change(
                change_gallery,
                inputs=[latent_processing,hr_process_enabled],
                outputs=[gallery],
                queue=False,
            )
            with gr.Tab("IP-Adapter"):
                with gr.Group():
                    with gr.Row():
                        ip_adapter = gr.Checkbox(label="Using IP-Adapter", value=False)
                        ip_adapter_multi = gr.Checkbox(label="Using Multi IP-Adapter", value=False)
                    model_ip_adapter = gr.Dropdown(
                        choices=[k for k in model_ip_adapter_lst],
                        label="Model IP-Adapter",
                        value=model_ip_adapter_lst[0],
                    )

                    with gr.Row():
                        inf_adapt_image = gr.Image(
                            label="IP-Adapter", source="upload", type="pil"
                        )
                        inf_control_adapt_image = gr.Image(
                            label="Region apply", source="upload", type="pil",image_mode='L'
                        )
                        inf_adapt_image_multi = gr.Image(
                            label="IP-Adapter", source="upload", type="pil",visible= False
                        )
                        inf_control_adapt_image_multi = gr.Image(
                            label="Region apply(Black masks are the regions you want to apply)", source="upload", type="pil",image_mode='L',visible= False
                        )
                    inf_adapt_image_strength = gr.Slider(
                        label="IP-Adapter scale",
                        minimum=0,
                        maximum=2,
                        step=0.01,
                        value=1,
                    )
                    inf_adapt_image_strength_multi = gr.Slider(
                        label="IP-Adapter scale",
                        minimum=0,
                        maximum=2,
                        step=0.01,
                        value=1,
                        visible= False,
                    )
                    with gr.Row():
                        previous_ip_adapter_setting = gr.Button(value="Previous setting",visible = False)
                        next_ip_adapter_setting = gr.Button(value="Next setting",visible = False)
                    with gr.Row():
                        edit_ip_adapter_setting = gr.Button(value="Edit previous setting",visible = False)
                        apply_ip_adapter_setting = gr.Button(value="Apply setting",visible = False)
                    with gr.Row():
                        apply_edit_ip_adapter_setting = gr.Button(value="Apply change",visible = False)
                        complete_cip_adapter_setting = gr.Button(value="Complete change",visible = False) 
                    ip_adapter.change(
                        ip_adpater_function,
                        inputs=[ip_adapter],
                        outputs=[ip_adapter_multi],
                        queue=False,
                    )
                    ip_adapter_multi.change(
                        ip_adpater_multi_function,
                        inputs=[ip_adapter_multi],
                        outputs=[ip_adapter,inf_adapt_image,inf_adapt_image_multi,inf_adapt_image_strength,inf_adapt_image_strength_multi,edit_ip_adapter_setting,apply_ip_adapter_setting,inf_control_adapt_image,inf_control_adapt_image_multi],
                        queue=False,
                    )
                    apply_ip_adapter_setting.click(
                        apply_ip_adapter_setting_function,
                        inputs = [model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,inf_control_adapt_image_multi],
                        outputs = [inf_adapt_image_multi,inf_control_adapt_image_multi],
                    )
                    edit_ip_adapter_setting.click(
                        edit_ip_adapter_setting_function,
                        inputs = [],
                        outputs =[model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,previous_ip_adapter_setting,next_ip_adapter_setting,apply_edit_ip_adapter_setting,complete_cip_adapter_setting,edit_ip_adapter_setting,apply_ip_adapter_setting,inf_control_adapt_image_multi],
                        queue =False,
                    )
                    previous_ip_adapter_setting.click(
                        previous_ip_adapter_setting_function,
                        inputs = [],
                        outputs = [model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,inf_control_adapt_image_multi],
                        queue = False,
                    )
                    next_ip_adapter_setting.click(
                        next_ip_adapter_setting_function,
                        inputs = [],
                        outputs = [model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,inf_control_adapt_image_multi],
                        queue = False,
                    )
                    apply_edit_ip_adapter_setting.click(
                        apply_edit_ip_adapter_setting_function,
                        inputs = [model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,inf_control_adapt_image_multi],
                        outputs =[model_ip_adapter,inf_adapt_image_multi,inf_adapt_image_strength_multi,previous_ip_adapter_setting,next_ip_adapter_setting,edit_ip_adapter_setting,apply_ip_adapter_setting,apply_edit_ip_adapter_setting,complete_cip_adapter_setting,inf_control_adapt_image_multi],
                        queue = False,
                    )
                    complete_cip_adapter_setting.click(
                        complete_cip_adapter_setting_function,
                        inputs = [],
                        outputs = [inf_adapt_image_multi,previous_ip_adapter_setting,next_ip_adapter_setting,edit_ip_adapter_setting,apply_ip_adapter_setting,apply_edit_ip_adapter_setting,complete_cip_adapter_setting,inf_control_adapt_image_multi],
                        queue = False,
                    )                 
            with gr.Tab("Controlnet"):
                with gr.Group():
                    with gr.Row(): 
                        controlnet_enabled = gr.Checkbox(label="Enable Controlnet", value=False)
                        disable_preprocessing = gr.Checkbox(label="Disable preprocessing", value=False)
                        multi_controlnet = gr.Checkbox(label="Enable Multi Controlnet", value=False)
                    model_control_net = gr.Dropdown(
                        choices=[k for k in controlnet_lst],
                        label="Model Controlnet",
                        value=controlnet_lst[0],
                    )
                    with gr.Row():
                        low_threshold = gr.Slider(
                            label="Canny low threshold", value=100, minimum=1, maximum=255, step=1
                        )
                        high_threshold = gr.Slider(
                            label="Canny high threshold", value=200, minimum=1, maximum=255, step=1
                        )
                    with gr.Row():
                        has_body_openpose = gr.Checkbox(label="Has body", value=True,visible= False)
                        has_hand_openpose = gr.Checkbox(label="Has hand", value=False,visible= False)
                        has_face_openpose = gr.Checkbox(label="Has face", value=False,visible= False)
                    preprocessor_name = gr.Radio(
                        label="Preprocessor",
                        type="value",
                        visible= False,
                    )
                    with gr.Row():
                        control_guidance_start = gr.Slider(
                            label="Control guidance start", value=0, minimum=0, maximum=1, step=0.01
                        )
                        control_guidance_end = gr.Slider(
                            label="Control guidance end", value=1, minimum=0, maximum=1, step=0.01
                        )
                    controlnet_scale = gr.Slider(
                            label="Controlnet scale", value=1, minimum=0, maximum=2, step=0.01
                        )
                    with gr.Row():
                        controlnet_img = gr.Image(
                            image_mode="RGB",
                            source="upload",
                            label = "Image",
                            type = 'pil',
                        )
                        image_condition = gr.Image(interactive=False,image_mode="RGB",label = "Preprocessor Preview",type = 'pil')
                    control_image_click = gr.Button(value="Preview")
                    with gr.Row():
                        previous_multi_control_image = gr.Button(value="Previous control setting",visible = False)
                        next_multi_control_image = gr.Button(value="Next control setting",visible = False)
                    with gr.Row():
                        edit_multi_control_image = gr.Button(value="Edit previous setting",visible = False)
                        apply_multi_control_image = gr.Button(value="Apply setting",visible = False)
                    with gr.Row():
                        apply_edit_multi = gr.Button(value="Apply change",visible = False)
                        complete_change_multi = gr.Button(value="Complete change",visible = False)

                    control_image_click.click(
                        preview_image,
                        inputs=[model_control_net,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,controlnet_img,preprocessor_name,multi_controlnet,disable_preprocessing],
                        outputs=[image_condition],
                        queue=False,
                    )
                    multi_controlnet.change(
                        multi_controlnet_function,
                        inputs=[multi_controlnet],
                        outputs=[controlnet_enabled,edit_multi_control_image,apply_multi_control_image,previous_multi_control_image,next_multi_control_image,apply_edit_multi,complete_change_multi],
                        queue=False,
                    )
                    edit_multi_control_image.click(
                        edit_multi_control_image_function,
                        inputs=[],
                        outputs=[previous_multi_control_image,next_multi_control_image,apply_edit_multi,complete_change_multi,edit_multi_control_image,apply_multi_control_image,model_control_net,controlnet_img,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,preprocessor_name,controlnet_scale,control_guidance_start,control_guidance_end,disable_preprocessing],
                        queue=False,
                    )

                    previous_multi_control_image.click(
                        previous_view_control,
                        inputs=[],
                        outputs=[model_control_net,controlnet_img,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,preprocessor_name,controlnet_scale,control_guidance_start,control_guidance_end,disable_preprocessing],
                        queue=False,
                    )

                    next_multi_control_image.click(
                        next_view_control,
                        inputs=[],
                        outputs=[model_control_net,controlnet_img,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,preprocessor_name,controlnet_scale,control_guidance_start,control_guidance_end,disable_preprocessing],
                        queue=False,
                    )

                    apply_multi_control_image.click(
                        control_net_muti,
                        inputs=[model_control_net,controlnet_img,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,preprocessor_name,controlnet_scale,control_guidance_start,control_guidance_end,disable_preprocessing],
                        outputs=[controlnet_img],
                        queue=False,
                    )
                    apply_edit_multi.click(
                        apply_edit_control_net,
                        inputs=[model_control_net,controlnet_img,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,preprocessor_name,controlnet_scale,control_guidance_start,control_guidance_end,disable_preprocessing],
                        outputs=[model_control_net,controlnet_img,low_threshold,high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,preprocessor_name,multi_controlnet,previous_multi_control_image,next_multi_control_image,apply_edit_multi,complete_change_multi,controlnet_scale,control_guidance_start,control_guidance_end,disable_preprocessing],
                        queue=False,
                    )

                    complete_change_multi.click(
                        complete_edit_multi,
                        inputs=[],
                        outputs=[edit_multi_control_image,apply_multi_control_image,controlnet_img,apply_edit_multi,complete_change_multi,next_multi_control_image,previous_multi_control_image],
                        queue=False,
                    )

                    controlnet_img.change(
                        change_image_condition,
                        inputs=[image_condition],
                        outputs=[image_condition],
                        queue=False,
                    )
                    
                    model_control_net.change(
                        change_control_net,
                        inputs=[model_control_net, low_threshold, high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose],
                        outputs=[low_threshold, high_threshold,has_body_openpose,has_hand_openpose,has_face_openpose,preprocessor_name],
                        queue=False,
                    )

            with gr.Tab("T2I Adapter"):
                with gr.Group():
                    with gr.Row(): 
                        adapter_enabled = gr.Checkbox(label="Enable T2I Adapter", value=False)
                        disable_preprocessing_adapter = gr.Checkbox(label="Disable preprocessing", value=False)
                        multi_adapter = gr.Checkbox(label="Enable Multi T2I Adapter", value=False)
                    model_adapter = gr.Dropdown(
                        choices=[k for k in adapter_lst],
                        label="Model Controlnet",
                        value=adapter_lst[0],
                    )
                    with gr.Row():
                        low_threshold_adapter = gr.Slider(
                            label="Canny low threshold", value=100, minimum=1, maximum=255, step=1
                        )
                        high_threshold_adapter = gr.Slider(
                            label="Canny high threshold", value=200, minimum=1, maximum=255, step=1
                        )
                    with gr.Row():
                        has_body_openpose_adapter = gr.Checkbox(label="Has body", value=True,visible= False)
                        has_hand_openpose_adapter = gr.Checkbox(label="Has hand", value=False,visible= False)
                        has_face_openpose_adapter = gr.Checkbox(label="Has face", value=False,visible= False)
                    preprocessor_adapter = gr.Radio(
                        label="Preprocessor",
                        type="value",
                        visible= False,
                    )
                    with gr.Row():
                        adapter_conditioning_scale = gr.Slider(
                            label="Conditioning scale", value=1, minimum=0, maximum=2, step=0.01
                        )
                        adapter_conditioning_factor = gr.Slider(
                            label="Conditioning factor", value=1, minimum=0, maximum=1, step=0.01
                        )
                    with gr.Row():
                        adapter_img = gr.Image(
                            image_mode="RGB",
                            source="upload",
                            label = "Image",
                            type = 'pil',
                        )
                        image_condition_adapter = gr.Image(interactive=False,image_mode="RGB",label = "Preprocessor Preview",type = 'pil')
                    adapter_image_click = gr.Button(value="Preview")
                    with gr.Row():
                        previous_multi_adapter_image = gr.Button(value="Previous adapter setting",visible = False)
                        next_multi_adapter_image = gr.Button(value="Next adapter setting",visible = False)
                    with gr.Row():
                        edit_multi_adapter_image = gr.Button(value="Edit previous setting",visible = False)
                        apply_multi_adapter_image = gr.Button(value="Apply setting",visible = False)
                    with gr.Row():
                        apply_edit_multi_adapter = gr.Button(value="Apply change",visible = False)
                        complete_change_multi_adapter = gr.Button(value="Complete change",visible = False)

                    adapter_image_click.click(
                        preview_image_adapter,
                        inputs=[model_adapter,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,adapter_img,preprocessor_adapter,multi_adapter,disable_preprocessing_adapter],
                        outputs=[image_condition_adapter],
                        queue=False,
                    )
                    multi_adapter.change(
                        multi_adapter_function,
                        inputs=[multi_adapter],
                        outputs=[adapter_enabled,edit_multi_adapter_image,apply_multi_adapter_image,previous_multi_adapter_image,next_multi_adapter_image,apply_edit_multi_adapter,complete_change_multi_adapter],
                        queue=False,
                    )
                    edit_multi_adapter_image.click(
                        edit_multi_adapter_image_function,
                        inputs=[],
                        outputs=[previous_multi_adapter_image,next_multi_adapter_image,apply_edit_multi_adapter,complete_change_multi_adapter,edit_multi_adapter_image,apply_multi_adapter_image,model_adapter,adapter_img,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter,adapter_conditioning_scale,adapter_conditioning_factor,disable_preprocessing_adapter],
                        queue=False,
                    )

                    previous_multi_adapter_image.click(
                        previous_view_adapter,
                        inputs=[],
                        outputs=[model_adapter,adapter_img,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter,adapter_conditioning_scale,adapter_conditioning_factor,disable_preprocessing_adapter],
                        queue=False,
                    )

                    next_multi_adapter_image.click(
                        next_view_adapter,
                        inputs=[],
                        outputs=[model_adapter,adapter_img,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter,adapter_conditioning_scale,adapter_conditioning_factor,disable_preprocessing_adapter],
                        queue=False,
                    )

                    apply_multi_adapter_image.click(
                        adapter_muti,
                        inputs=[model_adapter,adapter_img,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter,adapter_conditioning_scale,adapter_conditioning_factor,disable_preprocessing_adapter],
                        outputs=[adapter_img],
                        queue=False,
                    )
                    apply_edit_multi_adapter.click(
                        apply_edit_adapter,
                        inputs=[model_adapter,adapter_img,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter,adapter_conditioning_scale,adapter_conditioning_factor,disable_preprocessing_adapter],
                        outputs=[model_adapter,adapter_img,low_threshold_adapter,high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter,multi_adapter,previous_multi_adapter_image,next_multi_adapter_image,apply_edit_multi_adapter,complete_change_multi_adapter,adapter_conditioning_scale,adapter_conditioning_factor,disable_preprocessing_adapter],
                        queue=False,
                    )

                    complete_change_multi_adapter.click(
                        complete_edit_multi_adapter,
                        inputs=[],
                        outputs=[edit_multi_adapter_image,apply_multi_adapter_image,adapter_img,apply_edit_multi_adapter,complete_change_multi_adapter,next_multi_adapter_image,previous_multi_adapter_image],
                        queue=False,
                    )

                    adapter_img.change(
                        change_image_condition_adapter,
                        inputs=[image_condition_adapter],
                        outputs=[image_condition_adapter],
                        queue=False,
                    )
                    
                    model_adapter.change(
                        change_control_net,
                        inputs=[model_adapter, low_threshold_adapter, high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter],
                        outputs=[low_threshold_adapter, high_threshold_adapter,has_body_openpose_adapter,has_hand_openpose_adapter,has_face_openpose_adapter,preprocessor_adapter],
                        queue=False,
                    )

            diffuser_pipeline.change(
                mode_diffuser_pipeline_sampler,
                inputs=[diffuser_pipeline, sampler,sampler_hires],
                outputs=[diffuser_pipeline,sampler,sampler_hires],
                queue=False,
            )
            hr_enabled.change(
                lambda g, x, w, h: gr.Checkbox.update(
                    label=res_cap(g, w, h, x)
                ),
                inputs=[hr_enabled, hr_scale, width, height],
                outputs=hr_enabled,
                queue=False,
            )

            adapter_enabled.change(
                mode_diffuser_pipeline,
                inputs=[adapter_enabled],
                outputs=[adapter_enabled,multi_adapter],
                queue=False,
            )

            controlnet_enabled.change(
                mode_diffuser_pipeline,
                inputs=[controlnet_enabled],
                outputs=[controlnet_enabled,multi_controlnet],
                queue=False,
            )

            with gr.Tab("Embeddings/Loras"):

                ti_state = gr.State(dict())

                with gr.Group():
                    with gr.Row():
                        with gr.Column():
                            ti_vals = gr.CheckboxGroup(label="Chosing embeddings")
                            embs_choose = gr.Text(label="Embeddings chosen")
                            with gr.Row():
                                choose_em = gr.Button(value="Select Embeddings")
                                delete_em = gr.Button(value="Delete Embeddings")
                        choose_em.click(choose_tistate,inputs=[ti_vals],outputs=[ti_state,embs_choose,ti_vals],queue=False,)
                        delete_em.click(delete_embed,inputs=[ti_vals,ti_state,embs_choose],outputs=[ti_vals,ti_state,embs_choose],queue=False,)

                    with gr.Row():
                        with gr.Column():
                            lora_vals = gr.Dropdown(choices=[k for k in lora_lst],label="Chosing Lora",value=lora_lst[0],)
                            delete_lora_but = gr.Button(value="Delete Lora")
                        delete_lora_but.click(lora_delete,inputs=[lora_vals],outputs=[lora_vals],queue=False,)
                with gr.Row():

                    uploads = gr.Files(label="Upload new embeddings/lora")

                    with gr.Column():
                        lora_scale = gr.Slider(
                            label="Lora scale",
                            minimum=0,
                            maximum=2,
                            step=0.01,
                            value=1.0,
                        )
                        btn = gr.Button(value="Upload")
                        btn_del = gr.Button(value="Reset")

                btn.click(
                    add_net,
                    inputs=[uploads],
                    outputs=[ti_vals, lora_vals, uploads],
                    queue=False,
                )
                btn_del.click(
                    clean_states,
                    inputs=[ti_state],
                    outputs=[ti_state, ti_vals, lora_vals, uploads,embs_choose],
                    queue=False,
                )

    gr.HTML(
        f"""
            <div class="finetuned-diffusion-div">
              <div>
                <h1>Spatial Control</h1>
              </div>
              <p>
                Define the object's region.
                The black areas of images represent the regions which you want to control when extracting color regions.
              </p>
            </div>
        """
    )

    with gr.Row():

        with gr.Column(scale=55):
            formula_button = gr.Dropdown(
                choices=[k[0] for k in formula],
                label="Formual",
                value=formula[0][0],
            )

            rendered = gr.Image(
                invert_colors=True,
                source="canvas",
                interactive=False,
                image_mode="RGBA",
            )

        with gr.Column(scale=45):

            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=70):

                        text = gr.Textbox(
                            lines=2,
                            interactive=True,
                            label="Token to Draw: (Separate by comma)",
                        )

                        radio = gr.Radio([], label="Tokens",visible = False)

                    sk_update = gr.Button(value="Update").style(
                        rounded=(False, True, True, False)
                    )

            with gr.Tab("SketchPad"):

                sp = gr.Image(
                    width = 512,
                    height = 512,
                    image_mode="L",
                    tool="sketch",
                    source="canvas",
                    interactive=False
                )

                with gr.Row():
                    mask_outsides = gr.Slider(
                        label="Decrease unmarked region weight",
                        minimum=0,
                        maximum=3,
                        step=0.01,
                        value=0,
                    )

                    strength = gr.Slider(
                        label="Token-Region strength",
                        minimum=0,
                        maximum=3,
                        step=0.01,
                        value=0.5,
                    )

                width.change(
                    apply_size_sketch,
                    inputs=[width, height,global_stats,inf_image],
                    outputs=[global_stats, rendered,sp],
                    queue=False,
                )

                height.change(
                    apply_size_sketch,
                    inputs=[width, height,global_stats,inf_image],
                    outputs=[global_stats, rendered,sp],
                    queue=False,
                )

                inf_image.change(
                    apply_size_sketch,
                    inputs=[width, height,global_stats,inf_image],
                    outputs=[global_stats, rendered,sp],
                    queue=False,
                )


                sk_update.click(
                    detect_text,
                    inputs=[text, global_stats, width, height,formula_button,inf_image],
                    outputs=[global_stats, sp, radio, rendered,formula_button],
                    queue=False,
                )
                radio.change(
                    switch_canvas,
                    inputs=[radio, global_stats, width, height,inf_image],
                    outputs=[sp, strength, mask_outsides, rendered],
                    queue=False,
                )
                sp.edit(
                    apply_canvas,
                    inputs=[radio, sp, global_stats, width, height,inf_image],
                    outputs=[global_stats, rendered],
                    queue=False,
                )
                strength.change(
                    apply_weight,
                    inputs=[radio, strength, global_stats],
                    outputs=[global_stats],
                    queue=False,
                )
                mask_outsides.change(
                    apply_option,
                    inputs=[radio, mask_outsides, global_stats],
                    outputs=[global_stats],
                    queue=False,
                )

            with gr.Tab("UploadFile"):

                sp2 = gr.Image(
                    image_mode="RGB",
                    source="upload",
                )

                sp3 = gr.Image(
                    image_mode="L",
                    source="canvas",
                    visible = False,
                    interactive = False,
                )
                with gr.Row():
                    previous_page = gr.Button(value="Previous",visible = False,)
                    next_page = gr.Button(value="Next",visible = False,)


                with gr.Row():
                    mask_outsides2 = gr.Slider(
                        label="Decrease unmarked region weight",
                        minimum=0,
                        maximum=3,
                        step=0.01,
                        value=0,
                    )

                    strength2 = gr.Slider(
                        label="Token-Region strength",
                        minimum=0,
                        maximum=3,
                        step=0.01,
                        value=0.5,
                    )


                with gr.Row():
                    apply_style = gr.Button(value="Apply")
                    apply_clustering_style = gr.Button(value="Extracting color regions")

                with gr.Row():
                    add_style = gr.Button(value="Apply",visible = False)
                    complete_clustering = gr.Button(value="Complete",visible = False)
                
                apply_style.click(
                    apply_image,
                    inputs=[sp2, radio, width, height, strength2, mask_outsides2, global_stats,inf_image],
                    outputs=[global_stats, rendered],
                    queue=False,
                )
                apply_clustering_style.click(
                    apply_base_on_color,
                    inputs=[sp2,global_stats,width, height,inf_image],
                    outputs=[rendered,apply_style,apply_clustering_style,previous_page,next_page,complete_clustering,sp2,sp3,add_style,global_stats],
                    queue=False,
                )
                previous_page.click(
                    previous_image_page,
                    inputs=[sp3],
                    outputs=[sp3],
                    queue=False,
                )
                next_page.click(
                    next_image_page,
                    inputs=[sp3],
                    outputs=[sp3],
                    queue=False,
                )
                add_style.click(
                    apply_image_clustering,
                    inputs=[sp3, radio, width, height, strength2, mask_outsides2, global_stats,inf_image],
                    outputs=[global_stats,rendered],
                    queue=False,
                )
                complete_clustering.click(
                    completing_clustering,
                    inputs=[sp2],
                    outputs=[apply_style,apply_clustering_style,previous_page,next_page,complete_clustering,sp2,sp3,add_style],
                    queue=False,
                )


    inputs = [
        prompt,
        guidance,
        steps,
        width,
        height,
        clip_skip,
        seed,
        neg_prompt,
        global_stats,
        #g_strength,
        inf_image,
        inf_strength,
        hr_enabled,
        hr_method,
        hr_scale,
        hr_denoise,
        sampler,
        ti_state,
        model,
        lora_vals,
        lora_scale,
        formula_button,
        controlnet_enabled,
        model_control_net,
        low_threshold,
        high_threshold,
        has_body_openpose,
        has_hand_openpose,
        has_face_openpose,
        controlnet_img,
        image_condition,
        controlnet_scale,
        preprocessor_name,
        diffuser_pipeline,
        sampler_hires,
        latent_processing,
        control_guidance_start,
        control_guidance_end,
        multi_controlnet,
        disable_preprocessing,
        region_condition,
        hr_process_enabled,
        ip_adapter,
        model_ip_adapter,
        inf_adapt_image,
        inf_adapt_image_strength,
        hr_region_condition,
        adapter_enabled,
        model_adapter,
        low_threshold_adapter,
        high_threshold_adapter,
        has_body_openpose_adapter,
        has_hand_openpose_adapter,
        has_face_openpose_adapter,
        adapter_img,
        image_condition_adapter,
        preprocessor_adapter,
        adapter_conditioning_scale,
        adapter_conditioning_factor,
        multi_adapter,
        disable_preprocessing_adapter,
        ip_adapter_multi,
        guidance_rescale,
        inf_control_adapt_image,
        long_encode,
    ]
    outputs = [image_out,gallery]
    prompt.submit(inference, inputs=inputs, outputs=outputs)
    generate.click(inference, inputs=inputs, outputs=outputs)

print(f"Space built in {time.time() - start_time:.2f} seconds")
demo.queue().launch(share=True,debug=True)
demo.launch(enable_queue=True, server_name="0.0.0.0", server_port=7860)
