import torch
import os
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
import random

lst_model_segmentation = {
    "Convnet tiny": "openmmlab/upernet-convnext-tiny",
    "Convnet small": "openmmlab/upernet-convnext-small",
    "Convnet base": "openmmlab/upernet-convnext-base",
    "Convnet large": "openmmlab/upernet-convnext-large",
    "Convnet xlarge": "openmmlab/upernet-convnext-xlarge",
    "Swin tiny": "openmmlab/upernet-swin-tiny",
    "Swin small": "openmmlab/upernet-swin-small",
    "Swin base": "openmmlab/upernet-swin-base",
    "Swin large": "openmmlab/upernet-swin-large",
}

def preprocessing_segmentation(method,image):
    global lst_model_segmentation
    method = lst_model_segmentation[method]
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda' 
    image_processor = AutoImageProcessor.from_pretrained(method)
    image_segmentor = UperNetForSemanticSegmentation.from_pretrained(method).to(device)

    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        outputs = image_segmentor(pixel_values)
    seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3

    seg = seg.to('cpu')
    unique_values = torch.unique(seg)

    lst_color = []
    for i in unique_values:
        color = [random.randrange(0,256), random.randrange(0,256), random.randrange(0,256)]
        while color in lst_color:
           color = [random.randrange(0,256), random.randrange(0,256), random.randrange(0,256)]
        color_seg[seg == i, :] = color
        lst_color.append(color)
    color_seg = color_seg.astype(np.uint8)
    control_image = Image.fromarray(color_seg)
    return control_image