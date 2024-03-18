import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


class IPAdapterFullImageProjection(nn.Module):
    def __init__(self, image_embed_dim=1024, cross_attention_dim=1024, mult=1, num_tokens=1):
        super().__init__()
        from diffusers.models.attention import FeedForward

        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim
        self.ff = FeedForward(image_embed_dim, cross_attention_dim * num_tokens, mult=mult, activation_fn="gelu")
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds: torch.FloatTensor):
        if self.num_tokens == 4:
            x = self.ff(image_embeds)
            x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
            return self.norm(x)
        else:
            return self.norm(self.ff(image_embeds))