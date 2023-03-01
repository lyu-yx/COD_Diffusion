from abc import abstractmethod
import math
import sys
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
sys.path.append('')
from guided_diffusion.pvtv2 import pvt_v2_b2
from guided_diffusion.unet import PriorGuidedFeatureRefinement, CrossDomainFeatureFusion, UNetModel, EncoderUNetModel
from guided_diffusion.script_util import create_model, create_gaussian_diffusion


if __name__ == "__main__":
    model = create_model(352, 128, 2, num_heads=4, attention_resolutions='11')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")


    with open('model.txt', 'w') as f:
        f.write(str(model))