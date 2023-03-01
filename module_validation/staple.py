import argparse
import os
#import nibabel as nib
#from visdom import Visdom
#viz = Visdom(port=8850)
import sys
import random


sys.path.append(".")
import numpy as np
import time
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from guided_diffusion import dist_util, logger
from guided_diffusion.dataset import test_dataset as EvalDataset
# from guided_diffusion.bratsloader import BRATSDataset
import guided_diffusion.staple as staple

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)



if __name__ == "__main__":
    arrays = []
    filepaths = [
        'results/camo_115000/camourflage_00018_0.png',
        'results/camo_115000/camourflage_00018_1.png',
        'results/camo_115000/camourflage_00018_2.png']

    
    threshold = 128
    for filepath in filepaths:
        image = sitk.ReadImage(str(filepath), sitk.sitkUInt8)
        array = sitk.GetArrayFromImage(image)
        array = np.array(array)
        array[array <= threshold] = 0
        array[array > threshold] = 1
        # plt.imsave('mask.png', array, cmap='gist_gray') # save the generated mask
        arrays.append(array)

    foregroundValue = 1.0

    images = [sitk.GetImageFromArray(array) for array in arrays]
    
     
    staple_result = sitk.STAPLE(images, foregroundValue)
    staple_result = sitk.GetArrayFromImage(staple_result) 
    # staple_result = (staple_result - staple_result.min()) / (staple_result.max() - staple_result.min() + 1e-8)
    
    # staple_result = staple_result > threshold
        
    # staple_result = staple.STAPLE(sample_arrays, convergence_threshold=0)
    # result = staple_result.run()

    
    plt.imsave('tesdt_staple.png', staple_result, cmap='gist_gray') # save the generated mask

