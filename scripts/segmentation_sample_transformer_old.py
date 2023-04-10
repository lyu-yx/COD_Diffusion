"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

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

from guided_diffusion.script_util_transformer_old import (
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

# def visualize(img):
#     _min = img.min()
#     _max = img.max()
#     normalized_img = (img - _min)/ (_max - _min)
#     return normalized_img

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    val_loader = EvalDataset(args.data_dir, gt_root=args.gt_dir, edge_root=args.edge_dir, testsize=352)
    # datal = th.utils.data.DataLoader(
    #     val_loader,
    #     batch_size=1,
    #     shuffle=False)
    # data = iter(datal)


    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model = th.nn.DataParallel(model, device_ids=[int(id) for id in args.multi_gpu.split(',')])
    model.to(dist_util.dev())
    # model.to(device = th.device('cuda'))

    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    # while len(all_images) * args.batch_size < args.num_samples:
    for i in tqdm(range(val_loader.size)):
        img, gt, edges, name, _ = val_loader.load_data() # should return an image from the dataloader "data"  b: 1, 3, 352, 352, c: 1, 1, 352, 352
        noise = th.randn_like(img[:, :1, ...])
        img = th.cat((img, noise), dim=1)     # add a noise channel
        img_size = np.asarray(gt, np.float32).shape
        # logger.log("sampling...")
        # start = th.cuda.Event(enable_timing=True)
        # end = th.cuda.Event(enable_timing=True)
        # start.record()
        sample_arrays = []
        threshold = args.threshold
        foregroundValue = 1.0
        for i in range(args.num_ensemble):  # this is for the generation of an ensemble of n masks.
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, edge = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img, 
                step=args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            output = F.interpolate(sample, size=img_size, mode='bilinear', align_corners=False)
            output = output.detach().squeeze().cpu().numpy()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)
            output[output <= threshold] = 0
            output[output > threshold] = 1
            output = sitk.Cast(sitk.GetImageFromArray(output), sitk.sitkUInt8)
            sample_arrays.append(output)
            # plt.imsave(args.save_pth + str(name).split('.')[0] + '_' + str(i) + '.png', output, cmap='gist_gray') # save the generated mask
        
        
        images = [array for array in sample_arrays]
        staple_result = sitk.STAPLE(images, foregroundValue)
        threshold_filter = sitk.BinaryThresholdImageFilter()
        threshold_filter.SetLowerThreshold(0.75)
        threshold_filter.SetUpperThreshold(1.0)
        threshold_filter.SetInsideValue(1)
        threshold_filter.SetOutsideValue(0)
        binary_staple = threshold_filter.Execute(staple_result)
        binary_staple = sitk.GetArrayFromImage(binary_staple) 

        
        
        plt.imsave(args.save_pth + str(name).split('.')[0] + '.png', binary_staple, cmap='gist_gray') # save the generated mask
        
        # end.record()
        # th.cuda.synchronize()
        # print('time for {} sample: {} second'.format(args.num_ensemble, start.elapsed_time(end)/1000))  #time measurement for the generation of 1 sample

def create_argparser():
    defaults = dict(
        data_dir="../BUDG/dataset/TestDataset/CAMO/Imgs/",
        gt_dir="../BUDG/dataset/TestDataset/CAMO/GT/",
        edge_dir="../BUDG/dataset/TestDataset/CAMO/Edge/",
        size=352,
        num_channels=128,
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        gpu_dev = "0",
        multi_gpu = None, # "0,1,2"
        model_path="./results/savedmodel075000.pt",
        num_ensemble=5,      #number of samples in the ensemble
        save_pth="./results/val/",
        threshold=0.5,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
