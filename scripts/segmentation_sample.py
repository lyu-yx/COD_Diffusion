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
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.dataset import CamObjDataset
# from guided_diffusion.bratsloader import BRATSDataset
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

    ds = CamObjDataset(args.data_dir, gt_root=args.gt_dir, trainsize=352)
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=False)
    data = iter(datal)
    all_images = []
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    cnt = 0
    while len(all_images) * args.batch_size < args.num_samples:
        cnt += 1
        print('curr_cnt', cnt)
        img, gt = next(data)  #should return an image from the dataloader "data"  b: 1, 3, 352, 352, c: 1, 1, 352, 352
        noise = th.randn_like(img[:, :1, ...])
        img = th.cat((img, noise), dim=1)     # add a noise channel
        # slice_ID=path[0].split("/", -1)[3]

        # viz.image(visualize(img[0,0,...]), opts=dict(caption="img input0"))
        # viz.image(visualize(img[0, 1, ...]), opts=dict(caption="img input1"))
        # viz.image(visualize(img[0, 2, ...]), opts=dict(caption="img input2"))
        # viz.image(visualize(img[0, 3, ...]), opts=dict(caption="img input3"))
        # viz.image(visualize(img[0, 4, ...]), opts=dict(caption="img input4"))

        logger.log("sampling...")

        # start = th.cuda.Event(enable_timing=True)
        # end = th.cuda.Event(enable_timing=True)


        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            # start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            # end.record()
            th.cuda.synchronize()
            #print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            s = th.tensor(sample)
            # viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            th.save(s, './results/'+str(i)) #save the generated mask

def create_argparser():
    defaults = dict(
        data_dir="../BUDG/dataset/TestDataset/CAMO/Imgs/",
        gt_dir="../BUDG/dataset/TestDataset/CAMO/GT/",
        size=352,
        num_channels=128,
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        gpu_dev = "0",
        multi_gpu = None, # "0,1,2"
        model_path="./results/savedmodel075000.pt",
        num_ensemble=5      #number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
