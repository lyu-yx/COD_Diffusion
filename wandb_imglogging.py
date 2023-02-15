import wandb
import argparse
import blobfile as bf
import logging
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from torchvision.transforms import ToPILImage
from PIL import Image
from tqdm import tqdm

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def val_single_img(img_pth, gt_pth, itr_num):
    """
    validation function
    """
    # model_path = "./results/" + f"savedmodel{(5000 * itr_num):06d}.pt"
    model_path = "/.results/emasavedmodel_0.9999_000000.pt"
    def create_argparser():
        defaults = dict(
            data_dir="../BUDG/dataset/TestDataset/CAMO/",
            clip_denoised=True,
            num_samples=1,
            batch_size=1,
            use_ddim=False,
            model_path=model_path,
            num_ensemble=3      # number of samples in the ensemble
        )
        defaults.update(model_and_diffusion_defaults())
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser

    args = create_argparser().parse_args()

    with th.no_grad():
        model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

        image = Image.open(img_pth).resize((352,352))
        
        image = np.asarray(image, np.float32)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        
        gt = Image.open(gt_pth).resize((352,352))
        gt = np.asarray(gt, np.float32)
        gt = np.expand_dims(gt, axis=0)
        gt = np.expand_dims(gt, axis=0)
        
        img_size = np.asarray(gt, np.float32).shape[-2:]

        image = np.concatenate((image, gt), axis=1) 
        # print('image.shape', image.shape)
        image = th.Tensor(image).cuda()
        gt = th.Tensor(gt).cuda()

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            # start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, _, _ = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), image, # image = orgimg + noise
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            single_img_output = F.interpolate(sample, size=img_size, mode='bilinear', align_corners=False)
            single_img_output = single_img_output.squeeze().cpu().numpy()
            single_img_output = (single_img_output - single_img_output.min()) / (single_img_output.max() - single_img_output.min() + 1e-8)
            

            image = wandb.Image(single_img_output, caption="Input image")
            wandb.log({"diffusion_result": image})

if __name__=="__main__":


    wandb.init(project="Loging img test", name="diffusion_only")

    single_visimg_pth="../BUDG/dataset/TestDataset/COD10K/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-4.jpg"
    single_visgt_pth="../BUDG/dataset/TestDataset/COD10K/GT/COD10K-CAM-1-Aquatic-1-BatFish-4.png"

    val_single_img(single_visimg_pth, single_visgt_pth, 0)