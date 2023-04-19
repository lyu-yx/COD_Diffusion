"""
Train a diffusion model on images.
"""
import sys
import argparse
import wandb

sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.dataset import CamObjDataset, get_loader, test_dataset
from guided_diffusion.resample import create_named_schedule_sampler
# from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util_full_ablation_tsfm_only import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
#from visdom import Visdom
#viz = Visdom(port=8850)



# wandb.config = {
#   "learning_rate": 1e-5,
#   "epochs": 200,
#   "batch_size": 32
# }


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        # model.to(device = th.device('cuda'))
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, maxt=1000)

    logger.log("creating data loader...")
    #ds = BRATSDataset(args.data_dir, test_flag=False)
    #ds = CamObjDataset(args.data_dir, test_flag=False)
    
    train_loader = get_loader(image_root=args.train_root + 'Imgs/',
                              gt_root=args.train_root + 'GT/',
                              edge_root=args.train_root + 'Edge/',
                              batchsize=args.batch_size,
                              trainsize=args.train_size,
                              num_workers=4)
    val_loader = test_dataset(image_root=args.val_root + 'Imgs/',
                              gt_root=args.val_root + 'GT/',
                              edge_root=args.val_root + 'Edge/',
                              testsize=args.test_size)
    data = iter(train_loader)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        data_loader=train_loader,
        val_loader=val_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_pth=args.model_save_folder,
        edge_partition = args.edge_partition,
        single_visimg_pth=args.single_visimg_pth,
        single_visgt_pth=args.single_visgt_pth,
    ).run_loop()


def create_argparser():
    defaults = dict(
        train_root="../BUDG/dataset/TrainDataset/",
        val_root="../BUDG/dataset/TestDataset/CAMO/",
        train_size=352,
        test_size=352,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint="",  # "./results/pretrainedmodel.pt",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "0",
        multi_gpu = "0,1,2,3", # "0,1,2"
        model_save_folder= "",
        edge_partition = 0.002,
        single_visimg_pth="../BUDG/dataset/TestDataset/COD10K/Imgs/COD10K-CAM-1-Aquatic-1-BatFish-4.jpg",
        single_visgt_pth= "../BUDG/dataset/TestDataset/COD10K/GT/COD10K-CAM-1-Aquatic-1-BatFish-4.png",
    )                     
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    wandb.init(project="Diffusion", name="ablation_tsfm_only")
    main()
