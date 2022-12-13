import copy
import functools
import os

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

from . import metrics
from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
# from visdom import Visdom

# viz = Visdom(port=8850)
# loss_window = viz.line( Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(), opts=dict(xlabel='epoch', ylabel='Loss', title='loss'))
# grad_window = viz.line(Y=th.zeros((1)).cpu(), X=th.zeros((1)).cpu(),
#                            opts=dict(xlabel='step', ylabel='amplitude', title='gradient'))


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        classifier,
        diffusion,
        # data,
        data_loader,
        val_loader,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        single_visimg_pth,
        single_visgt_pth,
    ):
        self.model = model
        self.data_loader = data_loader
        self.classifier = classifier
        self.diffusion = diffusion
        # self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.single_visimg_pth = single_visimg_pth
        self.single_visgt_pth = single_visgt_pth
        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            print('resume model')
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        i = 0
        data_iter = iter(self.data_loader)

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):


            try:
                # for i, (img, gt, edge) in enumerate(self.data_loader, start=1):
                    batch, cond = next(data_iter)
                    
            except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader
                    data_iter = iter(self.data_loader)
                    batch, cond = next(data_iter)

            self.run_step(batch, cond)

            i += 1
          
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            # val_single_img(self.single_visimg_pth, self.single_visgt_pth)
            self.save()

    def run_step(self, batch, cond):
        batch=th.cat((batch, cond), dim=1)
        cond={}
        sample = self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        return sample

    def forward_backward(self, batch, cond):

        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }

            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses_segmentation,
                self.ddp_model,
                self.classifier,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses1 = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses1 = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            losses = losses1[0]
            sample = losses1[1]

            loss = (losses["loss"] * weights).mean()

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
            return sample

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"savedmodel{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"emasavedmodel_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"optsavedmodel{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
    
def create_argparser():
    defaults = dict(
        data_dir="../BUDG/dataset/TestDataset/CAMO/",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=3      # number of samples in the ensemble
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_metric_dict, best_score, best_epoch
    FM = metrics.Fmeasure()
    SM = metrics.Smeasure()
    EM = metrics.Emeasure()
    metrics_dict = dict()
    args = create_argparser().parse_args()
    model.eval()
    with th.no_grad():
        model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            #res = model(image)
            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)

            for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                sample, x_noisy, org = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size), image,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample

            sample_result = th.tensor(sample)
            # viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
            th.save(sample_result, './results/'+str(name)) #save the generated mask

            res = F.upsample(sample_result, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            # TODO
            '''
            Threshold of predict mask
            '''
            FM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)

        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(mxFm=FM.get_results()['fm']['curve'].max().round(3))
        metrics_dict.update(mxEm=EM.get_results()['em']['curve'].max().round(3))

        cur_score = metrics_dict['Sm'] + metrics_dict['mxFm'] + metrics_dict['mxEm']

        if epoch == 1:
            best_score = cur_score
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm']))
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch
                th.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))
            logging.info('[Cur Epoch: {}] Metrics (mxFm={}, Sm={}, mxEm={})\n[Best Epoch:{}] Metrics (mxFm={}, Sm={}, mxEm={})'.format(
                epoch, metrics_dict['mxFm'], metrics_dict['Sm'], metrics_dict['mxEm'],
                best_epoch, best_metric_dict['mxFm'], best_metric_dict['Sm'], best_metric_dict['mxEm']))


def val_single_img(img_pth, gt_pth):
    """
    validation function
    """
    args = create_argparser().parse_args()

    with th.no_grad():
        model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

        image = Image.open(img_pth)
        image = np.asarray(image, np.float32)
        
        gt = Image.open(gt_pth)
        gt = np.asarray(gt, np.float32)
        
        image = image.cuda()
        gt = gt.cuda()
        #res = model(image)
        # start = th.cuda.Event(enable_timing=True)
        # end = th.cuda.Event(enable_timing=True)

        for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            # start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, _, _ = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), image,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

        # end.record()
        # th.cuda.synchronize()
        # print('time for 1 sample', start.elapsed_time(end))  # time measurement for the generation of 1 sample

        sample_result = th.tensor(sample)
        
        # viz.image(visualize(sample[0, 0, ...]), opts=dict(caption="sampled output"))
        # th.save(sample_result, './results/'+str(name)) #save the generated mask

        img_out = F.upsample(sample_result, size=gt.shape, mode='bilinear', align_corners=False)
        img_out = img_out.sigmoid().data.cpu().numpy().squeeze()
        img_out = (img_out - img_out.min()) / (img_out.max() - img_out.min() + 1e-8)
        image = wandb.Image(img_out, caption="Input image")
        wandb.log({"diffusion_result": image})
        # TODO
        '''
        Threshold of predict mask
        '''
       
