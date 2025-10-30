#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from copy import deepcopy
import math
import time
import functools
import lpips
import numpy as np
import random
from einops import rearrange

from engine.trainer_base import TrainerBase
from utils import util_image
from utils import util_net
from utils import util_common
from models.basic_ops import mean_flat
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt


class TrainerDifIR(TrainerBase):
    def setup_optimizaton(self):
        super().setup_optimizaton()
        self.log_stream = torch.cuda.Stream()
        self.data_stream = torch.cuda.Stream()
        self.log_info_for_next_iter = None
        self.loss_mean = None
        self.loss_count = None
        
        def lr_lambda(step):
            step = step + 1
            warmup = self.configs.train.warmup_iterations
            if step < warmup:
                return step / warmup
            return 1.0
        
        if self.configs.train.lr_schedule == 'cosin':
            model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.optimizer,
                    T_max=self.configs.train.iterations - self.configs.train.warmup_iterations,
                    eta_min=self.configs.train.lr_min,
                    )
        else:
            model_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=0)
        
        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda),
                model_scheduler
            ],
            milestones=[self.configs.train.warmup_iterations]
        )

    def build_model(self):
        super().build_model()
        
        # autoencoder
        if self.configs.autoencoder is not None:
            ckpt = torch.load(self.configs.autoencoder.ckpt_path, map_location=f"cuda:{self.rank}")
            if self.rank == 0:
                self.logger.info(f"Restoring autoencoder from {self.configs.autoencoder.ckpt_path}")
            params = self.configs.autoencoder.get('params', dict)
            autoencoder = util_common.get_obj_from_str(self.configs.autoencoder.target)(**params)
            autoencoder = autoencoder.to(memory_format=torch.channels_last)
            autoencoder.cuda()
            if self.configs.autoencoder.tune_decoder:
                self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
                if self.rank == 0:
                    num_params = 0
                    for key, value in autoencoder.named_parameters():
                        if 'decoder' in key or 'post_quant_conv' in key:
                            num_params += value.numel()
                        else:
                            value.requires_grad = False
                    self.logger.info(f'Finetuning Decoder module: {num_params/10**6:.2f}M...')
            else:
                self.load_model(autoencoder, self.configs.autoencoder.ckpt_path, tag='autoencoder', strict=True)
                self.freeze_model(autoencoder)
                autoencoder.eval()
            if self.configs.train.compile.flag:
                if self.rank == 0:
                    self.logger.info("Begin compiling autoencoder model...")
                # autoencoder = torch.compile(autoencoder, mode=self.configs.train.compile.mode)
                autoencoder.encode = torch.compile(autoencoder.encode, mode=self.configs.train.compile.mode)
                autoencoder.decode = torch.compile(autoencoder.decode, mode=self.configs.train.compile.mode)
                if self.rank == 0:
                    self.logger.info("Compiling Done")
            self.autoencoder = autoencoder
        else:
            self.autoencoder = None

        if self.configs.autoencoder.params.lora_tune_decoder or self.configs.autoencoder.tune_decoder:
            self.freeze_model(self.model)

        # LPIPS metric
        if hasattr(self.configs, 'lpips'):
            lpips_net = self.configs.lpips.net
        else:
            lpips_net = 'vgg'
        if self.rank == 0:
            self.logger.info(f"Loading LPIPS Metric: {lpips_net}...")
        lpips_loss = lpips.LPIPS(net=lpips_net).to(f"cuda:{self.rank}").to(memory_format=torch.channels_last)
        for params in lpips_loss.parameters():
            params.requires_grad_(False)
        lpips_loss.eval()
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling LPIPS Metric...")
            lpips_loss = torch.compile(lpips_loss, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        self.lpips_loss = lpips_loss

        params = self.configs.diffusion.get('params', dict)
        self.base_diffusion = util_common.get_obj_from_str(self.configs.diffusion.target)(**params)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_size'):
            self.queue_size = self.configs.degradation.get('queue_size', b*10)

        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0

            self.queue_lr = torch.empty(self.queue_size, c, h, w, dtype=self.lq.dtype, device=f"cuda:{self.rank}")  
            _, c2, h2, w2 = self.gt.size()
            self.queue_gt = torch.empty(self.queue_size, c2, h2, w2, dtype=self.gt.dtype, device=f"cuda:{self.rank}")
            self.write_ptr = 0
            self.perm = torch.arange(self.queue_size, device=f"cuda:{self.rank}")


        end = self.write_ptr + b
        if end <= self.queue_size:
            self.queue_lr[self.write_ptr:end].copy_(self.lq.to(f"cuda:{self.rank}", non_blocking=True))
            self.queue_gt[self.write_ptr:end].copy_(self.gt.to(f"cuda:{self.rank}", non_blocking=True))
        else:
            first = self.queue_size - self.write_ptr
            self.queue_lr[self.write_ptr:].copy_(self.lq[:first].to(f"cuda:{self.rank}", non_blocking=True))
            self.queue_gt[self.write_ptr:].copy_(self.gt[:first].to(f"cuda:{self.rank}", non_blocking=True))
            self.queue_lr[:b-first].copy_(self.lq[first:].to(f"cuda:{self.rank}", non_blocking=True))
            self.queue_gt[:b-first].copy_(self.gt[first:].to(f"cuda:{self.rank}", non_blocking=True))
        self.write_ptr = (self.write_ptr + b) % self.queue_size


        sel = self.perm[:b]
        self.perm = torch.roll(self.perm, -b)


        lq_dequeue = self.queue_lr.index_select(0, sel)
        gt_dequeue = self.queue_gt.index_select(0, sel)

        self.lq = lq_dequeue.detach()
        self.gt = gt_dequeue.detach()


    @torch.no_grad()
    def prepare_data(self, data, dtype=torch.float32, realesrgan=None, phase='train'):
        if realesrgan is None:
            realesrgan = self.configs.data.get(phase, dict).type == 'realesrgan'
        if realesrgan and phase == 'train':
            if not hasattr(self, 'jpeger'):
                self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
            if not hasattr(self, 'use_sharpener'):
                self.use_sharpener = USMSharp().cuda()

            im_gt = data['gt'].cuda().to(dtype=dtype) / 255.
            kernel1 = data['kernel1'].cuda()
            kernel2 = data['kernel2'].cuda()
            sinc_kernel = data['sinc_kernel'].cuda()

            ori_h, ori_w = im_gt.size()[2:4]
            if isinstance(self.configs.degradation.sf, int):
                sf = self.configs.degradation.sf
            else:
                assert len(self.configs.degradation.sf) == 2
                sf = random.uniform(*self.configs.degradation.sf)

            if self.configs.degradation.use_sharp:
                im_gt = self.use_sharpener(im_gt)

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out = filter2D(im_gt, kernel1)
            # random resize
            updown_type = random.choices(
                    ['up', 'down', 'keep'],
                    self.configs.degradation['resize_prob'],
                    )[0]
            if updown_type == 'up':
                scale = random.uniform(1, self.configs.degradation['resize_range'][1])
            elif updown_type == 'down':
                scale = random.uniform(self.configs.degradation['resize_range'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, scale_factor=scale, mode=mode)
            # add noise
            gray_noise_prob = self.configs.degradation['gray_noise_prob']
            if random.random() < self.configs.degradation['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out,
                    sigma_range=self.configs.degradation['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    )
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=self.configs.degradation['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            # ----------------------- The second degradation process ----------------------- #
            if random.random() < self.configs.degradation['second_order_prob']:
                # blur
                if random.random() < self.configs.degradation['second_blur_prob']:
                    out = filter2D(out, kernel2)
                # random resize
                updown_type = random.choices(
                        ['up', 'down', 'keep'],
                        self.configs.degradation['resize_prob2'],
                        )[0]
                if updown_type == 'up':
                    scale = random.uniform(1, self.configs.degradation['resize_range2'][1])
                elif updown_type == 'down':
                    scale = random.uniform(self.configs.degradation['resize_range2'][0], 1)
                else:
                    scale = 1
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                        mode=mode,
                        )
                # add noise
                gray_noise_prob = self.configs.degradation['gray_noise_prob2']
                if random.random() < self.configs.degradation['gaussian_noise_prob2']:
                    out = random_add_gaussian_noise_pt(
                        out,
                        sigma_range=self.configs.degradation['noise_range2'],
                        clip=True,
                        rounds=False,
                        gray_prob=gray_noise_prob,
                        )
                else:
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.configs.degradation['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False,
                        )

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if random.random() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.configs.degradation['jpeg_range2'])
                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(
                        out,
                        size=(ori_h // sf, ori_w // sf),
                        mode=mode,
                        )
                out = filter2D(out, sinc_kernel)

            # resize back
            if self.configs.degradation.resize_back:
                out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')
                temp_sf = self.configs.degradation['sf']
            else:
                temp_sf = self.configs.degradation['sf']

            # clamp and round
            im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.configs.degradation['gt_size']
            im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, temp_sf)
            im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
            im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
            self.lq, self.gt, flag_nan = replace_nan_in_batch(im_lq, im_gt)
            if flag_nan:
                with open(f"records_nan_rank{self.rank}.log", 'a') as f:
                    f.write(f'Find Nan value in rank{self.rank}\n')

            # training pair pool
            self._dequeue_and_enqueue()
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract

            return {'lq':self.lq, 'gt':self.gt}
        elif phase == 'val':
            offset = self.configs.train.get('val_resolution', 256)
            for key, value in data.items():
                value = value.permute(0, 3, 1, 2)
                value = value.to(dtype=dtype) / 255.
                h, w = value.shape[2:]
                if h > offset and w > offset:
                    h_end = int((h // offset) * offset)
                    w_end = int((w // offset) * offset)
                    value = value[:, :, :h_end, :w_end]
                else:
                    h_pad = math.ceil(h / offset) * offset - h
                    w_pad = math.ceil(w / offset) * offset - w
                    padding_mode = self.configs.train.get('val_padding_mode', 'reflect')
                    value = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)
                # Convert from [0, 1] to [-1, 1]
                value = (value - 0.5) / 0.5
                data[key] = value
            return {key:value.cuda().to(dtype=dtype, memory_format=torch.channels_last) for key, value in data.items()}
        else:
            return {key:value.cuda().to(dtype=dtype, memory_format=torch.channels_last) for key, value in data.items()}

    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        if self.configs.train.use_amp:
            with autocast(device_type="cuda"):
                losses, z_t, z0_pred = dif_loss_wrapper()
                losses['loss'] = losses['mse']
                loss = losses['loss'].mean() / num_grad_accumulate
        else:
            losses, z_t, z0_pred = dif_loss_wrapper()
            losses['loss'] = losses['mse']
            loss = losses['loss'].mean() / num_grad_accumulate
            
        if self.amp_scaler is None:
            loss.backward()
            
        else:
            self.amp_scaler.scale(loss).backward()
            
        return losses, z0_pred, z_t

    def training_step(self, data):
        # print(self.log_info_for_next_iter)


        current_batchsize = data['gt'].shape[0]
        micro_batchsize = self.configs.train.microbatch
        num_grad_accumulate = math.ceil(current_batchsize / micro_batchsize)

        log_info = None
        for jj in range(0, current_batchsize, micro_batchsize):
            
            torch.compiler.cudagraph_mark_step_begin()

            micro_data = {key:value[jj:jj+micro_batchsize,] for key, value in data.items()}
            last_batch = (jj+micro_batchsize >= current_batchsize)
            tt = torch.randint(
                    0, self.base_diffusion.num_timesteps,
                    size=(micro_data['gt'].shape[0],),
                    device=f"cuda:{self.rank}",
                    )

            if self.autoencoder is not None:
                latent_downsamping_sf = 2 ** (len(self.configs.autoencoder.params.ddconfig.ch_mult) - 1)
                latent_resolution = micro_data['gt'].shape[-1] // latent_downsamping_sf
                noise_chn = self.configs.autoencoder.params.embed_dim
            else:
                latent_resolution = micro_data['gt'].shape[-1]
                noise_chn = micro_data['gt'].shape[1]
            noise = torch.randn(
                    size= (micro_data['gt'].shape[0], noise_chn,) + (latent_resolution, ) * 2,
                    device=micro_data['gt'].device,
                    )
            if self.configs.model.params.cond_lq:
                model_kwargs = {'lq':micro_data['lq'],}
                if 'mask' in micro_data:
                    model_kwargs['mask'] = micro_data['mask']
            else:
                model_kwargs = None

            
            if self.configs.model.params.cond_lq:
                micro_data['lq'] = micro_data['lq'].detach()
                if 'mask' in micro_data:
                    micro_data['mask'] = micro_data['mask'].detach()

            compute_losses = functools.partial(
                self.base_diffusion.training_losses,
                self.model,
                micro_data['gt'],
                micro_data['lq'],
                tt,
                first_stage_model=self.autoencoder,
                model_kwargs=model_kwargs,
                noise=noise,
            )
            if last_batch or self.num_gpus <= 1:
                losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
                
            else:
                with self.model.no_sync():
                    losses, z0_pred, z_t = self.backward_step(compute_losses, micro_data, num_grad_accumulate, tt)
                    

            # make logging
            if last_batch:
                log_info = (losses, tt, micro_data, z_t, z0_pred.detach(), self.current_iters)

        if self.log_stream is not None: torch.cuda.current_stream().wait_stream(self.log_stream)

        
        if self.configs.train.use_amp:
            self.amp_scaler.step(self.optimizer)
            self.amp_scaler.update()
        else:
            self.optimizer.step()

        # grad zero
        self.optimizer.zero_grad(set_to_none=True)

        if log_info is not None:
            self.log_info_for_next_iter = log_info

        if self.log_info_for_next_iter is not None:
            with torch.cuda.stream(self.log_stream):
                if hasattr(self.configs.train, 'ema_rate'):
                    self.update_ema_model()
                self.log_step_train(*self.log_info_for_next_iter)

    def adjust_lr(self, current_iters=None):
        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

    @torch.no_grad()
    def log_step_train(self, loss, tt, batch, z_t, z0_pred, current_iters, phase='train'):
        '''
        param loss: a dict recording the loss informations
        param tt: 1-D tensor, time steps
        '''
        if self.rank == 0:
            chn = batch['gt'].shape[1]
            num_timesteps = self.base_diffusion.num_timesteps
            record_steps = [1, (num_timesteps // 2) + 1, num_timesteps]
            if current_iters % self.configs.train.log_freq[0] == 1:
                self.loss_mean = {key:torch.zeros(size=(len(record_steps),), dtype=torch.float64)
                                  for key in loss.keys()}
                self.loss_count = torch.zeros(size=(len(record_steps),), dtype=torch.float64)
            
            for jj in range(len(record_steps)):
                for key, value in loss.items():
                    index = record_steps[jj] - 1
                    mask = torch.where(tt == index, torch.ones_like(tt), torch.zeros_like(tt))
                    current_loss = torch.sum(value.detach() * mask)
                    self.loss_mean[key][jj] += current_loss.item()
                self.loss_count[jj] += mask.sum().item()

            if current_iters % self.configs.train.log_freq[0] == 0:
                if torch.any(self.loss_count == 0):
                    self.loss_count += 1e-4
                for key in loss.keys():
                    self.loss_mean[key] /= self.loss_count
                log_str = 'Train: {:06d}/{:06d}, Loss/MSE: '.format(
                        current_iters,
                        self.configs.train.iterations)
                for jj, current_record in enumerate(record_steps):
                    log_str += 't({:d}):{:.1e}/{:.1e}, '.format(
                            current_record,
                            self.loss_mean['loss'][jj].item(),
                            self.loss_mean['mse'][jj].item(),
                            )
                log_str += 'lr:{:.2e}'.format(self.optimizer.param_groups[0]['lr'])
                self.logger.info(log_str)
                self.logging_metric(self.loss_mean, tag='Loss', phase=phase, add_global_step=True)
            if current_iters % self.configs.train.log_freq[1] == 0:
                self.logging_image(batch['lq'], tag='lq', phase=phase, add_global_step=False)
                self.logging_image(batch['gt'], tag='gt', phase=phase, add_global_step=False)
                
                inference_batch_size = self.configs.train.batch[1]
                

                z_t_scaled = self.base_diffusion._scale_input(z_t, tt)
                x_t_list = []
                for i in range(0, z_t_scaled.shape[0], inference_batch_size):
                    z_t_batch = z_t_scaled[i:i+inference_batch_size]
                    x_t_batch = self.base_diffusion.decode_first_stage(
                            z_t_batch,
                            self.autoencoder,
                            )
                    x_t_list.append(x_t_batch)
                x_t = torch.cat(x_t_list, dim=0)
                self.logging_image(x_t, tag='diffused', phase=phase, add_global_step=False)
                

                x0_pred_list = []
                for i in range(0, z0_pred.shape[0], inference_batch_size):
                    z0_pred_batch = z0_pred[i:i+inference_batch_size]
                    x0_pred_batch = self.base_diffusion.decode_first_stage(
                            z0_pred_batch,
                            self.autoencoder,
                            )
                    x0_pred_list.append(x0_pred_batch)
                x0_pred = torch.cat(x0_pred_list, dim=0)
                self.logging_image(x0_pred, tag='x0-pred', phase=phase, add_global_step=True)

            if current_iters % self.configs.train.save_freq == 1:
                self.tic = time.time()
            if current_iters % self.configs.train.save_freq == 0:
                self.toc = time.time()
                elaplsed = (self.toc - self.tic)
                self.logger.info(f"Elapsed time: {elaplsed:.2f}s")
                self.logger.info("="*100)

    @torch.inference_mode()
    def validation(self, phase='val'):
        if self.rank == 0:
            if self.configs.train.use_ema_val:
                self.reload_ema_model()
                self.ema_model.eval()
            else:
                self.model.eval()

            print(f"validation. current iters: {self.current_iters}")

            indices = np.linspace(
                    0,
                    self.base_diffusion.num_timesteps,
                    self.base_diffusion.num_timesteps if self.base_diffusion.num_timesteps < 5 else 4,
                    endpoint=False,
                    dtype=np.int64,
                    ).tolist()
            if not (self.base_diffusion.num_timesteps-1) in indices:
                indices.append(self.base_diffusion.num_timesteps-1)
            batch_size = self.configs.train.batch[1]
            num_iters_epoch = math.ceil(len(self.datasets[phase]) / batch_size)
            mean_psnr = mean_lpips = 0
            for ii, data in enumerate(self.dataloaders[phase]):
                torch.compiler.cudagraph_mark_step_begin()
                data = self.prepare_data(data, phase='val')
                if 'gt' in data:
                    im_lq, im_gt = data['lq'], data['gt']
                else:
                    im_lq = data['lq']
                num_iters = 0
                if self.configs.model.params.cond_lq:
                    model_kwargs = {'lq':data['lq'],}
                    if 'mask' in data:
                        model_kwargs['mask'] = data['mask']
                else:
                    model_kwargs = None
                tt = torch.tensor(
                        [self.base_diffusion.num_timesteps, ]*im_lq.shape[0],
                        dtype=torch.int64,
                        ).cuda()
                im_sr_progress_list = []
                for sample in self.base_diffusion.p_sample_loop_progressive(
                        y=im_lq,
                        model=self.ema_model.module if self.configs.train.use_ema_val else self.model,
                        first_stage_model=self.autoencoder,
                        noise=None,
                        clip_denoised=True if self.autoencoder is None else False,
                        model_kwargs=model_kwargs,
                        device=f"cuda:{self.rank}",
                        progress=False,

                        ):
                    sample_decode = {}
                    if num_iters in indices:
                        for key, value in sample.items():
                            if key in ['sample', ]:
                                sample_decode[key] = self.base_diffusion.decode_first_stage(
                                        value,
                                        self.autoencoder,
                                        ).clamp(-1.0, 1.0)
                        im_sr_progress = sample_decode['sample']
                        im_sr_progress_list.append(im_sr_progress)
                    num_iters += 1
                    tt -= 1
                
                im_sr_all = torch.cat(im_sr_progress_list, dim=1) # b, k*c, h, w
                val_sample = sample_decode['sample']

                if 'gt' in data:
                    mean_psnr += util_image.batch_PSNR(
                            val_sample * 0.5 + 0.5,
                            im_gt * 0.5 + 0.5,
                            ycbcr=self.configs.train.val_y_channel,
                            )
                    mean_lpips += self.lpips_loss(
                            val_sample,
                            im_gt,
                            ).sum().item()

                if (ii + 1) % self.configs.train.log_freq[2] == 0:
                    self.logger.info(f'Validation: {ii+1:02d}/{num_iters_epoch:02d}...')

                    im_sr_all = rearrange(im_sr_all, 'b (k c) h w -> (b k) c h w', c=im_lq.shape[1])
                    self.logging_image(
                            im_sr_all,
                            tag='progress',
                            phase=phase,
                            add_global_step=False,
                            nrow=len(indices),
                            )
                    if 'gt' in data:
                        self.logging_image(im_gt, tag='gt', phase=phase, add_global_step=False)
                    self.logging_image(im_lq, tag='lq', phase=phase, add_global_step=True)




            if 'gt' in data:
                mean_psnr /= len(self.datasets[phase])
                mean_lpips /= len(self.datasets[phase])
                self.logger.info(f'Validation Metric: PSNR={mean_psnr:5.2f}, LPIPS={mean_lpips:6.4f}...')
                self.logging_metric(mean_psnr, tag='PSNR', phase=phase, add_global_step=False)
                self.logging_metric(mean_lpips, tag='LPIPS', phase=phase, add_global_step=True)

            self.logger.info("="*100)

            if not (self.configs.train.use_ema_val and hasattr(self.configs.train, 'ema_rate')):
                self.model.train()

    


class TrainerDifIRLPIPS(TrainerDifIR):
    def backward_step(self, dif_loss_wrapper, micro_data, num_grad_accumulate, tt):
        loss_coef = self.configs.train.get('loss_coef')
        
        # diffusion loss
        if self.configs.train.use_amp:
            with autocast(device_type="cuda"):
                losses, z_t, z0_pred = dif_loss_wrapper()
                x0_pred = self.base_diffusion.decode_first_stage(
                        z0_pred,
                        self.autoencoder,
                        ) # f16
                self.current_x0_pred = x0_pred.detach()

                # lpips loss
                losses["lpips"] = self.lpips_loss(
                        x0_pred,
                        micro_data['gt'],
                        ).to(z0_pred.dtype).view(-1)
                flag_nan = torch.any(torch.isnan(losses["lpips"]))
                if flag_nan:
                    losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)
                losses["lpips"] *= loss_coef[1]

                if loss_coef[0] > 0:    # calculate mse in latent space
                    losses["mse"] *= loss_coef[0]
                else:                   # calculate mse in pixel space
                    assert loss_coef[2] > 0

                    # calculate mse in pixel space
                    x0_pred_pixel = x0_pred
                    gt_pixel = micro_data['gt']
                    losses["mse"] = F.mse_loss(x0_pred_pixel, gt_pixel, reduction='none').mean(dim=[1,2,3]).view(-1)
                    losses["mse"] *= loss_coef[2]
        else:
            losses, z_t, z0_pred = dif_loss_wrapper()
            x0_pred = self.base_diffusion.decode_first_stage(
                    z0_pred,
                    self.autoencoder,
                    ) # f16
            self.current_x0_pred = x0_pred.detach()

            # lpips loss
            losses["lpips"] = self.lpips_loss(
                    x0_pred,
                    micro_data['gt'],
                    ).to(z0_pred.dtype).view(-1)
            flag_nan = torch.any(torch.isnan(losses["lpips"]))
            if flag_nan:
                losses["lpips"] = torch.nan_to_num(losses["lpips"], nan=0.0)
            losses["lpips"] *= loss_coef[1]

            if loss_coef[0] > 0:    # calculate mse in latent space
                losses["mse"] *= loss_coef[0]
            else:                   # calculate mse in pixel space
                assert loss_coef[2] > 0

                # calculate mse in pixel space
                x0_pred_pixel = x0_pred
                gt_pixel = micro_data['gt']
                losses["mse"] = F.mse_loss(x0_pred_pixel, gt_pixel, reduction='none').mean(dim=[1,2,3]).view(-1)
                losses["mse"] *= loss_coef[2]

        losses['loss'] = sum(losses.values())
        loss = losses['loss'].mean() / num_grad_accumulate
        if self.amp_scaler is None:
            loss.backward()
            
        else:
            self.amp_scaler.scale(loss).backward()
            

        return losses, z0_pred, z_t

def replace_nan_in_batch(im_lq, im_gt):
    flag_nan = False
    if torch.any(torch.isnan(im_lq)):
        flag_nan = True
        im_lq = torch.nan_to_num(im_lq, nan=0.0)
    if torch.any(torch.isnan(im_gt)):
        flag_nan = True
        im_gt = torch.nan_to_num(im_gt, nan=0.0)
    return im_lq, im_gt, flag_nan