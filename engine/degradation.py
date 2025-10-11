#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
import random
import math

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt


@torch.no_grad()
def prepare_data_realesrgan(data, configs, rank, replace_nan_in_batch_func):
    """Prepare Real-ESRGAN style degradation data"""
    if not hasattr(prepare_data_realesrgan, 'jpeger'):
        prepare_data_realesrgan.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
    if not hasattr(prepare_data_realesrgan, 'use_sharpener'):
        prepare_data_realesrgan.use_sharpener = USMSharp().cuda()

    im_gt = data['gt'].cuda()
    kernel1 = data['kernel1'].cuda()
    kernel2 = data['kernel2'].cuda()
    sinc_kernel = data['sinc_kernel'].cuda()

    ori_h, ori_w = im_gt.size()[2:4]
    if isinstance(configs.degradation.sf, int):
        sf = configs.degradation.sf
    else:
        assert len(configs.degradation.sf) == 2
        sf = random.uniform(*configs.degradation.sf)

    if configs.degradation.use_sharp:
        im_gt = prepare_data_realesrgan.use_sharpener(im_gt)

    # ----------------------- The first degradation process ----------------------- #
    # blur
    out = filter2D(im_gt, kernel1)
    # random resize
    updown_type = random.choices(
            ['up', 'down', 'keep'],
            configs.degradation['resize_prob'],
            )[0]
    if updown_type == 'up':
        scale = random.uniform(1, configs.degradation['resize_range'][1])
    elif updown_type == 'down':
        scale = random.uniform(configs.degradation['resize_range'][0], 1)
    else:
        scale = 1
    mode = random.choice(['area', 'bilinear', 'bicubic'])
    out = F.interpolate(out, scale_factor=scale, mode=mode)
    # add noise
    gray_noise_prob = configs.degradation['gray_noise_prob']
    if random.random() < configs.degradation['gaussian_noise_prob']:
        out = random_add_gaussian_noise_pt(
            out,
            sigma_range=configs.degradation['noise_range'],
            clip=True,
            rounds=False,
            gray_prob=gray_noise_prob,
            )
    else:
        out = random_add_poisson_noise_pt(
            out,
            scale_range=configs.degradation['poisson_scale_range'],
            gray_prob=gray_noise_prob,
            clip=True,
            rounds=False)
    # JPEG compression
    jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range'])
    out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
    out = prepare_data_realesrgan.jpeger(out, quality=jpeg_p)

    # ----------------------- The second degradation process ----------------------- #
    if random.random() < configs.degradation['second_order_prob']:
        # blur
        if random.random() < configs.degradation['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(
                ['up', 'down', 'keep'],
                configs.degradation['resize_prob2'],
                )[0]
        if updown_type == 'up':
            scale = random.uniform(1, configs.degradation['resize_range2'][1])
        elif updown_type == 'down':
            scale = random.uniform(configs.degradation['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(int(ori_h / sf * scale), int(ori_w / sf * scale)),
                mode=mode,
                )
        # add noise
        gray_noise_prob = configs.degradation['gray_noise_prob2']
        if random.random() < configs.degradation['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=configs.degradation['noise_range2'],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
                )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=configs.degradation['poisson_scale_range2'],
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
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = prepare_data_realesrgan.jpeger(out, quality=jpeg_p)
    else:
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*configs.degradation['jpeg_range2'])
        out = torch.clamp(out, 0, 1)
        out = prepare_data_realesrgan.jpeger(out, quality=jpeg_p)
        # resize back + the final sinc filter
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
                out,
                size=(ori_h // sf, ori_w // sf),
                mode=mode,
                )
        out = filter2D(out, sinc_kernel)

    # resize back
    if configs.degradation.resize_back:
        out = F.interpolate(out, size=(ori_h, ori_w), mode='bicubic')
        temp_sf = configs.degradation['sf']
    else:
        temp_sf = configs.degradation['sf']

    # clamp and round
    im_lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

    # random crop
    gt_size = configs.degradation['gt_size']
    im_gt, im_lq = paired_random_crop(im_gt, im_lq, gt_size, temp_sf)
    im_lq = (im_lq - 0.5) / 0.5  # [0, 1] to [-1, 1]
    im_gt = (im_gt - 0.5) / 0.5  # [0, 1] to [-1, 1]
    im_lq, im_gt, flag_nan = replace_nan_in_batch_func(im_lq, im_gt)
    
    return im_lq, im_gt, flag_nan


def replace_nan_in_batch(im_lq, im_gt):
    flag_nan = False
    if torch.any(torch.isnan(im_lq)):
        flag_nan = True
        im_lq = torch.nan_to_num(im_lq, nan=0.0)
    if torch.any(torch.isnan(im_gt)):
        flag_nan = True
        im_gt = torch.nan_to_num(im_gt, nan=0.0)
    return im_lq, im_gt, flag_nan


def prepare_data_val(data, configs, dtype=torch.float32):
    """Prepare validation data"""
    offset = configs.train.get('val_resolution', 256)
    for key, value in data.items():
        h, w = value.shape[2:]
        if h > offset and w > offset:
            h_end = int((h // offset) * offset)
            w_end = int((w // offset) * offset)
            data[key] = value[:, :, :h_end, :w_end]
        else:
            h_pad = math.ceil(h / offset) * offset - h
            w_pad = math.ceil(w / offset) * offset - w
            padding_mode = configs.train.get('val_padding_mode', 'reflect')
            data[key] = F.pad(value, pad=(0, w_pad, 0, h_pad), mode=padding_mode)
    return {key:value.cuda().to(dtype=dtype) for key, value in data.items()}