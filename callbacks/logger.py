#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torchvision.utils as vutils
from utils import util_image


class LoggerCallback:
    def __init__(self, writer=None, image_dir=None):
        self.writer = writer
        self.image_dir = image_dir

    def log_image(self, im_tensor, tag, phase, log_step_img, nrow=8, add_global_step=False):
        """
        Log images to tensorboard and/or save to disk
        Args:
            im_tensor: b x c x h x w tensor
            tag: str
            phase: 'train' or 'val'
            log_step_img: dict tracking image logging steps
            nrow: number of displays in each row
            add_global_step: whether to increment the step counter
        """
        if self.writer is None and self.image_dir is None:
            return
            
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        
        # Save to disk if requested
        if self.image_dir is not None:
            im_path = str(self.image_dir / phase / f"{tag}-{log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
            
        # Log to tensorboard if available
        if self.writer is not None:
            self.writer.add_image(
                    f"{phase}-{tag}-{log_step_img[phase]}",
                    im_tensor,
                    log_step_img[phase],
                    )
                    
        if add_global_step:
            log_step_img[phase] += 1

    def log_metric(self, metrics, tag, phase, log_step, add_global_step=False):
        """
        Log metrics to tensorboard
        Args:
            metrics: dict or scalar
            tag: str
            phase: 'train' or 'val'
            log_step: dict tracking logging steps
            add_global_step: whether to increment the step counter
        """
        if self.writer is None:
            return
            
        tag = f"{phase}-{tag}"
        if isinstance(metrics, dict):
            self.writer.add_scalars(tag, metrics, log_step[phase])
        else:
            self.writer.add_scalar(tag, metrics, log_step[phase])
            
        if add_global_step:
            log_step[phase] += 1