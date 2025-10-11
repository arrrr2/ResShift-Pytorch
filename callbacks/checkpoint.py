#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from pathlib import Path


class CheckpointCallback:
    def __init__(self, ckpt_dir, ema_ckpt_dir=None):
        self.ckpt_dir = ckpt_dir
        self.ema_ckpt_dir = ema_ckpt_dir

    def save_checkpoint(self, model, optimizer, amp_scaler, current_iters, log_step, log_step_img, ema_model=None):
        """Save model checkpoint"""
        ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(current_iters)
        ckpt = {
                'iters_start': current_iters,
                'log_step': log_step,
                'log_step_img': log_step_img,
                'state_dict': model.state_dict(),
                }
        if amp_scaler is not None:
            ckpt['amp_scaler'] = amp_scaler.state_dict()
        torch.save(ckpt, ckpt_path)
        
        # Save EMA checkpoint if available
        if ema_model is not None and self.ema_ckpt_dir is not None:
            ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(current_iters)
            torch.save(ema_model.module.state_dict(), ema_ckpt_path)

    def load_checkpoint(self, ckpt_path, model, optimizer, amp_scaler, device, ema_model=None, ema_ckpt_dir=None):
        """Load model checkpoint"""
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        
        # Load optimizer state
        if optimizer is not None:
            # Note: In a real implementation, you would load optimizer state here
            pass
            
        # Load AMP scaler state
        if amp_scaler is not None and "amp_scaler" in ckpt:
            amp_scaler.load_state_dict(ckpt["amp_scaler"])
            
        # Load EMA state if available
        if ema_model is not None and ema_ckpt_dir is not None:
            ema_ckpt_path = ema_ckpt_dir / ("ema_"+Path(ckpt_path).name)
            if ema_ckpt_path.exists():
                ema_ckpt = torch.load(ema_ckpt_path, map_location=device)
                ema_model.module.load_state_dict(ema_ckpt)
        
        return ckpt.get('iters_start', 0), ckpt.get('log_step', {}), ckpt.get('log_step_img', {})