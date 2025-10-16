#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from copy import deepcopy


# class EMA:
#     def __init__(self, module, decay, ignore_keys=()):
#         self.module = deepcopy(module).eval()
#         for p in self.module.parameters(): 
#             p.requires_grad_(False)
#         self.decay = decay
#         self.ignore = set(ignore_keys)

#     def __getattr__(self, name):
#         return getattr(self.module, name)

#     @torch.no_grad()
#     def update(self, src):
#         ms, ss = self.module.state_dict(), src.state_dict()
#         d = self.decay
        
#         # Handle DDP wrapper: remove 'module.' prefix if present
#         if any(k.startswith('module.') for k in ss.keys()):
#             ss_processed = {}
#             for k, v in ss.items():
#                 if k.startswith('module.'):
#                     ss_processed[k[7:]] = v
#                 else:
#                     ss_processed[k] = v
#             ss = ss_processed
        
#         for k in ms.keys():
#             if any(x in k for x in self.ignore):
#                 if k in ss:
#                     ms[k] = ss[k]
#             else:
#                 if k in ss and ss[k].dtype.is_floating_point and ms[k].dtype.is_floating_point:
#                     ms[k].lerp_(ss[k], 1.0 - d)
#         self.module.load_state_dict(ms, strict=True)
class EMA: # EMA_v2
    def __init__(self, module, decay, ignore_keys=()):
        self.module = deepcopy(module).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        
        # 优化：预先处理 ignore_keys，方便后续检查
        self.ignore_keys = set(ignore_keys)
        
        # 存储 EMA 模型的 named_parameters 和 named_buffers 以便快速访问
        self.ema_state = {name: p for name, p in self.module.named_parameters()}
        self.ema_state.update({name: b for name, b in self.module.named_buffers()})

    def __getattr__(self, name):
        return getattr(self.module, name)
    
    @torch.no_grad()
    def update(self, src):
        is_ddp = isinstance(src, nn.parallel.DistributedDataParallel)
        for name, src_tensor in src.named_parameters():
            self._update_tensor(name, src_tensor, is_ddp)
        for name, src_tensor in src.named_buffers():
            self._update_tensor(name, src_tensor, is_ddp)

    def _update_tensor(self, name, src_tensor, is_ddp):
        if is_ddp:
            name = name.replace('module.', '', 1)
        if name not in self.ema_state:
            return
        ema_tensor = self.ema_state[name]
        if any(key in name for key in self.ignore_keys):
            ema_tensor.copy_(src_tensor)
            return
        if not ema_tensor.dtype.is_floating_point:
            ema_tensor.copy_(src_tensor)
            return
        ema_tensor.mul_(self.decay).add_(src_tensor, alpha=1.0 - self.decay)