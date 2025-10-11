#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from copy import deepcopy


class EMA:
    def __init__(self, module, decay, ignore_keys=()):
        self.module = deepcopy(module).eval()
        for p in self.module.parameters(): 
            p.requires_grad_(False)
        self.decay = decay
        self.ignore = set(ignore_keys)

    def __getattr__(self, name):
        return getattr(self.module, name)

    @torch.no_grad()
    def update(self, src):
        ms, ss = self.module.state_dict(), src.state_dict()
        d = self.decay
        
        # Handle DDP wrapper: remove 'module.' prefix if present
        if any(k.startswith('module.') for k in ss.keys()):
            ss_processed = {}
            for k, v in ss.items():
                if k.startswith('module.'):
                    ss_processed[k[7:]] = v
                else:
                    ss_processed[k] = v
            ss = ss_processed
        
        for k in ms.keys():
            if any(x in k for x in self.ignore):
                if k in ss:
                    ms[k] = ss[k]
            else:
                if k in ss and ss[k].dtype.is_floating_point and ms[k].dtype.is_floating_point:
                    ms[k].lerp_(ss[k], 1.0 - d)
        self.module.load_state_dict(ms, strict=True)
