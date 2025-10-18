#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2022-05-18 13:04:06

import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext

from datapipe.datasets import create_dataset

from utils import util_net
from utils import util_common
from utils import util_image

from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from ldm.modules.ema import EMA


def my_worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # setup seed
        self.setup_seed()

    def setup_dist(self):
        num_gpus = torch.cuda.device_count()

        if num_gpus > 1:
            # if mp.get_start_method(allow_none=True) is None:
            #     mp.set_start_method('spawn')
            rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(
                    timeout=datetime.timedelta(seconds=3600),
                    backend='nccl',
                    init_method='env://',
                    )

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # CUDNN settings for reproducibility
        torch.backends.cudnn.benchmark = not self.configs.train.get('deterministic', False)
        torch.backends.cudnn.deterministic = self.configs.train.get('deterministic', False)
        torch.set_float32_matmul_precision("high")

        

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # setting log counter
        if self.rank == 0:
            self.log_step = {phase: 1 for phase in ['train', 'val']}
            self.log_step_img = {phase: 1 for phase in ['train', 'val']}

        # text logging
        logtxet_path = save_dir / 'training.log'
        if self.rank == 0:
            if logtxet_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(logtxet_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # tensorboard logging
        log_dir = save_dir / 'tf_logs'
        self.tf_logging = self.configs.train.tf_logging
        if self.rank == 0 and self.tf_logging:
            if not log_dir.exists():
                log_dir.mkdir()
            self.writer = SummaryWriter(str(log_dir))

        # checkpoint saving
        ckpt_dir = save_dir / 'ckpts'
        self.ckpt_dir = ckpt_dir
        if self.rank == 0 and (not ckpt_dir.exists()):
            ckpt_dir.mkdir()
        if 'ema_rate' in self.configs.train:
            self.ema_rate = self.configs.train.ema_rate
            assert isinstance(self.ema_rate, float), "Ema rate must be a float number"
            ema_ckpt_dir = save_dir / 'ema_ckpts'
            self.ema_ckpt_dir = ema_ckpt_dir
            if self.rank == 0 and (not ema_ckpt_dir.exists()):
                ema_ckpt_dir.mkdir()

        # save images into local disk
        self.local_logging = self.configs.train.local_logging
        if self.rank == 0 and self.local_logging:
            image_dir = save_dir / 'images'
            if not image_dir.exists():
                (image_dir / 'train').mkdir(parents=True)
                (image_dir / 'val').mkdir(parents=True)
            self.image_dir = image_dir

        # logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def close_logger(self):
        if self.rank == 0 and self.tf_logging:
            self.writer.close()

    def resume_from_ckpt(self):

        if self.configs.resume:
            assert self.configs.resume.endswith(".pth") and os.path.isfile(self.configs.resume)

            if self.rank == 0:
                self.logger.info(f"=> Loaded checkpoint from {self.configs.resume}")
            ckpt = torch.load(self.configs.resume, map_location=f"cuda:{self.rank}")
            util_net.reload_model(self.model, ckpt['state_dict'])
            torch.cuda.empty_cache()

            # learning rate scheduler
            self.iters_start = ckpt['iters_start']
            for ii in range(1, self.iters_start+1):
                self.adjust_lr(ii)

            # logging
            if self.rank == 0:
                self.log_step = ckpt['log_step']
                self.log_step_img = ckpt['log_step_img']

            # EMA model
            if self.rank == 0 and hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / ("ema_"+Path(self.configs.resume).name)
                self.logger.info(f"=> Loaded EMA checkpoint from {str(ema_ckpt_path)}")
                ema_ckpt = torch.load(ema_ckpt_path, map_location=f"cuda:{self.rank}")
                self.ema_model.module.load_state_dict(ema_ckpt)
            torch.cuda.empty_cache()

            # AMP scaler
            if self.amp_scaler is not None:
                if "amp_scaler" in ckpt:
                    self.amp_scaler.load_state_dict(ckpt["amp_scaler"])
                    if self.rank == 0:
                        self.logger.info("Loading scaler from resumed state...")

            # reset the seed
            self.setup_seed(seed=self.iters_start)
        else:
            self.iters_start = 0

    def setup_optimizaton(self):
        try:
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.configs.train.lr,
                                               weight_decay=self.configs.train.weight_decay,
                                               fused=True)
        except TypeError:
            # fused is not supported in this version, fallback to foreach
            print(f"Fused AdamW is not supported, fallback to foreach implementation.")
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.configs.train.lr,
                                               weight_decay=self.configs.train.weight_decay,
                                               foreach=True)

        # amp settings
        self.amp_scaler = GradScaler() if self.configs.train.use_amp else None

    def build_model(self):
        params = self.configs.model.get('params', dict)
        model = util_common.get_obj_from_str(self.configs.model.target)(**params)
        model = model.to(memory_format=torch.channels_last)
        model.cuda()
        if self.configs.model.ckpt_path is not None:
            ckpt_path = self.configs.model.ckpt_path
            if self.rank == 0:
                self.logger.info(f"Initializing model from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
            if 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            util_net.reload_model(model, ckpt)
        if self.configs.train.compile.flag:
            if self.rank == 0:
                self.logger.info("Begin compiling model...")
            model = torch.compile(model, mode=self.configs.train.compile.mode)
            if self.rank == 0:
                self.logger.info("Compiling Done")
        if self.num_gpus > 1:
            self.model = DDP(model, device_ids=[torch.cuda.current_device()], static_graph=False)  # wrap the network
        else:
            self.model = model

        # EMA
        if self.rank == 0 and hasattr(self.configs.train, 'ema_rate'):
            self.ema_model = EMA(model, self.configs.train.ema_rate, ignore_keys=['running_', 'num_batches_tracked'])

        # model information
        self.print_model_info()

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                    datasets['train'],
                    num_replicas=self.num_gpus,
                    rank=self.rank,
                    )
        else:
            sampler = None
        num_workers = max(self.configs.train.num_workers, 4)
        loader_kwargs = dict(
            batch_size=self.configs.train.batch[0] // self.num_gpus,
            shuffle=False if self.num_gpus > 1 else True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=my_worker_init_fn,
            sampler=sampler,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.configs.train.get('prefetch_factor', 2)
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["pin_memory_device"] = f"cuda:{self.rank}"
        dataloaders = {'train': _wrap_loader(udata.DataLoader(datasets['train'], **loader_kwargs))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                 )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def print_model_info(self):
        if self.rank == 0:
            num_params = util_net.calculate_parameters(self.model) / 1000**2
            # self.logger.info("Detailed network architecture:")
            # self.logger.info(self.model.__repr__())
            self.logger.info(f"Number of parameters: {num_params:.2f}M")

    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        data = {key:value.cuda().to(dtype=dtype, memory_format=torch.channels_last) for key, value in data.items()}
        return data

    def validation(self):
        pass

    def train(self):
        self.init_logger()       # setup logger: self.logger

        # performance optimizations
        torch.set_float32_matmul_precision('high')

        self.build_model()       # build model: self.model, self.loss

        self.setup_optimizaton() # setup optimization: self.optimzer, self.sheduler

        self.resume_from_ckpt()  # resume if necessary

        self.build_dataloader()  # prepare data: self.dataloaders, self.datasets, self.sampler

        self.model.train()
        num_iters_epoch = math.ceil(len(self.datasets['train']) / self.configs.train.batch[0])

        # Prefetch the first batch of data
        raw_data = next(self.dataloaders['train'])
        with torch.cuda.stream(self.data_stream):
            prepared_data = self.prepare_data(raw_data)


        # torch.cuda.memory._record_memory_history()

        for ii in range(self.iters_start, self.configs.train.iterations):
            self.current_iters = ii + 1


            # Wait for the data from the previous iteration to be ready
            torch.cuda.current_stream().wait_stream(self.data_stream)
            current_prepared_data = prepared_data

            # Asynchronously prepare data for the next iteration
            if (ii+1) < self.configs.train.iterations:
                raw_data = next(self.dataloaders['train'])
                with torch.cuda.stream(self.data_stream):
                    prepared_data = self.prepare_data(raw_data)

            # training phase
            self.training_step(current_prepared_data)
            # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
            # if ii >= 2: break
            # validation phase
            if 'val' in self.dataloaders and (ii+1) % self.configs.train.get('val_freq', 10000) == 0:
                self.validation()

            #update learning rate
            self.adjust_lr()

            # save checkpoint
            if (ii+1) % self.configs.train.save_freq == 0:
                self.save_ckpt()

            if (ii+1) % num_iters_epoch == 0 and self.sampler is not None:
                self.sampler.set_epoch(ii+1)

        # close the tensorboard
        self.close_logger()

        # destroy process group
        if self.num_gpus > 1 and dist.is_initialized():
            dist.destroy_process_group()

    def training_step(self, data):
        pass

    def adjust_lr(self, current_iters=None):
        assert hasattr(self, 'lr_scheduler')
        self.lr_scheduler.step()

    def save_ckpt(self):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / 'model_{:d}.pth'.format(self.current_iters)
            ckpt = {
                    'iters_start': self.current_iters,
                    'log_step': {phase:self.log_step[phase] for phase in ['train', 'val']},
                    'log_step_img': {phase:self.log_step_img[phase] for phase in ['train', 'val']},
                    'state_dict': self.model.state_dict(),
                    }
            if self.amp_scaler is not None:
                ckpt['amp_scaler'] = self.amp_scaler.state_dict()
            torch.save(ckpt, ckpt_path)
            if hasattr(self, 'ema_rate'):
                ema_ckpt_path = self.ema_ckpt_dir / 'ema_model_{:d}.pth'.format(self.current_iters)
                torch.save(self.ema_model.module.state_dict(), ema_ckpt_path)

    def reload_ema_model(self):
        if self.rank == 0:
            # EMA model is now a wrapper, so we don't need to manually load state dict
            pass

    @torch.no_grad()
    def update_ema_model(self):
        if self.num_gpus > 1:
            dist.barrier()
        if self.rank == 0:
            self.ema_model.update(self.model)

    def logging_image(self, im_tensor, tag, phase, add_global_step=False, nrow=8):
        """
        Args:
            im_tensor: b x c x h x w tensor
            im_tag: str
            phase: 'train' or 'val'
            nrow: number of displays in each row
        """
        assert self.tf_logging or self.local_logging
        im_tensor = vutils.make_grid(im_tensor, nrow=nrow, normalize=True, scale_each=True) # c x H x W
        if self.local_logging:
            im_path = str(self.image_dir / phase / f"{tag}-{self.log_step_img[phase]}.png")
            im_np = im_tensor.cpu().permute(1,2,0).numpy()
            util_image.imwrite(im_np, im_path)
        if self.tf_logging:
            self.writer.add_image(
                    f"{phase}-{tag}-{self.log_step_img[phase]}",
                    im_tensor,
                    self.log_step_img[phase],
                    )
        if add_global_step:
            self.log_step_img[phase] += 1

    def logging_metric(self, metrics, tag, phase, add_global_step=False):
        """
        Args:
            metrics: dict
            tag: str
            phase: 'train' or 'val'
        """
        if self.tf_logging:
            tag = f"{phase}-{tag}"
            if isinstance(metrics, dict):
                self.writer.add_scalars(tag, metrics, self.log_step[phase])
            else:
                self.writer.add_scalar(tag, metrics, self.log_step[phase])
            if add_global_step:
                self.log_step[phase] += 1
        else:
            pass

    def freeze_model(self, net):
        for params in net.parameters():
            params.requires_grad = False

    def load_model(self, model, ckpt_path=None, tag='model', strict=True):
        if self.rank == 0:
            self.logger.info(f'Loading {tag} from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        if strict:
            util_net.reload_model(model, ckpt)
        else:
            model.load_state_dict(ckpt, strict=False)
        if self.rank == 0:
            self.logger.info('Loaded Done')