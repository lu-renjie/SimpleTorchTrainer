import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader

import utils

collate_fn = None


class TrainerBase:
    """
    A simple trainer implements data parallel training.
    """

    def __init__(self, datasets, agent):
        self.datasets = datasets
        self.agent = agent

        # about logging
        self.log_dir = None
        self.writer = None  # tensorboard summary writer
        self.logger = utils.Logger(None)
    
        # about training
        self.optimizer = None
        self.lr_scheduler = None

        self.rank = 0
        self.world_size = 1
        self.distributed = False

        self.use_amp = False  # amp is not tested

    def set_use_amp(self):
        self.use_amp = True
        self.scaler = torch.GradScaler("cuda")

    def set_log_dir(self, log_dir):
        if self.rank != 0:
            return

        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.logger = utils.Logger(log_dir)

    def set_distributed(self, rank, world_size):
        assert self.optimizer is None, 'You must set distributed before set optimizer'

        self.distributed = True
        self.rank = rank
        self.world_size = world_size

        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", init_method=None, world_size=world_size, rank=rank)
        self.agent = DDP(self.agent, device_ids=[rank], find_unused_parameters=False)
        print(f'RANK {self.rank}: Setup distributed training.')

    def set_optimizer(self, optimizer_cls, **kwargs):
        """
        Args:
            optmizer_cls: for example, torch.optim.SGD
            kwargs: arguments of optimizer
        """
        assert self.optimizer is None, 'Optmizer is already set.'

        total_params = sum(p.numel() for p in self.agent.parameters())
        self.log(f'Total parameter num {total_params:,}')

        trainable_params = [p for p in self.agent.parameters() if p.requires_grad]
        trainable_num = sum(p.numel() for p in trainable_params)
        self.log(f'Training parameter num {trainable_num:,}')

        self.optimizer = optimizer_cls(self.agent.parameters(), **kwargs)

    def set_lrscheduler(self, max_step, warmup_ratio, fn):
        """
        Only support none,linear,cosine lr scheduer. If you want to use other lr scheduler, you
        need to change the code here. Remember to record params in training description.
        """
        assert self.optimizer is not None
        self.lr_scheduler = utils.LRScheduler(self.optimizer, max_step, warmup_ratio, fn)
        # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)

    def log(self, message, no_prefix=False):
        if self.rank != 0:
            return

        self.logger.log(message, no_prefix)

    def _get_gradient_norm(self):
        total_norm = 0
        for params in self.agent.parameters():
            param_norm = params.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5
        return total_norm

    def _optimize(self, loss, max_norm):
        """
        Args:
            loss: loss that need backward
            max_norm: used to clip grad norm

        Returns:
            gradient of all parameters
        """
        if self.use_amp:
            self.scaler.scale(loss).backward()
            grad_norm = self._get_gradient_norm()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=max_norm, error_if_nonfinite=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = self._get_gradient_norm()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=max_norm, error_if_nonfinite=True)
            self.optimizer.step()
        self.optimizer.zero_grad()
        return grad_norm

    def _get_dataloader(self, dataset, batch_size):
        num_workers = min(8, batch_size)
        sampler = DistributedSampler(dataset) if self.distributed else RandomSampler(dataset)

        loader = DataLoader(
            dataset, batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        return loader

    def train(self):
        raise NotImplementedError()
