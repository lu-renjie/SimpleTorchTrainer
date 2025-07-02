import json
import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F


class AgentBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_step = None
        self.tensorboard_writer = None

    def add_hook(self):
        """
        add hook for recording data in tensorboard

        remember to add hooks in self.hooks, otherwise the hook will be added every step, which is slow
        """
        self.hooks = []

    def remove_all_hook(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def set_global_step(self, global_step):
        self.global_step = global_step

    def set_tensorboard_writer(self, writer):
        self.tensorboard_writer = writer

    def log_tensorboard(self, type, key, value):
        assert type in ('image', 'scalar', 'histogram')
        if self.tensorboard_writer is None:
            return
        
        if type == 'image':
            self.tensorboard_writer.add_image(key, value, self.global_step)
        if type == 'scalar':
            self.tensorboard_writer.add_scalar(key, value, self.global_step)
        if type == 'histogram':
            self.tensorboard_writer.add_histogram(key, value, self.global_step)

    def forward(self, batch, mode):
        if mode == 'loss':
            return self.forward_loss(batch)
        elif mode == 'eval':
            return self.forward_eval(batch)
        raise ValueError()

    def forward_loss(self, batch):
        """
        Args:
            batch: data from dataloader

        Returns:
            loss: pytorch tensor
            log_log: pd.Series
        """
        raise NotImplementedError()

    @torch.no_grad()
    def forward_eval(self, batch):
        """
        Returns:
            dataframe, each row is a sample, each column is a metric
        """
        raise NotImplementedError()

    @torch.no_grad()
    def forward_inference(self, batch_img):
        """
        Args:
            batch_img: (B, C, H, W)
        Returns:
            probs: (B, class_num)
        """
        raise NotImplementedError()

    def save(self, log_dir):
        for name, model in self.named_children():
            path = osp.join(log_dir, f'{name}.pt')
            torch.save(model.state_dict(), path)

    def load(self, log_dir, strict=True):
        for name, model in self.named_children():
            path = osp.join(log_dir, f'{name}.pt')
            checkpoint = torch.load(path, map_location='cpu')
            model.load_state_dict(checkpoint, strict=strict)
            print(f'{name}.pt loaded')
