import json
import torch
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint

from torchvision.models import resnet18, vit_b_16
from torchvision.utils import make_grid

from model import ViT
from config import device

from .agent_base import AgentBase


class AgentClassification(AgentBase):
    """
    An example classification agent.
    
    Agent is used to encapsulate model for loss and metrics calculation.
    """

    def __init__(self, args):
        super().__init__()

        # self.model = resnet18(num_classes=10)
        # self.model = vit_b_16(num_classes=10, dropout=0.1, attention_dropout=0.1)
        self.model = ViT(class_num=10, dropout=0.1, attn_dropout=0.1, pre_norm=True)

    def add_hook(self):
        """
        add hook for recording data in tensorboard

        remember to add hooks in self.hooks, otherwise the hook will be added every step, which is slow
        """
        self.hooks = []

        # record attention weight
        for i in [0, 6, 11]:
            def hook(module, args, output, layer_id=i):
                _, attention_weight = output
                attention_weight = attention_weight.detach().cpu()  # (B, head_num, n, n)
                attention_weight = attention_weight[0, :, 0, 1:].reshape(12, 1, 14, 14)
                attention_weight = attention_weight.repeat(1, 3, 1, 1)  # (head_num, 3, h, w)
                attention_weight = make_grid(attention_weight, nrow=4, normalize=True, pad_value=1)
                self.log_tensorboard('image', f'attn/layer{layer_id}', attention_weight.numpy())
            handle = self.model.layers[i].attention.register_forward_hook(hook)
            self.hooks.append(handle)

        # record grad using hook
        def hook(grad):
            data = grad.detach().cpu().numpy()
            self.log_tensorboard('histogram', 'pos_embedding_grad', data)
        handle = self.model.position_embedding.register_hook(hook)
        self.hooks.append(handle)

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
        batch_img, batch_label = batch

        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)

        probs = self.model(batch_img)
        loss = F.cross_entropy(probs, batch_label)

        loss_log = pd.Series(dtype=np.float32)
        loss_log['ce_loss'] = loss.item()
        return loss, loss_log

    @torch.no_grad()
    def forward_eval(self, batch):
        """
        Returns:
            dataframe, each row is a sample, each column is a metric
        """
        batch_img, batch_label = batch

        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)

        probs = self.model(batch_img)
        accuracy = (probs.argmax(dim=1) == batch_label).float()

        log_dict = pd.DataFrame(dtype=np.float32)
        log_dict['accuracy'] = accuracy.cpu().numpy()
        return log_dict

    @torch.no_grad()
    def forward_inference(self, batch_img):
        """
        Args:
            batch_img: (B, C, H, W)
        Returns:
            probs: (B, class_num)
        """
        probs = self.model(batch_img)
        return probs

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
