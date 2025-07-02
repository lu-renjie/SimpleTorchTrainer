import math
import torch
import torch.nn as nn
from .transformer import TransformerEncoderBlock


class ViT(nn.Module):
    def __init__(self,
            img_size=224,
            layer_num=12,
            hidden_dim=768,
            head_num=12,
            patch_size=16,
            class_num=1000,
            dropout=0,
            attn_dropout=0,
            pre_norm=False):
        super().__init__()
        if type(img_size) == int:
            img_size = (img_size, img_size)
        if type(patch_size) == int:
            patch_size = (patch_size, patch_size)
        assert img_size[0] % patch_size[0] == 0
        assert img_size[1] % patch_size[1] == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.class_num = class_num
        self.token_num = 1 + (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embed = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, self.token_num, hidden_dim))
    
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, head_num, dropout, attn_dropout, pre_norm) for i in range(layer_num)
        ])
    
        self.fc = nn.Linear(hidden_dim, class_num)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.patch_embed.bias)

        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.position_embedding, std=0.02)

        nn.init.zeros_(self.fc.bias)

    def forward(self, image_batch):
        B, C, H, W = image_batch.shape
        assert (H, W) == self.img_size

        tokens = self.patch_embed(image_batch)  # (B, C, H', W')
        tokens = tokens.permute(0, 2, 3, 1).reshape(B, self.token_num - 1, self.hidden_dim)
        cls_token = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)
        
        tokens = tokens + self.position_embedding
        for i, module in enumerate(self.layers):
            tokens = module(tokens)
        
        cls = tokens[:, 0, :]
        logits = self.fc(cls)
        return logits
