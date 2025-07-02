"""
A minimum implementation of transformer.
"""

import math
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, layer_num, hidden_dim=768, head_num=12, dropout=0, attn_dropout=0, pre_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, head_num, dropout, attn_dropout, pre_norm) for _ in range(layer_num)
        ])
    
    def forward(self, x, attn_mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, layer_num, hidden_dim=768, head_num=12, dropout=0, attn_dropout=0, pre_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, head_num, dropout, attn_dropout, pre_norm) for i in range(layer_num)
        ])
    
    def forward(
        self,
        x,
        encoder_hidden_state,
        self_attention_mask=None,
        cross_attention_mask=None
    ):
        for i, layer in enumerate(self.layers):
            x = layer(x, encoder_hidden_state, self_attention_mask, cross_attention_mask)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout, attn_dropout, pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm

        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.attention = MultiheadAttention(hidden_dim, head_num, attn_dropout)
        self.norm1 = nn.RMSNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.RMSNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (B, N, C)
            attention_mask: (N, N)
        Returns:
            (B, N, C)
        """
        if self.pre_norm:
            residual_x = self.norm1(x)
            residual_x, _ = self.attention(residual_x, residual_x, residual_x, attention_mask)
            y = x + self.dropout(residual_x)

            residual_y = self.mlp(self.norm2(y))
            out = y + self.dropout(residual_y)
        else:
            residual_x, _ = self.attention(x, x, x, attention_mask)
            y = self.norm1(x + self.dropout(residual_x))

            residual_y = self.mlp(y)
            out = self.norm2(y + self.dropout(residual_y))

        return out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, head_num, dropout, attn_dropout, pre_norm=False):
        super().__init__()
        self.pre_norm = pre_norm

        self.hidden_dim = hidden_dim
        self.head_num = head_num

        self.self_attention = MultiheadAttention(hidden_dim, head_num, attn_dropout)
        self.norm1 = nn.RMSNorm(hidden_dim)

        self.cross_attention = MultiheadAttention(hidden_dim, head_num, attn_dropout)
        self.norm2 = nn.RMSNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.RMSNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_hidden_state, self_attention_mask=None, cross_attention_mask=None):
        """
        Args:
            x: (B, N1, C)
            encoder_hidden_state: (B, N2, C)
            attention_mask: (N1, N2)
        Returns:
            (B, N, C)
        """
        if self.pre_norm:
            residual_x = self.norm1(x)
            residual_x, _ = self.self_attention(residual_x, residual_x, residual_x, self_attention_mask)
            y = x + self.dropout(residual_x)

            residual_y = self.norm2(y)
            residual_y, _ = self.cross_attention(residual_y, encoder_hidden_state, encoder_hidden_state, cross_attention_mask)
            z = y + self.dropout(residual_y)

            residual_z = self.mlp(self.norm3(z))
            out = z + self.dropout(residual_z)
        else:
            residual_x, _ = self.self_attention(x, x, x, self_attention_mask)
            y = self.norm1(x + self.dropout(residual_x))

            residual_y, _ = self.cross_attention(y, encoder_hidden_state, encoder_hidden_state, cross_attention_mask)
            z = self.norm2(y + self.dropout(residual_y))

            residual_z = self.mlp(z)
            out = self.norm3(z + self.dropout(residual_z))

        return out


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_dim, head_num, attn_dropout, strict_attention=False):
        assert hidden_dim % head_num == 0
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.qkv_dim = hidden_dim // head_num
        self.scale = math.sqrt(self.hidden_dim)
        self.strict_attention = strict_attention

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attention_mask=None):
        """
        Args:
            q: (B, N1, C)
            k: (B, N2, C)
            v: (B, N2, C)
            attention_mask: (B, N1, N2)
        Returns:
            result: (B, N1, C)
            attention_weight: (B, head_num, N1, N2)
        """
        B, N1, C = q.shape
        B, N2, C = k.shape

        q = self.q(q).reshape(B, N1, self.head_num, self.qkv_dim).permute(0, 2, 1, 3)
        k = self.k(k).reshape(B, N2, self.head_num, self.qkv_dim).permute(0, 2, 3, 1)
        attention_score = (q @ k) / self.scale

        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)  # (B, head, N1, N2)
            attention_score = attention_score.masked_fill(attention_mask == 0.0, float('-inf'))

        attention_weight = torch.softmax(attention_score, dim=-1)  # (B, head, N1, N2)
        attention_weight = self.attn_dropout(attention_weight)

        v = self.v(v).reshape(B, N2, self.head_num, self.qkv_dim).permute(0, 2, 1, 3)
        result = attention_weight @ v  # (B, head, N1, C)
        result = result.permute(0, 2, 1, 3).reshape(B, N1, C)

        result = self.out_fc(result)
        return result, attention_weight
