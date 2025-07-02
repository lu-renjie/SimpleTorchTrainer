"""
Create mask for transformer.
"""

import torch


def get_causal_mask(size):
    x = torch.ones(size, size)
    x = torch.triu(x, diagonal=1)
    mask = (x == 0)
    return mask


def length_to_mask(length, size=None, device=None):
    """
    Returns:
        A bool matric indicate which elements are pad values
    """
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (
        torch.arange(size, dtype=torch.int64, device=device).unsqueeze(0).repeat(batch_size, 1)
        >
        (torch.tensor(length, dtype=torch.int64, device=device) - 1).unsqueeze(1)
    )
    return mask


def merge_mask(attention_mask, padding_mask):
    """
    Args:
        attention_mask: (N1, N2)
        padding_mask: (B, N2)
    Returns:
        (B, N1, N2)
    """
    attention_mask = attention_mask.unsqueeze(0)
    padding_mask = ~padding_mask.unsqueeze(1)
    mask = attention_mask & padding_mask
    return mask


if __name__ == '__main__':
    key_padding_mask = length_to_mask([10, 8])
    mask = get_causal_mask(key_padding_mask.shape[1])
    print(mask.float())
    print(key_padding_mask.float())
    mask = merge_mask(mask, key_padding_mask).float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, 0.0)
    print(mask)
    print(torch.softmax(mask, dim=-1))
