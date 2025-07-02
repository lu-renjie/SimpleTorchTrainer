from .logger import Logger
from .pbar import PrintProgress
from .mask import get_causal_mask, length_to_mask, merge_mask
from .lr_scheduler import LRScheduler


import functools
import traceback
def print_exception(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print('RANK:', args[0])
            traceback.print_exception(type(e), e, e.__traceback__)
        exit()
    return wrapper


import os
import torch
import random
import numpy as np
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # sometimes?


import socket
def get_available_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    return port


from torch.utils.data import DistributedSampler
class EndlessDataLoader:
    def __init__(self, dataloader):
        self.epoch = 0
        self.dataloader = dataloader
    
    def __iter__(self):
        while True:
            for batch in self.dataloader:
                yield batch
            self.epoch += 1
            if isinstance(self.dataloader.sampler, DistributedSampler):
                self.dataloader.sampler.set_epoch(self.epoch)
