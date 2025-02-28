# python3
# Author: Scc_hy
# Create Date: 2025-02-26
# ===========================================================================================

import torch 
from torch import nn, optim
import numpy as np 
from collections import deque
import random 
from typing import List, AnyStr
import os
import copy


def all_seed(seed=6666):
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    # python全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')



def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


class ReplayBuffer:
    def __init__(self, max_len: int, np_save: bool=False):
        self._buffer = deque(maxlen=max_len)
        self.np_save = np_save
    
    def add(self, state, action, reward, next_state, done):
        self._buffer.append( (state, action, reward, next_state, done) )
    
    def __len__(self):
        return len(self._buffer)

    def sample(self, batch_size: int) -> deque:
        sample = random.sample(self._buffer, batch_size)
        return sample
