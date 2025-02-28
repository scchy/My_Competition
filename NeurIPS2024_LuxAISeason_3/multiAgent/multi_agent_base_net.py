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



class mActor(nn.Module):
    """
     n-Actor pi(a'|o'; theta)  
        gradient: log(pi(a'|o'; theta))*(r' + \gamma q_{t+1} - q)
    """
    def __init__(self, info_state_dim, action_dim):
        super(mActor, self).__init__()
        self.global_cnn = nn.Sequential(
            nn.Conv2d(4, 16, 6, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.actor_obs_cnn = nn.Sequential(
            nn.Conv2d(4, 16, 6, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor_obs_nn = nn.Sequential(
            nn.Linear(info_state_dim, 64),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 128, action_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # batch, channel, width, height
        # print(f'{x.shape=}')
        if len(x.shape) < 4:
            x = x.unsqueeze(axis=0)
        pic_f = x[:, :4, ...]
        unit_pic_f = x[:, 4:8, ...]
        state_info = x[:, 8, 0, :self.info_state_dim] 
        
        pic_o = self.global_cnn(pic_f)
        unit_pic_o = self.actor_obs_cnn(unit_pic_f)
        state_o = self.actor_obs_nn(state_info)
        o = torch.concat([pic_o, unit_pic_o, state_o], axis=1)
        return self.head(o)


class centerSingleQ(nn.Module):
    """
    n-Value q(O, A; w') 
    """
    def __init__(self, signle_info_state_dim, signle_action_dim, n):
        super(centerSingleQ, self).__init__()
        self.signle_info_state_dim = signle_info_state_dim
        self.signle_action_dim = signle_action_dim
        self.n = n
        self.global_cnn = nn.Sequential(
            nn.Conv2d(4, 16, 6, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.O_cnn = nn.ModuleList([
            self.agent_obs_cnn() for i in range(self.n)
        ]) # 128 * n
        self.O_nn = nn.ModuleList([
            self.agent_obs_nn() for i in range(self.n)
        ]) # 32 * n
        self.O_linear = nn.Sequential(
            nn.Linear(160 * self.n, 2048),
            nn.ReLU(),
            nn.Linear(2048, 256),
            nn.ReLU(),
        )
        self.A_nn = nn.Sequential(
            nn.Linear(self.signle_action_dim * self.n, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.q = nn.Linear(128 + 256 + 128, 1)

    def agent_obs_cnn(self):
        obs_cnn = nn.Sequential(
            nn.Conv2d(4, 16, 6, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Flatten()
        ) # 128
        return obs_cnn

    def agent_obs_nn(self):
        return nn.Sequential(
            nn.Linear(self.info_state_dim, 32),
            nn.ReLU()
        )
        
    def forward(self, obs_list, a_list):
        global_p = obs_list[0][:, :4, ...]
        O_cnn_l = []
        O_nn_l = []
        a_l = []
        global_o = self.global_cnn(global_p)
        for idx, (x, a) in enumerate(zip(obs_list, a_list)):
            act_pic = x[:, 4:8, ...]
            act_info = x[:, 8, 0, :self.info_state_dim] 
            O_cnn_l.append(
                self.O_cnn[idx](act_pic)
            )
            O_nn_l.append(
                self.O_nn[idx](act_info)
            )
            a_l.append(
                self.A_nn[idx](a)
            )

        unit_o_f = self.O_linear(torch.concat(O_cnn_l + O_nn_l, axis=1))
        unit_a_f = torch.concat(a_l, axis=1)
        return self.q(torch.concat([global_p, unit_o_f, unit_a_f], axis=1))



