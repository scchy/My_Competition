# python3
# Author: Scc_hy
# Create Date: 2025-02-26
# Reference: https://www.kaggle.com/code/sangrampatil5150/nuralbrain-v0-5-model-train-and-win
#  action adv: [stay, up, right, down, left, 
#               sap(上), sap(右上), sap(右), sap(右下), sap(下), sap(左下), sap(左), sap(左上)] ->
#   action_dim = 5+8=13 
#   action_out:
#       0~4: [x, 0, 0]
#       5:   [5, x, y]
# state:
#   global_map & near_space_map & unit_info
# Centralized training
# Decentralized Execution - actor-sharing parameter
#   n-Actor pi(a'|o'; theta)  log(pi(a'|o'; theta))*(r' + \gamma q_{t+1} - q)
#   n-Value q(O, A; w')        TDError = r' + \gamma q_{t+1} - q
#       O: o1, o2, o3, ..., on
#       A: a1, a2, a3, ..., an
# ===========================================================================================

import torch 
from torch import nn, optim
import numpy as np 
from collections import deque
import random 
from typing import List, AnyStr
import os
import copy
from multi_agent_utils import all_seed, direction_to, ReplayBuffer
from multi_agent_base_net import mActor, centerSingleQ




class multiA2C:
    def __init__(
        self, 
        player: AnyStr,
        env_cfg,
        state_dim: int,
        hidden_layers_dim: List,
        action_dim: int,
        max_len: int,
        actor_lr: float=0.0001,
        critic_lr: float=0.0001,
        gamma: float=0.99,
        epsilon: float=0.05, 
        target_update_freq: int=1,
        dqn_type: AnyStr='DQN',
        epsilon_start: float=None,
        epsilon_decay_factor: float=None,
        device: AnyStr='cuda',
        random_flag: bool=False,
        min_samples: int=10000
    ):
        # player
        self.random_flag = random_flag
        self.min_samples = min_samples
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.max_len = max_len
        self.buffer = ReplayBuffer(max_len=max_len)
        self.unit_sap_range = self.env_cfg['unit_sap_range']
        self.unit_sap_cost = self.env_cfg['unit_sap_cost']
        self.max_units = self.env_cfg['max_units']

        self.state_dim = state_dim
        self.hidden_layers_dim = hidden_layers_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.epsilon_end = epsilon
        self.target_update_freq = target_update_freq
        self.dqn_type = dqn_type
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start if epsilon_start is not None else epsilon
        # print(f'{self.epsilon_start=}')
        self.epsilon_decay_factor = epsilon_decay_factor
        self.device = device

        # loss 
        self.q_loss_fn = nn.MSELoss()
        # opt 
        self.opt = optim.Adam(self.q.parameters(), lr=self.learning_rate)
        self.dqn_type = dqn_type
        self.count = 1
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()    
        self.create_net()    

    def create_net(self):
        self.share_actor = mActor(self.state_dim, self.action_dim)
        self.center_qs = [
            centerSingleQ(self.state_dim, self.action_dim, self.max_units) 
            for i in range(self.max_units)
        ]
        self.actor_opt = optim.Adam(self.share_actor.parameters(), lr=self.actor_lr)
        self.q_opt = optim.Adam([
                {'params': q_net.parameters(), 'lr': self.critic_lr, "eps": 1e-5}
            for q_net in self.center_qs
        ])

    def reset(self):
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()    
    
    def train(self):
        self.training = True
        self.share_actor.train()
        for q_net in self.center_qs:
            q_net.train()

    def eval(self):
        self.training = False
        self.share_actor.eval()
        for q_net in self.center_qs:
            q_net.eval()

    def _epsilon_update(self):
        self.epsilon = self.epsilon * self.epsilon_decay_factor
        if self.epsilon > self.epsilon_end:
            return self.epsilon
        return self.epsilon_end

    @torch.no_grad()
    def policy(self, step, obs, remainingOverageTime: int = 60):
        tmp_random_flag = len(self.buffer) < self.min_samples
        idx = self.team_id
        unit_mask = np.array(obs['units_mask'][idx]) # 
        unit_positions = np.array(obs['units']['position'][idx])
        unit_energys = np.array(obs['units']['energy'][idx])
        relic_nodes = np.array(obs['relic_nodes'])
        relic_mask = np.array(obs['relic_nodes_mask'])
        obv_relic_node_positions = np.array(obs['relic_nodes'])
        obv_relic_node_mask = np.array(obs['relic_nodes_mask'])

        # 可行动units
        available_units = np.where(unit_mask)[0]
        # visible relic nodes
        visible_relic_node_ids = set(np.where(obv_relic_node_mask)[0])
        # action
        actions = np.zeros((self.env_cfg['max_units'], 3), dtype=int)
        # basic strategy here is simply to have some units randomly explore 
        #           and some units collecting as much energy as possible
        # and once a relic node is found, we send all units to move randomly around the first relic node to gain points
        # and information about where relic nodes are found are saved for the next match
        for id_ in visible_relic_node_ids:
            if id_ not in self.discovered_relic_nodes_ids:
                self.discovered_relic_nodes_ids.add(id_)
                self.relic_node_positions.append(obv_relic_node_positions[id_])

        # unit ids range from 0 to max_units - 1
        for unit_id, (msk, unit_pos) in enumerate(zip(unit_mask, unit_positions)):
            if not msk:
                continue 

            unit_energy = unit_energys[unit_id]
            state = self._state_representation(
                unit_pos,
                unit_energy,
                relic_nodes,
                step,
                relic_mask,
                obs
            )
            if tmp_random_flag or self.random_flag or (self.training and np.random.random() < self.epsilon):
                if len(self.relic_node_positions) > 0:
                # if len(visible_relic_node_ids) > 0:
                    nearest_relic_node_position = self.relic_node_positions[0]
                    manhattan_distance = abs(unit_pos[0] - nearest_relic_node_position[0]) + \
                        abs(unit_pos[1] - nearest_relic_node_position[1])
                    # if close to the relic node we want to move randomly around it 
                    # and hope to gain points
                    if manhattan_distance <= 4:
                        random_direction = np.random.randint(0, 5)
                        actions[unit_id] = [random_direction, 0, 0]
                    else:
                        actions[unit_id] = [direction_to(unit_pos, nearest_relic_node_position), 0, 0]
                else:
                    # pick a random location on the map for the unit to explore
                    if step % 20 == 0 or unit_id not in self.unit_explore_locations:
                        rand_loc = (np.random.randint(0, self.env_cfg["map_width"]), np.random.randint(0, self.env_cfg["map_height"]))
                        self.unit_explore_locations[unit_id] = rand_loc
                    actions[unit_id] = [direction_to(unit_pos, self.unit_explore_locations[unit_id]), 0, 0]
                continue 
            # DQN policy 
            q_sa = self.share_actor(torch.FloatTensor(state).to(self.device))
            act_np = q_sa.cpu().detach().numpy()
            act = np.argmax(act_np)
            if act >= 5:  # Sap action
                delta_pos = self.sap_target_pos(
                    unit_pos, act, 
                    wh_array=np.array([self.env_cfg['map_width']-1, self.env_cfg['map_height']-1]),
                    move_step=self.unit_sap_range // 2,
                    valid_targets=[] #self._find_opp_units(obs, unit_pos)
                )
                actions[unit_id] = [5, delta_pos[0], delta_pos[1]]
            else:
                actions[unit_id] = [act, 0, 0]

        return actions

    def sap_target_pos(self, unit_pos, act_num, wh_array=np.array([23, 23]), move_step=3, valid_targets=[]):
        # map_width: int = 24
        # map_height: int = 24
        min_bound = 0
        sap_move = {
            5: [0, move_step], # sap(上)
            6: [move_step, move_step], # sap(右上), 
            7: [move_step, 0], # sap(右), 
            8: [move_step, -move_step], # sap(右下), 
            9: [0, -move_step], # sap(下), 
            10: [-move_step, -move_step], # sap(左下), 
            11: [-move_step, 0], # sap(左), 
            12: [-move_step, move_step], # sap(左上)
        }
        m = np.array(sap_move[act_num])
        return m
        # need_t_idx = -1
        # min_d = np.inf
        # for i, t in enumerate(valid_targets):
        #     if t[0] == act_num:
        #         if t[2] < min_d:
        #             min_d = t[2]
        #             need_t_idx = i
        # if need_t_idx < 0:
        #     return m
        # tar_arr = np.array(valid_targets[need_t_idx][1])
        # return tar_arr - unit_pos
        # return np.clip(unit_pos + m, a_min=np.array([0, 0]), a_max=wh_array)

    def _find_opp_units(self, obs, unit_pos):
        opp_positions = obs['units']['position'][self.opp_team_id]
        opp_mask = obs['units_mask'][self.opp_team_id]
        valid_targets = []
        for opp_id, pos in enumerate(opp_positions):
            if (opp_mask[opp_id] and pos[0] != -1 
                and np.abs(pos[0] - unit_pos[0]) <= self.unit_sap_range
                and np.abs(pos[1] - unit_pos[1]) <= self.unit_sap_range
                ):
                manhattan_distance = np.abs(pos[0] - unit_pos[0]) + np.abs(pos[1] - unit_pos[1])
                valid_targets.append([-1, pos, manhattan_distance])
                # 位置
                # if pos[0] - unit_pos[0] < 0: # 
                #     if pos[1] - unit_pos[1] < 0: 
                #         valid_targets.append([10, pos, manhattan_distance]) # 左下
                #         continue 
                #     if pos[1] - unit_pos[1] > 0: 
                #         valid_targets.append([12, pos, manhattan_distance]) #  左上
                #         continue 
                #     valid_targets.append([11, pos, manhattan_distance]) # 左
                #     continue
                # elif pos[0] - unit_pos[0] == 0:
                #     if pos[1] - unit_pos[1] < 0: 
                #         valid_targets.append([9, pos, manhattan_distance]) # 下
                #         continue 
                #     if pos[1] - unit_pos[1] > 0: 
                #         valid_targets.append([5, pos, manhattan_distance]) #  上
                #         continue 
                # else:# 右
                #     if pos[1] - unit_pos[1] < 0: 
                #         valid_targets.append([8, pos, manhattan_distance])  # 右下
                #         continue 
                #     if pos[1] - unit_pos[1] > 0: 
                #         valid_targets.append([6, pos, manhattan_distance]) #  右上
                #         continue 
                #     valid_targets.append([7, pos, manhattan_distance]) # 右

        return valid_targets

    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask, obs):
        if self.random_flag:
            return None
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]

        enemy_nums = len(self._find_opp_units(obs, unit_pos))
        info_state = np.concatenate([
            unit_pos, # 2
            [enemy_nums],
            closest_relic, # 2
            [unit_energy], # 1
            [unit_energy - self.unit_sap_cost], # 1
            [step/505.0],  # 1 Normalize step 
        ])

        info_state_map = np.zeros((self.env_cfg['map_width'], self.env_cfg['map_height']), dtype=np.float32)
        info_state_map[0, :self.state_dim] = info_state
        
        unit_mask = np.array(obs['units_mask'][self.team_id]) 
        available_units = np.where(unit_mask)[0]
        unit_positions = np.array(obs['units']['position'][self.team_id])
        obs_pic = self.create_map(obs, available_units, unit_positions)
        unit_pic = self.get_unit_local_pic(unit_pos, obs_pic)
        
        # print(f'{obs.shape=}{info_state_map.shape=}')
        state_final = np.concatenate([obs_pic, unit_pic, info_state_map[np.newaxis, :] ], axis=0)
        return state_final # [5, 24, 24]
    
    def get_unit_local_pic(self, unit_pos, pic):
        d_ = self.unit_sap_range + 1
        min_x = min(unit_pos[0] - d_, 0)
        max_x = min(unit_pos[0] + d_, self.env_cfg['map_width'])
        min_y = min(unit_pos[1] - d_, 0)
        max_y = min(unit_pos[1] + d_, self.env_cfg['map_height'])
        pic_mark = np.zeros_like(pic)
        # c, w, h
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                pic_mark[:, x, y] = 1

        pic_new = copy.deepcopy(pic)
        pic_new[pic_mark == 0] = -9
        return pic_new
    
    def create_obv_relic_map(self):
        obv_relic_map = np.zeros((self.env_cfg['map_width'], self.env_cfg['map_height']), dtype=np.int8)
        for p in self.relic_node_positions:
            obv_relic_map[p[0], p[1]] = 1    
        return obv_relic_map

    def create_map(self, obs, available_units, unit_positions):
        """
        tile_map + obs_relic + agent_map + energy
        """
        energy = obs['map_features']['energy']
        tile_map = obs['map_features']['tile_type']
        opp_positions = obs['units']['position'][self.opp_team_id]
        opp_mask = obs['units_mask'][self.opp_team_id]
        obs_relic = self.create_obv_relic_map()
        
        agent_map = np.zeros((self.env_cfg['map_width'], self.env_cfg['map_height']), dtype=np.int8)
        for unit_id in available_units:
            unit_pos = unit_positions[unit_id]
            x, y = unit_pos
            agent_map[x, y] = 1

        # 增加敌军信息
        for opp_id, pos in enumerate(opp_positions):
            if opp_mask[opp_id] and pos[0] != -1:
                agent_map[pos[0], pos[1]] -= -1

        obs = np.stack([tile_map, obs_relic, agent_map, energy], axis=0) # channel first
        return obs
    
    def update(self, batch_size):
        if self.random_flag:
            return 
        if len(self.buffer) < batch_size:
            return 
        if self.training and self.epsilon_start is not None:
            self.epsilon = self._epsilon_update()
        self.count += 1
        samples = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = zip(*samples)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.Tensor(np.array(actions)).view(-1, 1).to(self.device)
        rewards = torch.Tensor(np.array(rewards)).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        done = torch.Tensor(np.array(done)).view(-1, 1).to(self.device)

        ut = self.q(states).gather(1, actions.long())
        Q_sa1 = self.target_q(next_states)
        # 下一状态最大值
        if 'DoubleDQN' in self.dqn_type:
            # a* = argmax Q(s_{t+1}, a; w)
            a_star = self.q(next_states).max(1)[1].view(-1, 1)
            # doubleDQN Q(s_{t+1}, a*; w')
            ut_1 = Q_sa1.gather(1, a_star)
        else:
            # simple method:  avoid bootstrapping 
            ut_1 = Q_sa1.max(1)[0].view(-1, 1)
        
        q_tar = rewards + self.gamma * ut_1 * (1 - done)
        # update
        self.opt.zero_grad()
        loss = self.cost_func(ut.float(), q_tar.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 0.5) 
        self.opt.step()
        if self.count > 0 and self.count % self.target_update_freq == 0:
            self.soft_update(self.q, self.target_q)

    def soft_update(self, net, target_net):
        self.tau = 0.01
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(
                param_target.data * (1 - self.tau) + param.data * self.tau
            )

    def save_model(self, file_path, player=None):
        pl = self.player if player is None else player
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        q_f = os.path.join(file_path, f'{self.dqn_type}_model_{pl}_rd_{self.random_flag}.pth')
        torch.save(self.q.state_dict(), q_f)

    def load_model(self, file_path, player=None):
        pl = self.player if player is None else player
        q_f = os.path.join(file_path, f'{self.dqn_type}_model_{pl}_rd_{self.random_flag}.pth')
        # print(f'load_model -> {q_f}')
        try: 
            self.target_q.load_state_dict(torch.load(q_f))
            self.q.load_state_dict(torch.load(q_f))
        except Exception as e:
            self.target_q.load_state_dict(torch.load(q_f, map_location='cpu'))
            self.q.load_state_dict(torch.load(q_f, map_location='cpu'))

        self.q.to(self.device)
        self.target_q.to(self.device)
        self.opt = optim.Adam(self.q.parameters(), lr=self.learning_rate)
