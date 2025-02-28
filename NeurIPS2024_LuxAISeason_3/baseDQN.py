# python3
# Author: Scc_hy
# Create Date: 2025-01-17
# Reference: https://www.kaggle.com/code/sangrampatil5150/nuralbrain-v0-5-model-train-and-win
#   todo: 
#           move_action, sap_action
# ===========================================================================================

import torch 
from torch import nn, optim
import numpy as np 
from collections import deque
import random 
from typing import List, AnyStr
import os


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


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_layers_dim, action_dim):
        super(QNet, self).__init__()
        self.features = nn.ModuleList()
        for idx, h in enumerate(hidden_layers_dim):
            self.features.append(
                nn.ModuleDict({
                    'linear': nn.Linear(hidden_layers_dim[idx-1] if idx else state_dim, h),
                    'linear_active': nn.ReLU(inplace=True)
                })
            )
        self.head = nn.Linear(hidden_layers_dim[-1], action_dim) 
    
    def forward(self, x):
        for layer in self.features:
            x = layer['linear_active'](layer['linear'](x))
        return self.head(x)


class DQN:
    def __init__(
        self, 
        player: AnyStr,
        env_cfg,
        state_dim: int,
        hidden_layers_dim: List,
        action_dim: int,
        max_len: int,
        learning_rate: float=0.0001,
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

        self.state_dim = state_dim
        self.hidden_layers_dim = hidden_layers_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_end = epsilon
        self.target_update_freq = target_update_freq
        self.dqn_type = dqn_type
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start if epsilon_start is not None else epsilon
        # print(f'{self.epsilon_start=}')
        self.epsilon_decay_factor = epsilon_decay_factor
        self.device = device
        # qNet
        self.q = QNet(state_dim, hidden_layers_dim, action_dim).to(self.device)
        self.target_q = QNet(state_dim, hidden_layers_dim, action_dim).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        # loss 
        self.cost_func = nn.MSELoss()
        # opt 
        self.opt = optim.Adam(self.q.parameters(), lr=self.learning_rate)
        self.dqn_type = dqn_type
        self.count = 1
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()        
        self.train()

    def reset(self):
        self.unit_explore_locations = dict()
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()    
    
    def train(self):
        self.training = True
        self.q.train()
        self.target_q.train()

    def eval(self):
        self.training = False
        self.q.eval()
        self.target_q.eval()

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
        for unit_id in available_units:
            unit_pos = unit_positions[unit_id]
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
            q_sa = self.q(torch.FloatTensor(state).to(self.device))
            act_np = q_sa.cpu().detach().numpy()
            act = np.argmax(act_np)
            if act == 5:  # Sap action
                # Find closest enemy unit
                # valid_targets = self._find_opp_units(obs)
                valid_targets = np.array(self._find_opp_units(obs, unit_pos))
                if len(valid_targets):
                    target_pos = valid_targets[0] # Choose first valid target
                    actions[unit_id] = [5, target_pos[0], target_pos[1]]
                else:
                    # act_bool = np.argsort(act_np) == self.action_dim - 2
                    # actions[unit_id] = [np.arange(self.action_dim)[act_bool][0], 0, 0]  # 采用次优动作
                    actions[unit_id] = [0, 0, 0] # 留在原地
                    # print("act == 5 ERROR")
            else:
                actions[unit_id] = [act, 0, 0]

        # if not self.random_flag:
        #     print(f"policy {actions=}")
        return actions

    def _find_opp_units(self, obs, unit_pos):
        opp_positions = obs['units']['position'][self.opp_team_id]
        opp_mask = obs['units_mask'][self.opp_team_id]
        valid_targets = []
        for opp_id, pos in enumerate(opp_positions):
            if (opp_mask[opp_id] and pos[0] != -1 
                and np.abs(pos[0] - unit_pos[0]) <= self.unit_sap_range
                and np.abs(pos[1] - unit_pos[1]) <= self.unit_sap_range
            ):
                valid_targets.append(pos - unit_pos)
        return valid_targets

    def _state_representation(self, unit_pos, unit_energy, relic_nodes, step, relic_mask, obs):
        if not relic_mask.any():
            closest_relic = np.array([-1, -1])
        else:
            visible_relics = relic_nodes[relic_mask]
            distances = np.linalg.norm(visible_relics - unit_pos, axis=1)
            closest_relic = visible_relics[np.argmin(distances)]
        
        valid_targets = self._find_opp_units(obs, unit_pos)
        # if len(valid_targets):
        #     enemy_dis = np.linalg.norm(valid_targets - unit_pos, axis=1)
        #     closest_enemy_pos = valid_targets[np.argmin(enemy_dis)]
        #     can_sap_num = ((np.abs(valid_targets - unit_pos) <= self.unit_sap_range).sum(axis=1) >= 2).sum()
        # else:
        #     can_sap_num = 0
        #     closest_enemy_pos = np.array([-1, -1])

        mach_num = step // (self.env_cfg['max_steps_in_match'] + 1) + 1
        state = np.concatenate([
            unit_pos,
            closest_relic,
            [len(valid_targets)],
            [unit_energy],
            [step/505.0],  # Normalize step
        ])
        return state

    def update(self, batch_size):
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
        self.opt.step()
        if self.count > 0 and self.count % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

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
            self.target_q.load_state_dict(torch.load(q_f, weights_only=True))
            self.q.load_state_dict(torch.load(q_f, weights_only=True))
        except Exception as e:
            self.target_q.load_state_dict(torch.load(q_f, map_location='cpu', weights_only=True))
            self.q.load_state_dict(torch.load(q_f, map_location='cpu', weights_only=True))

        self.q.to(self.device)
        self.target_q.to(self.device)
        self.opt = optim.Adam(self.q.parameters(), lr=self.learning_rate)
