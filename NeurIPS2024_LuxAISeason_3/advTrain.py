# python3
# Author: Scc_hy
# Create Date: 2025-01-17
# Reference: https://www.kaggle.com/code/sangrampatil5150/nuralbrain-v0-5-model-train-and-win
# ===========================================================================================
import os 
import copy
import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from collections import deque
from argparse import Namespace
from advDQN import all_seed, DQN
import gymnasium as gym
from luxai_s3.wrappers import LuxAIS3GymEnv


class unitReward(gym.Wrapper):
    def __init__(self, env, add_pre_total=True): #False):
        gym.Wrapper.__init__(self, env)
        self.env_params = env.env_params
        self.pre_obs = None
        self.add_pre_total = add_pre_total
        self.score = 0
        self.pre_unit_reward = {
            'player_0': np.array([0]*self.env_params.max_units),
            'player_1': np.array([0]*self.env_params.max_units),
        }
        self.pre_total_reward = {
            'player_0': 0,
            'player_1': 0,
        }
    
    def action2delta_xy(self, a):
        # 0 
        delta_xy_l = [
                [0, 0],  # 0 Do nothing
                [0, -1],  # 1 Move up
                [1, 0],  # 2 Move right
                [0, 1],  # 3 Move down
                [-1, 0],  # 4 Move left
                [0, 0] # 5 sap
        ]
        return delta_xy_l[a]

    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['player_0_units_reward'] = [0] * self.env_params.max_units
        info['player_1_units_reward'] = [0] * self.env_params.max_units
        self.pre_obs = obs
        self.score = 0
        self.pre_unit_reward = {
            'player_0': np.array([0]*self.env_params.max_units),
            'player_1': np.array([0]*self.env_params.max_units),
        }
        self.pre_total_reward = {
            'player_0': 0,
            'player_1': 0,
        }
        return obs, info

    def step(self, action):
        """
        reference: https://www.kaggle.com/code/yizhewang3/ppo-stable-baselines3#train.py
        根据动作更新环境状态，并返回 (observation, reward, done, info)。
        修改后的奖励逻辑：
          1. 每个 unit 单独计算 unit_reward。
          2. 若移动动作导致超出地图或目标 tile 为 Asteroid，则判定为无效，unit_reward -0.2。
          3. Sap 动作：
             - 检查 unit 局部 obs 中 relic_nodes_mask 是否存在 relic；
             - 如果存在，统计 unit 8 邻域内敌方单位数，若数目>=2，则 sap 奖励 = +1.0×敌方单位数，否则扣 -2.0；
             - 若无 relic 可见，则同样扣 -2.0。
          4. 非 sap 动作：
             - 成功移动后，检查该 unit 是否位于任一 relic 配置内的潜力点：
                  * 若首次访问该潜力点，unit_reward +2.0，并标记 visited；
                  * 如果该潜力点尚未兑现 team point，则增加 self.score 1，同时 unit_reward +5.0 并标记为 team_points_space；
                  * 如果已在 team_points_space 上，则每回合奖励 +5.0；
             - 若 unit 位于能量节点（energy == Global.MAX_ENERGY_PER_TILE），unit_reward +0.2；
             - 若 unit 位于 Nebula（tile_type==1），unit_reward -0.2；
             - 如果 unit 移动后与敌方 unit 重合，且对方能量低于己方，则对每个满足条件的敌方 unit 奖励 +1.0。
          5. 全局探索奖励：所有己方单位联合视野中新发现 tile，每个奖励 +0.1。
          6. 每一step结束，奖励 point*0.3的奖励 + 规则*0.7的奖励
        """
        # compute reward 
        player_0_units_reward = self._compute_score(action['player_0'], 'player_0', 'player_1')
        player_1_units_reward = self._compute_score(action['player_1'], 'player_1', 'player_0')
        
        obs, reward, terminated, truncated, info = self.env.step(action)

        info['player_0_units_reward'] = np.array(player_0_units_reward) - 0.1 * self.pre_unit_reward['player_0']
        info['player_1_units_reward'] = np.array(player_1_units_reward) - 0.1 * self.pre_unit_reward['player_1']
        if self.add_pre_total:
            p0_tt_inc = obs["player_0"]["team_points"][0] - self.pre_total_reward["player_0"]
            p1_tt_inc = obs["player_1"]["team_points"][0] - self.pre_total_reward["player_1"]
            info['player_0_units_reward'] = info['player_0_units_reward']  + 0.2 * p0_tt_inc
            info['player_1_units_reward'] = info['player_1_units_reward']  + 0.2 * p1_tt_inc
        
        self.pre_obs = obs
        self.pre_unit_reward['player_0'] = np.array(player_0_units_reward)
        self.pre_unit_reward['player_1'] = np.array(player_1_units_reward)
        self.pre_total_reward = {
            "player_0": obs["player_0"]["team_points"][0],
            "player_1": obs["player_1"]["team_points"][1]
        }  
        return obs, reward, terminated, truncated, info

    def _find_opp_units(self, unit_pos, team, delta_pos):
        opp_team = 1 if team == "player_0" else 0
        opp_positions = self.pre_obs[team]['units']['position'][opp_team]
        opp_mask = self.pre_obs[team]['units_mask'][opp_team]
        valid_targets = []
        target_p = np.array(unit_pos) + np.array(delta_pos)
        for opp_id, pos in enumerate(opp_positions):
            if (opp_mask[opp_id] and pos[0] != -1 
                and np.abs(pos[0] - unit_pos[0]) <= self.env_params.unit_sap_range
                and np.abs(pos[1] - unit_pos[1]) <= self.env_params.unit_sap_range
                ):
                # manhattan_distance = np.abs(pos[0] - unit_pos[0]) + np.abs(pos[1] - unit_pos[1])
                manhattan_distance = np.abs(pos[0] - target_p[0]) + np.abs(pos[1] - target_p[1])
                valid_targets.append([-1, pos, manhattan_distance])
        return valid_targets
    
    def _compute_score(self, actions, player, opp_payer):
        unit_obs = self.pre_obs[player]
        unit_reward_l = []
        idx = 0 if player == 'player_0' else 1
        unit_mask = np.array(unit_obs['units_mask'][idx]) # 
        unit_positions = np.array(unit_obs['units']['position'][idx])
        available_units = np.where(unit_mask)[0]
        tile_map = unit_obs['map_features']['tile_type']

        obv_relic_node_positions = np.array(unit_obs['relic_nodes'])
        obv_relic_node_mask = np.array(unit_obs['relic_nodes_mask'])

        for unit_id, msk in enumerate(unit_mask):
            unit_reward = 0.0
            if not msk:
                unit_reward_l.append(unit_reward)
                continue 
            unit_pos = unit_positions[unit_id]
            act = actions[unit_id, :]
            delta_pos = act[1:]
            action_type = act[0]
            enemy_details = self._find_opp_units(unit_pos, player, delta_pos)
            enemy_count = len(enemy_details)
            # 如果动作为 sap
            if action_type == 5:
                if enemy_count >= 1 and any(i[-1] <= 3 for i in enemy_details):
                    unit_reward += enemy_count * 2
                elif  enemy_count >= 1: # 20250227 
                    unit_reward -= 1.0
                else:
                    unit_reward -= 2.0
            elif action_type == 0:
                unit_reward -= 0.1 # do nothing neg reward
            elif action_type != 5 and enemy_count >= 2:
                unit_reward -= 0.5 # enemy near but do nothing 
            else:
                dx, dy = self.action2delta_xy(action_type)
                new_x = unit_pos[0] + dx
                new_y = unit_pos[1] + dy
                if (
                    new_x < 0 or new_x >= self.env_params.map_width or 
                    new_y < 0 or new_y >= self.env_params.map_height):
                    unit_reward -= 0.2
                    unit_reward_l.append(unit_reward)
                    continue
                # 检查边界和障碍
                if tile_map[new_x, new_y] == 2:
                    unit_reward -= 0.2  # 遇到 Asteroid

                # 检查 relic 配置奖励：遍历所有 relic 配置，判断该 unit 是否位于配置中（计算时考虑边界）
                for (rx, ry), mask in zip(obv_relic_node_positions, obv_relic_node_mask):

                    if mask and rx - 2 <= new_x <= rx + 2 and ry - 2 <= new_y <= ry + 2 :
                        unit_reward += 5.0
                # 能量节点奖励 unit_obs update
                if unit_obs["map_features"]["energy"][new_x, new_y] == self.env_params.max_energy_per_tile:
                    unit_reward += 0.2
                # Nebula 惩罚
                if unit_obs["map_features"]["tile_type"][new_x, new_y] == 1:
                    unit_reward -= 0.2

            unit_reward_l.append(unit_reward)

        return  unit_reward_l  


def train_off_policy(
        env, 
        player_0, 
        player_1,
        cfg,
        wandb_flag=False,
        wandb_project_name="LuxAI",
    ):
    
    obs, info = env.reset(seed=cfg.seed)
    env_cfg = info["params"]   # UNIT_SAP_RANGE
    if wandb_flag:
        wandb.login()
        cfg_dict = cfg.__dict__ 
        algo = player_0.__class__.__name__
        now_ = datetime.now().strftime('%Y%m%d__%H%M')
        wandb.init(
            project=wandb_project_name,
            name= f"{algo}__LuxAI__{now_}",
            config=cfg_dict,
            monitor_gym=True
        )

    tq_bar = tqdm(range(cfg.num_episode))
    final_seed = cfg.seed
    palyers_rewards_list = {
        'player_0': deque(maxlen=10),
        'player_1': deque(maxlen=10),
    }
    now_reward= {
        'player_0': -np.inf,
        'player_1': -np.inf
    }
    palyers_win_list = {
        'player_0': deque(maxlen=10),
        'player_1': deque(maxlen=10),
    }
    now_win_rate = {
        'player_0': 0,
        'player_1': 0
    }
    best_test_r0 = -np.inf
    best_test_r1 = -np.inf
    for i in tq_bar:
        if (i > 25) and i % 10 == 0:
            plyer_test_r0, plyer_test_r1 = env_test(env, player_0, player_1, cfg, test_nums=5, focus_player=None, print_flag=False)
            if not player_0.random_flag and plyer_test_r0 >= best_test_r0:
                best_test_r0 = plyer_test_r0
                player_0.save_model(cfg.save_path + '_bst')
            if not player_1.random_flag and plyer_test_r1 >= best_test_r1:
                best_test_r1 = plyer_test_r1
                player_1.save_model(cfg.save_path + '_bst')

        obs, info = env.reset(seed=final_seed)
        # dqn collection reset 
        player_0.train()
        player_1.train()
        player_0.reset()
        player_1.reset()

        done = False 
        step = 0
        last_obs = None
        last_actions = None
        tq_bar.set_description(f'Episode [ {i+1} / {cfg.num_episode}|(seed={final_seed}) ]')
        episode_rewards = {
            'player_0': 0,
            'player_1': 0,
        }
        while not done:
            actions = {}
            # Store current observation for learning
            last_obs = {
                "player_0": obs["player_0"].copy(),
                "player_1": obs["player_1"].copy()
            }
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.policy(step=step, obs=obs[agent.player])
            
            last_actions = copy.deepcopy(actions)
            # Environment step
            # print(f'{actions=}')
            obs, rewards ,terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }  
            unit_rewards_map = {
                "player_0": info['player_0_units_reward'],
                "player_1": info['player_1_units_reward'],
            }
            if last_obs is not None:
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._state_representation(
                                last_obs[agent.player]['units']['position'][agent.team_id][unit_id],
                                last_obs[agent.player]['units']['energy'][agent.team_id][unit_id],
                                last_obs[agent.player]['relic_nodes'],
                                step,
                                last_obs[agent.player]['relic_nodes_mask'],
                                last_obs[agent.player]
                            )
                            next_state = agent._state_representation(
                                obs[agent.player]['units']['position'][agent.team_id][unit_id],
                                obs[agent.player]['units']['energy'][agent.team_id][unit_id],
                                obs[agent.player]['relic_nodes'],
                                step + 1,
                                obs[agent.player]['relic_nodes_mask'],
                                obs[agent.player]
                            )
                            agent.buffer.add(
                                current_state,
                                last_actions[agent.player][unit_id][0],
                                0.35 * rewards[agent.player] + 0.65 * unit_rewards_map[agent.player][unit_id],
                                next_state,
                                dones[agent.player]
                            )
                            
                            episode_rewards[agent.player] += rewards[agent.player]

                player_0.update(cfg.batch_size)
                player_1.update(cfg.batch_size)

            if dones["player_0"] or dones["player_1"]:
                done = True
                player_0.save_model(cfg.save_path)
                player_1.save_model(cfg.save_path)
                p0_win_flag = (episode_rewards["player_0"] > episode_rewards["player_1"])
                palyers_win_list["player_0"].append(1 if p0_win_flag else 0)
                palyers_win_list["player_1"].append(0 if p0_win_flag else 1)

            step += 1

        # 参数平滑
        if not player_0.random_flag and not player_1.random_flag and (i > 25) and i % 15 == 0:
            player_0_cp = copy.deepcopy(player_0)
            player_0.param_smooth_fusion(player_1)
            player_1.param_smooth_fusion(player_0_cp)

        for p in ["player_0", "player_1"]:
            palyers_rewards_list[p].append(episode_rewards[p])
            now_reward[p] = np.mean(palyers_rewards_list[p])
            now_win_rate[p] = np.mean(palyers_win_list[p])

        # print(f'{palyers_win_list=}\n{episode_rewards=}')
        p0_r = now_reward["player_0"]
        p0_w_r = now_win_rate["player_0"]
        p1_r = now_reward["player_1"]
        p1_w_r = now_win_rate["player_1"]
        tq_bar.set_postfix({
            "steps": step,
            'p0-lstR': f'{p0_r:.2f}',
            'p0-winR': f'{p0_w_r:.2f}',
            'p1-lstR': f'{p1_r:.2f}',
            'p1-winR': f'{p1_w_r:.2f}',
            'best_test_r0': f'{best_test_r0:.2f}',
            'best_test_r1': f'{best_test_r1:.2f}'
        })
        if wandb_flag:
            log_dict = {
                "steps": step,
                'p0-lstR': p0_r,
                'p0-winR': p0_w_r,
                'p1-lstR': p1_r,
                'p1-winR': p1_w_r,
                'best_test_r0': best_test_r0,
                'best_test_r1': best_test_r1
            }
            wandb.log(log_dict)
    if wandb_flag:
        wandb.finish()
    env.close()



def env_test(
        env, 
        player_0, 
        player_1,
        cfg,
        test_nums=10,
        focus_player='player_0',
        **kwargs
    ):
    player_0.eval()
    player_1.eval()
    obs, info = env.reset(seed=cfg.seed)
    env_cfg = info["params"]   # UNIT_SAP_RANGE
    tq_bar = tqdm(range(test_nums))
    final_seed = cfg.seed
    palyers_rewards_list = {
        'player_0': deque(maxlen=10),
        'player_1': deque(maxlen=10),
    }
    now_reward= {
        'player_0': -np.inf,
        'player_1': -np.inf
    }
    palyers_win_list = {
        'player_0': deque(maxlen=10),
        'player_1': deque(maxlen=10),
    }
    now_win_rate = {
        'player_0': 0,
        'player_1': 0
    }
    for i in tq_bar:
        obs, info = env.reset(seed=final_seed)
        # dqn collection reset 
        player_0.reset()
        player_1.reset()

        done = False 
        step = 0
        last_obs = None
        last_actions = None
        tq_bar.set_description(f'Episode [ {i+1} / {test_nums}|(seed={final_seed}) ]')
        episode_rewards = {
            'player_0': 0,
            'player_1': 0,
        }
        while not done:
            actions = {}
            # Store current observation for learning
            last_obs = {
                "player_0": obs["player_0"].copy(),
                "player_1": obs["player_1"].copy()
            }
            for agent in [player_0, player_1]:
                actions[agent.player] = agent.policy(step=step, obs=obs[agent.player])
            
            last_actions = copy.deepcopy(actions)
            # Environment step
            # print(f'{actions=}')
            obs, rewards ,terminated, truncated, info = env.step(actions)
            dones = {k: terminated[k] | truncated[k] for k in terminated}
            rewards = {
                "player_0": obs["player_0"]["team_points"][player_0.team_id],
                "player_1": obs["player_1"]["team_points"][player_1.team_id]
            }  
            if last_obs is not None:
                for agent in [player_0, player_1]:
                    for unit_id in range(env_cfg["max_units"]):
                        if obs[agent.player]["units_mask"][agent.team_id][unit_id]:
                            current_state = agent._state_representation(
                                last_obs[agent.player]['units']['position'][agent.team_id][unit_id],
                                last_obs[agent.player]['units']['energy'][agent.team_id][unit_id],
                                last_obs[agent.player]['relic_nodes'],
                                step,
                                last_obs[agent.player]['relic_nodes_mask'],
                                last_obs[agent.player]
                            )
                            next_state = agent._state_representation(
                                obs[agent.player]['units']['position'][agent.team_id][unit_id],
                                obs[agent.player]['units']['energy'][agent.team_id][unit_id],
                                obs[agent.player]['relic_nodes'],
                                step + 1,
                                obs[agent.player]['relic_nodes_mask'],
                                obs[agent.player]
                            )
                            
                            episode_rewards[agent.player] += rewards[agent.player]

            if dones["player_0"] or dones["player_1"]:
                done = True
                p0_win_flag = (episode_rewards["player_0"] > episode_rewards["player_1"])
                palyers_win_list["player_0"].append(1 if p0_win_flag else 0)
                palyers_win_list["player_1"].append(0 if p0_win_flag else 1)

            step += 1

        for p in ["player_0", "player_1"]:
            palyers_rewards_list[p].append(episode_rewards[p])
            now_reward[p] = np.mean(palyers_rewards_list[p])
            now_win_rate[p] = np.mean(palyers_win_list[p])
 
        p0_r = now_reward["player_0"]
        p0_w_r = now_win_rate["player_0"]
        p1_r = now_reward["player_1"]
        p1_w_r = now_win_rate["player_1"]
        tq_bar.set_postfix({
            "steps": step,
            'p0-lstR': f'{p0_r:.2f}',
            'p0-winR': f'{p0_w_r:.2f}',
            'p1-lstR': f'{p1_r:.2f}',
            'p1-winR': f'{p1_w_r:.2f}'
        })
        log_dict = {
            "steps": step,
            'p0-lstR': p0_r,
            'p0-winR': p0_w_r,
            'p1-lstR': p1_r,
            'p1-winR': p1_w_r
        }
        if kwargs.get('print_flag', True):
            print(log_dict)
    env.close() 
    if focus_player is None:
        return (np.mean(palyers_rewards_list['player_0']), np.mean(palyers_rewards_list['player_1']))
    return np.mean(palyers_rewards_list[focus_player])


def step_train(
    player0_random_flag=False, 
    player0_load_dir=None, 
    player0_load_palyer=None,
    player1_random_flag=False,
    player1_load_dir=None, 
    player1_load_palyer=None,
    num_episode=500,
    save_dir=None,
    epsilon_start=0.99,
    learning_rate=2.5e-4
    ):
    all_seed(202501)
    env = LuxAIS3GymEnv(numpy_output=True)
    env = unitReward(env)
    obs, info = env.reset(seed=202501)
    env_cfg = info["params"]
    path_ = os.path.dirname(__file__)
    dt = datetime.now().strftime('%Y%m%d')
    config = Namespace(
        seed=202502,
        num_episode=num_episode,
        batch_size=128, # 128
        min_samples=256,
        save_path=os.path.join(path_, "test_models" ,f'DQN_LuxAI_V0_{dt}') if save_dir is None else save_dir,
        state_dim=8, # unit_pos(2)   + unit_energy(1) + step(1) + enemy
        action_dim=5 + 8, # stay, up, right, down, left, sap-8
        hidden_layers_dim=[220, 220],
        buffer_max_len=8000, #  
        learning_rate=learning_rate,  # 4.5e-4,
        gamma=0.99,
        epsilon=0.01, # 0.01
        target_update_freq=2,
        dqn_type="DoubleDQN",
        epsilon_start=epsilon_start,
        epsilon_decay_factor=0.995, # 0.995,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        env_cfg=env_cfg
    )
    rd_player_0 = DQN("player_0",
        env_cfg,
        state_dim=config.state_dim,
        hidden_layers_dim=config.hidden_layers_dim,
        action_dim=config.action_dim,
        max_len=config.buffer_max_len,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
        target_update_freq=config.target_update_freq,
        dqn_type=config.dqn_type,
        epsilon_start=config.epsilon_start,
        epsilon_decay_factor=config.epsilon_decay_factor,
        device=config.device,
        random_flag=player0_random_flag,
        min_samples=config.min_samples
    )
    if player0_load_dir is not None:
        rd_player_0.random_flag = False 
        rd_player_0.load_model(player0_load_dir, player=player0_load_palyer)
        rd_player_0.random_flag = player0_random_flag
    
    rd_player_1 = DQN("player_1",
        env_cfg,
        state_dim=config.state_dim,
        hidden_layers_dim=config.hidden_layers_dim,
        action_dim=config.action_dim,
        max_len=config.buffer_max_len,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
        target_update_freq=config.target_update_freq,
        dqn_type=config.dqn_type,
        epsilon_start=config.epsilon_start,
        epsilon_decay_factor=config.epsilon_decay_factor,
        device=config.device,
        random_flag=player1_random_flag,
        min_samples=config.min_samples
    )

    if player1_load_dir is not None:
        rd_player_1.random_flag = False 
        rd_player_1.load_model(player1_load_dir, player=player1_load_palyer)
        rd_player_1.random_flag = player1_random_flag

    train_off_policy(
        env, 
        rd_player_0, 
        rd_player_1,
        config,
        wandb_flag=True,
        wandb_project_name="LuxAI-adv-v2.1",
    )
    

def step_test(
    player0_random_flag=False, 
    player0_load_dir=None, 
    player0_load_palyer=None,
    player1_random_flag=False,
    player1_load_dir=None, 
    player1_load_palyer=None
    ):
    all_seed(202501)
    env = LuxAIS3GymEnv(numpy_output=True)
    env = unitReward(env)
    obs, info = env.reset(seed=202501)
    env_cfg = info["params"]  
    path_ = os.path.dirname(__file__)
    dt = datetime.now().strftime('%Y%m%d')
    config = Namespace(
        seed=202502,
        batch_size=128, # 128
        min_samples=256,
        save_path=os.path.join(path_, "test_models" ,f'DQN_LuxAI_V0_{dt}'),
        state_dim=8, # unit_pos(2)   + unit_energy(1) + step(1) + enemy_nums
        action_dim=5 + 8, # stay, up, right, down, left, sap-8
        hidden_layers_dim=[220, 220],
        buffer_max_len=8000, #  
        learning_rate=1.5e-4,  # 4.5e-4,
        gamma=0.99,
        epsilon=0.05, # 0.01
        target_update_freq=2,
        dqn_type="DoubleDQN",
        epsilon_start=0.01,
        epsilon_decay_factor=0.995,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        env_cfg=env_cfg
    )
    rd_player_0 = DQN("player_0",
        env_cfg,
        state_dim=config.state_dim,
        hidden_layers_dim=config.hidden_layers_dim,
        action_dim=config.action_dim,
        max_len=config.buffer_max_len,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
        target_update_freq=config.target_update_freq,
        dqn_type=config.dqn_type,
        epsilon_start=config.epsilon_start,
        epsilon_decay_factor=config.epsilon_decay_factor,
        device=config.device,
        random_flag=player0_random_flag,
        min_samples=config.min_samples
    )
    if player0_load_dir is not None:
        rd_player_0.random_flag = False 
        rd_player_0.load_model(player0_load_dir, player=player0_load_palyer)
        rd_player_0.random_flag = player0_random_flag
    
    rd_player_1 = DQN("player_1",
        env_cfg,
        state_dim=config.state_dim,
        hidden_layers_dim=config.hidden_layers_dim,
        action_dim=config.action_dim,
        max_len=config.buffer_max_len,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        epsilon=config.epsilon,
        target_update_freq=config.target_update_freq,
        dqn_type=config.dqn_type,
        epsilon_start=config.epsilon_start,
        epsilon_decay_factor=config.epsilon_decay_factor,
        device=config.device,
        random_flag=player1_random_flag,
        min_samples=config.min_samples
    )

    if player1_load_dir is not None:
        rd_player_1.random_flag = False 
        rd_player_1.load_model(player1_load_dir, player=player1_load_palyer)
        rd_player_1.random_flag = player1_random_flag
    
    
    rd_player_0.eval()
    rd_player_1.eval()
    res = env_test(
        env, 
        rd_player_0, 
        rd_player_1,
        config
    )
    return res


if __name__ == '__main__':
    path_ = os.path.dirname(__file__)
    model_d_s1 = os.path.join(path_, "test_models" ,f'DQN_LuxAI_s1_dqn_vs_random_v2_1_0228')
    model_d_s2 = os.path.join(path_, "test_models" ,f'DQN_LuxAI_s1_random_vs_dqn_v2_1_0228')
    # step1 dqn VS random
    step_train(
        player0_random_flag=False, 
        player0_load_dir=None, 
        player0_load_palyer=None, 

        player1_random_flag=True,
        player1_load_dir=None, 
        player1_load_palyer=None,

        num_episode=360,
        save_dir=model_d_s1,
        epsilon_start=0.75, # 0.95
        learning_rate=5.5e-4
    )

    # step2 random VS dqn
    step_train(
        player0_random_flag=True, 
        player0_load_dir=None,
        player0_load_palyer=None,

        player1_random_flag=False,
        player1_load_dir=None, 
        player1_load_palyer=None,

        num_episode=360, #1024,
        save_dir=model_d_s2,
        epsilon_start=0.75,
        learning_rate=5.5e-4
    )
    
    # step3 dqn VS dqn [ Freq communicate ]
    # step_train(
    #     player0_random_flag=False, 
    #     player0_load_dir=model_d_s1 + '_bst',
    #     player0_load_palyer='player_0',

    #     player1_random_flag=False,
    #     player1_load_dir=model_d_s2 + '_bst',
    #     player1_load_palyer='player_1',

    #     num_episode=240,  
    #     save_dir=model_d_s3,
    #     epsilon_start=0.5,
    #     learning_rate=5.5e-4
    # )

    # res = step_test(
    #     player0_random_flag=False, 
    #     player0_load_dir=model_d_s2 + '_bst', 
    #     player0_load_palyer='player_0',

    #     player1_random_flag=True,
    #     player1_load_dir=None, 
    #     player1_load_palyer=None
    # )
    # print(f"model_d_s2 {res=}")
