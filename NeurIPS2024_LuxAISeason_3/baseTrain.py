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
from baseDQN import all_seed, DQN
# from advDQN import all_seed, DQN
from luxai_s3.wrappers import LuxAIS3GymEnv


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
    for i in tq_bar:
        obs, info = env.reset(seed=final_seed)
        # dqn collection reset 
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
                                rewards[agent.player],
                                next_state,
                                dones[agent.player]
                            )
                            
                            episode_rewards[agent.player] += rewards[agent.player]

                if not player_0.random_flag:
                    player_0.update(cfg.batch_size)
                if not player_1.random_flag:
                    player_1.update(cfg.batch_size)

            if dones["player_0"] or dones["player_1"]:
                done = True
                player_0.save_model(cfg.save_path)
                player_1.save_model(cfg.save_path)
                p0_win_flag = (episode_rewards["player_0"] > episode_rewards["player_1"])
                palyers_win_list["player_0"].append(1 if p0_win_flag else 0)
                palyers_win_list["player_1"].append(0 if p0_win_flag else 1)

            step += 1

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
            'p1-winR': f'{p1_w_r:.2f}'
        })
        if wandb_flag:
            log_dict = {
                "steps": step,
                'p0-lstR': p0_r,
                'p0-winR': p0_w_r,
                'p1-lstR': p1_r,
                'p1-winR': p1_w_r
            }
            wandb.log(log_dict)
    if wandb_flag:
        wandb.finish()
    env.close()


def step_train(
    player0_random_flag=False, 
    player0_load_dir=None, 
    player0_load_palyer=None,
    player1_random_flag=False,
    player1_load_dir=None, 
    player1_load_palyer=None,
    num_episode=500,
    save_dir=None,
    epsilon_start=0.99
):
    all_seed(202501)
    env = LuxAIS3GymEnv(numpy_output=True)
    obs, info = env.reset(seed=202501)
    env_cfg = info["params"]  
    path_ = os.path.dirname(__file__)
    dt = datetime.now().strftime('%Y%m%d')
    config = Namespace(
        seed=202502,
        num_episode=num_episode,
        batch_size=128,
        min_samples=256,
        save_path=os.path.join(path_, "test_models" ,f'DQN_LuxAI_V0_{dt}') if save_dir is None else save_dir,
        state_dim=7, # unit_pos(2) + closest_relic(2) + unit_energy(1) + step(1) 
        action_dim=6, # stay, up, right, down, left, sap
        hidden_layers_dim=[220, 220],
        buffer_max_len=20000,
        learning_rate=2.5e-4, # 0.0001
        gamma=0.99,
        epsilon=0.01,
        target_update_freq=env_cfg['max_units'] * 10,
        dqn_type="DoubleDQN",
        epsilon_start=epsilon_start,
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

    train_off_policy(
        env, 
        rd_player_0, 
        rd_player_1,
        config,
        wandb_flag=True,
        wandb_project_name="LuxAI",
    )



if __name__ == '__main__':
    path_ = os.path.dirname(__file__)
    model_d_s0 = os.path.join(path_, "test_models" ,f'DQN_LuxAI_s1_dqn_vs_dqn_v0')
    model_d_s1 = os.path.join(path_, "test_models" ,f'DQN_LuxAI_s1_dqn_vs_random_v1')
    model_d_s2 = os.path.join(path_, "test_models" ,f'DQN_LuxAI_s1_dqn_vs_dqn_v1')
    # step_train(
    #     player0_random_flag=False, 
    #     player0_load_dir=None, 
    #     player0_load_palyer=None,

    #     player1_random_flag=False,
    #     player1_load_dir=None, 
    #     player1_load_palyer=None,

    #     num_episode=800,
    #     save_dir=model_d_s0,
    #     epsilon_start=0.1
    # )

    # step1 dqn VS random
    step_train(
        player0_random_flag=False, 
        player0_load_dir=None, # model_d_s0, 
        player0_load_palyer=None, #'player_0',

        player1_random_flag=True,
        player1_load_dir=None, 
        player1_load_palyer=None,

        num_episode=488,
        save_dir=model_d_s1
    )
    # step2 dqn(load model) VS dqn(load model) 
    # step_train(
    #     player0_random_flag=False, 
    #     player0_load_dir=model_d_s1, 
    #     player0_load_palyer='player_0',

    #     player1_random_flag=False,
    #     player1_load_dir=model_d_s1, 
    #     player1_load_palyer='player_0',

    #     num_episode=1200,
    #     save_dir=model_d_s2,
    #     epsilon_start=0.1
    # )

