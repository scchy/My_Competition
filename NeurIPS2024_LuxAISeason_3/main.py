# python 
# Author: Scc_hy
# Create Date: 2025-02-08
# func: run:
#   luxai-s3  main.py  main.py --seed 2025 -o replay.html
# ==========================================================================================
import json
from typing import Dict
import os 
import sys
from argparse import Namespace
import numpy as np
from kit import from_json
from baseDQN import all_seed, DQN
import torch 


agent_dict = dict() 
agent_prev_obs = dict()


def agent_fn(observation, configurations):
    """
    agent definition for kaggle submission.
    """
    global agent_dict
    obs = observation.obs
    if type(obs) == str:
        obs = json.loads(obs)

    env_cfg = configurations["env_cfg"]
    path_ = os.path.dirname(__file__)
    model_d_s1 = os.path.join(path_, "test_models" ,f'DQN_LuxAI_s1_dqn_vs_random_v1')
    config = Namespace(
        seed=202502,
        num_episode=100,
        batch_size=128,
        min_samples=0,
        save_path=model_d_s1,
        state_dim=7, # unit_pos(2) + closest_relic(2) + unit_energy(1) + step(1) 
        action_dim=6, # stay, up, right, down, left, sap
        hidden_layers_dim=[220, 220],
        buffer_max_len=10000,
        learning_rate=2.5e-4, # 0.0001
        gamma=0.99,
        epsilon=0.01,
        target_update_freq=100,
        dqn_type="DoubleDQN",
        epsilon_start=0.1,
        epsilon_decay_factor=0.995,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        env_cfg=env_cfg
    )
    step = observation.step
    player = observation.player
    remainingOverageTime = observation.remainingOverageTime
    if step == 0:
        agent_dict[player] = DQN(
            player,
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
            random_flag=False,
            min_samples=config.min_samples
        )
        agent_dict[player].load_model(model_d_s1, player='player_0')
        agent_dict[player].eval()

    agent = agent_dict[player]
    actions = agent.policy(step, from_json(obs), remainingOverageTime)
    return dict(action=actions.tolist())


if __name__ == "__main__":
    
    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)
    # pwd = os.popen('pwd').readlines()
    # ls = os.popen('ls').readlines()
    # print(f'{pwd=}\n{ls=}')
    step = 0
    player_id = 0
    env_cfg = None
    i = 0
    while True:
        inputs = read_input()
        raw_input = json.loads(inputs)
        observation = Namespace(**dict(step=raw_input["step"], obs=raw_input["obs"], remainingOverageTime=raw_input["remainingOverageTime"], player=raw_input["player"], info=raw_input["info"]))
        if i == 0:
            env_cfg = raw_input["info"]["env_cfg"]
            player_id = raw_input["player"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=env_cfg))
        # send actions to engine
        print(json.dumps(actions))

