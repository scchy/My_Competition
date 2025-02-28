import time

import flax.serialization
from luxai_s3.params import EnvParams
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
import numpy as np


def overview_random_play():
    np.random.seed(2)
    env = LuxAIS3GymEnv()
    env = RecordEpisode(env, save_dir="episodes")
    env_params = EnvParams(map_type=0, max_steps_in_match=100)
    obs, info = env.reset(seed=1, options=dict(params=env_params))

    print("Benchmarking time")
    stime = time.time()
    N = env_params.max_steps_in_match * env_params.match_count_per_episode
    print(env.action_space['player_0'].high.shape, env.action_space['player_0'].low.shape)
    for _ in range(N):
        # 观察环境
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        # print(f"{obs=} {reward=} {terminated=} {truncated=} {info=}" )
        units_p = obs['player_0'].units.position
        print(f'{units_p=}')
        print(f"{obs['player_0'].units.position.shape=}  {obs['player_0'].units.energy.shape=}" )
        print(f"{obs['player_0'].units_mask.shape=}" )
        unit_mask = obs['player_0'].units_mask
        available_unit_ids = np.where(unit_mask)[0]
        print(f'{available_unit_ids=}')
        print(f"{obs['player_0'].sensor_mask.shape=}" )
        print(f"{obs['player_0'].map_features.energy.shape=}" )
        print(f"{obs['player_0'].map_features.tile_type.shape=}" )
        print(f"{obs['player_0'].map_features=}")
        print(f"{obs['player_0'].relic_nodes.shape=}" )
        print(f"{obs['player_0'].relic_nodes_mask.shape=}" ) # (6,)
        # relic_nodes[relic_nodes_mask]
        relic_nodes = obs['player_0'].relic_nodes
        relic_nodes_mask = obs['player_0'].relic_nodes_mask
        v_relic_nodes = relic_nodes[relic_nodes_mask]
        print(f"{relic_nodes=} {relic_nodes_mask=} {v_relic_nodes=}" )
        print(f"{obs['player_0'].team_points.shape=}" )
        print(f"{obs['player_0'].steps.shape=}" )
        print(f"{obs['player_0'].match_steps.shape=}" )
        unit_mask = np.array(obs['player_0'].units_mask[0]) # 
        available_units = np.where(unit_mask)
        print(f'{available_units=}')
        print(f"{obs['player_0'].team_points=}" )
        print(f"{reward=} \n{terminated=} \n{truncated=} \n{info.keys()=}: \n{info['discount']=} {info['player_0']=} {info['player_1']=}")
        break
    
    etime = time.time()
    print(f"FPS: {N / (etime - stime)}")

    env.close()



if __name__ == "__main__":
    overview_random_play()

    
