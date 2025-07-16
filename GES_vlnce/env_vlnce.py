import os
import habitat_sim
from typing import Dict, List, Tuple, Union
import numpy as np
import magnum as mn
import time
import random
import imageio
from tqdm import tqdm
import cv2


import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import attr
# vln_ce (conda activate shouwei-nav-vlnce)
from habitat import Env
from habitat.utils.visualizations import maps
from VLN_CE.vlnce_baselines.config.default import get_config
from habitat.datasets import make_dataset
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

import sys
sys.path.append("/home/orbit/桌面/Nav-2025/")
from args import get_args

def get_vlnce_env(args):
    config = get_config(args.MP3D_CONFIG_PATH, opts=None)
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    # dataset_split = dataset.get_splits(args.split_num)[args.split_id]

    sim = Env(config.TASK_CONFIG, dataset)
    return sim


def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map_vlnce'],1024),cv2.COLOR_BGR2RGB)

def keyboard_control():
    k = cv2.waitKey(0)
    if k == ord("w"):
        action = "move_forward"
    elif k == ord("a"):
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("4"):
        action = "look_up"
    elif k == ord("5"):
        action = "look_down"
    elif k == ord("q"):
        action = 'stop'
    elif k == ord(" "):
        return k, "record"
    elif k == -1:
        return k, None
    else:
        return -1, None
    return k, action

def show_obs(obs):
    # 获取 RGB 图像并转换为 BGR 格式
    bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
    cv2.imshow("RGB", bgr)

if __name__ == "__main__":
    
    args = get_args()

    # env = NavBenchmarkEnv(args)
    # habitat_env = env.sims
    # evaluation_metrics = []
    
    # for i in tqdm(range(args.eval_episodes)):
    #     obs = habitat_env.reset()
    #     dir = "./tmp/trajectory_%d"%i
    #     os.makedirs(dir, exist_ok=True)
    #     fps_writer = imageio.get_writer("%s/fps.mp4"%dir, fps=4)
    #     topdown_writer = imageio.get_writer("%s/metric.mp4"%dir,fps=4)
    #     episode_images = [obs['rgb']]
    #     episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]
        
    #     print(habitat_env.current_episode.object_category)


    #     while not habitat_env.episode_over:
    #         show_obs(obs)
    #         k, action = keyboard_control()
    #         print(action)
    #         obs = habitat_env.step(action)
    #         agent_state = env.agent.get_state()
    #         print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

    #         episode_images.append(obs['rgb'])
    #         episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
    #         # print(habitat_env.get_metrics())
        
    #     for image,topdown in zip(episode_images,episode_topdowns):
    #         fps_writer.append_data(image)
    #         topdown_writer.append_data(topdown)
    #     fps_writer.close()
    #     topdown_writer.close()
    #     evaluation_metrics.append({'success':habitat_env.get_metrics()['success'],
    #                            'spl':habitat_env.get_metrics()['spl'],
    #                            'distance_to_goal':habitat_env.get_metrics()['distance_to_goal'],
    #                            'object_goal':habitat_env.current_episode.object_category})
        
    # print(evaluation_metrics)
        

    # env = NavEnv(args)
    
    # env.keyboard_explore()
    # env.move2point(goal=np.array([1.16672724,  3.2034254, -0.4141059]))
    env = get_vlnce_env(args)
    obs = env.reset()
    dir = "./tmp/trajectory_test"
    os.makedirs(dir, exist_ok=True)
    fps_writer = imageio.get_writer("%s/fps.mp4"%dir, fps=4)
    topdown_writer = imageio.get_writer("%s/metric.mp4"%dir,fps=4)
    episode_images = [obs['rgb']]
    episode_topdowns = [adjust_topdown(env.get_metrics())]

    while not env.episode_over:
        show_obs(obs)
        k, action = keyboard_control()
        print(action)
        obs = env.step(action.upper())
        agent_state = env.sim.agents[0].get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

        episode_images.append(obs['rgb'])
        episode_topdowns.append(adjust_topdown(env.get_metrics()))
        # print(habitat_env.get_metrics())
    
    for image,topdown in zip(episode_images,episode_topdowns):
        fps_writer.append_data(image)
        topdown_writer.append_data(topdown)
    fps_writer.close()
    topdown_writer.close()