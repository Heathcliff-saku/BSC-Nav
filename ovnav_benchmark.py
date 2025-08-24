import habitat
import os
import argparse
from args import get_args
from env import *
from tqdm import tqdm
import imageio
import cv2
import csv
from pathlib import Path
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
import re
from openai import OpenAI
import open_clip
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from ultralytics import YOLOWorld
from LLMAgent import long_memory_localized, imagenary_helper, succeed_determine_singleview, touching_helper
from memory_2 import VoxelTokenMemory
from utils import keyboard_control_fast, adaptive_clustering
import time
import gc
import torch.nn.functional as F

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from diffusers import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel

from objnav_benchmark import *


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# 🙀1. 建图一次性把多个楼层建好 --> 根据初始位置（高度/半径）设定Voxel搜索范围 , 根据距离远近设定dict搜索顺序。


if __name__ == "__main__":

    csv_path = "ovnav_mp3d_results_forfig.csv"
    args = get_args()

    dinov2 = torch.hub.load('facebookresearch/dinov2', args.dino_size, source='github').to('cuda')
    yolow = YOLOWorld("yolov8x-worldv2.pt").to('cuda')
    yolow.set_classes(args.detect_classes)
    # diffusion = Quantizing(args.diffusion_id).to('cuda')

    habitat_benchmark_env = get_ovon_env(args)
    # habitat_benchmark_env.sim.episode_iterator.set
    # goal = [habitat_benchmark_env.episode_iterator.episodes[i].object_category for i in range(200)]
    # goal = list(set(goal))
    # print(goal)

    memory = VoxelTokenMemory(args, build_map=False, preload_dino=dinov2, preload_yolo=yolow)
    Robot = GESObjectNavRobot(memory, habitat_benchmark_env) # task ['objnav', 'instance_imgnav','imgnav']

    start_episode = get_start_episode(csv_path)
    start_episode = 28

    for i in tqdm(range(args.eval_episodes)):
        if i < start_episode:
            obs = Robot.benchmark_env.reset()
            continue
           
        # 每次迭代开始前清理显存
        torch.cuda.empty_cache()
        # 🙀 问题存在于memory的reset中
        obs = Robot.benchmark_env.reset()

        # Robot.keyboard_explore()
        
        dir = "./tmp/trajectory_%d"%i
        os.makedirs(dir, exist_ok=True)
        fps_writer = imageio.get_writer("%s/fps.mp4"%dir, fps=4)
        topdown_writer = imageio.get_writer("%s/metric.mp4"%dir,fps=4)

        # memory create (if not exsit)
        current_scense = Path(Robot.benchmark_env.current_episode.scene_id).parent.name

        state = Robot.benchmark_env.sim.agents[0].state
        current_island = Robot.benchmark_env.sim.pathfinder.get_island(state.position)
        area_shape = Robot.benchmark_env.sim.pathfinder.island_area(current_island)

        memory_path = f'{args.memory_path}/ovnav/{args.benchmark_dataset}/{current_scense}_island_{current_island}'
        print(memory_path)
        
        Robot.memory.args.dataset = args.benchmark_dataset
        Robot.memory.args.scene_dataset_config_file = Robot.benchmark_env.current_episode.scene_dataset_config
        Robot.memory.args.dataset_dir = Path(Robot.benchmark_env.current_episode.scene_id).parent.parent
        Robot.memory.args.scene_name = current_scense
        Robot.memory.args.load_memory_path = memory_path

        if os.path.exists(memory_path):
            Robot.memory.load_memory(init_state=state)

        else:
            print("creating memory...")
            Robot.memory.create_memory()

        # perform task
        Robot.reset(obs, dir)
        print(f"find {Robot.benchmark_env.current_episode.object_category}")
        # goal_img.show()
        episode_images, episode_topdowns, episode_vedio = Robot.move2NaturalLanguageprompt(f'a {habitat_benchmark_env.current_episode.object_category}')
        
        for image,topdown in zip(episode_images,episode_topdowns):
            fps_writer.append_data(image)
            topdown_writer.append_data(topdown)
        fps_writer.close()
        topdown_writer.close()
        create_simple_navigation_video(episode_images, episode_topdowns, Robot.action_hist, "%s/navigation.mp4"%dir)
        evaluation_metrics = {'success':habitat_benchmark_env.get_metrics()['success'],
                               'spl':habitat_benchmark_env.get_metrics()['spl'],
                               'distance_to_goal':habitat_benchmark_env.get_metrics()['distance_to_goal'],
                               'object_goal': habitat_benchmark_env.current_episode.object_category,
                               'id': args.scene_name,
                               'island': current_island,
                               'island_area': area_shape,
                               'long_memory_query': Robot.nav_log['long_memory_query'],
                               'working_memory_query': Robot.nav_log['working_memory_query'],
                               'search_point': Robot.nav_log['search_point']
                                }
        print(habitat_benchmark_env.get_metrics())
        write_metrics(evaluation_metrics, csv_path) 
        
        del fps_writer
        del topdown_writer
        del evaluation_metrics
        gc.collect()
        torch.cuda.empty_cache()
