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
from LLMAgent import long_memory_localized, imagenary_helper, succeed_determine_singleview
from memory_2 import VoxelTokenMemory
from utils import keyboard_control_fast
import time
import gc
import torch.nn.functional as F

from diffusers import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from objnav_benchmark import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


def write_metrics(metrics,path="objnav_mp3d_metadata.csv"):
    if os.path.exists(path):
        with open(path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
            writer.writerow(metrics)
    else:
        with open(path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)
            
if __name__ == "__main__":

    args = get_args()

    dinov2 = torch.hub.load('facebookresearch/dinov2', args.dino_size, source='github').to('cuda')
    yolow = YOLOWorld("yolov8x-worldv2.pt").to('cuda')
    yolow.set_classes(args.detect_classes)
    # diffusion = Quantizing(args.diffusion_id).to('cuda')

    habitat_benchmark_env = get_objnav_env(args)
    # habitat_benchmark_env.sim.episode_iterator.set

    memory = VoxelTokenMemory(args, build_map=False, preload_dino=dinov2, preload_yolo=yolow)
    Robot = GESObjectNavRobot(memory, habitat_benchmark_env, load_local_qwen=False)

    for i in tqdm(range(2195)):
        if i < 158:
            obs = Robot.benchmark_env.reset()
            continue
        else:
            obs = Robot.benchmark_env.reset()

            # 每次迭代开始前清理显存
            torch.cuda.empty_cache()
            # 🙀 问题存在于memory的reset中
            # obs = Robot.benchmark_env.reset()
            
            # memory create (if not exsit)
            current_scense = Path(Robot.benchmark_env.current_episode.scene_id).parent.name
            state = Robot.benchmark_env.sim.agents[0].state
            current_island = Robot.benchmark_env.sim.pathfinder.get_island(state.position)
            area_shape = Robot.benchmark_env.sim.pathfinder.island_area(current_island)
            memory_path = f'{args.memory_path}/objectnav/{args.benchmark_dataset}/{current_scense}_island_{current_island}'
            
            Robot.memory.args.random_move_num = int(area_shape / 2) + 1
            Robot.memory.args.dataset = args.benchmark_dataset
            Robot.memory.args.scene_dataset_config_file = Robot.benchmark_env.current_episode.scene_dataset_config
            Robot.memory.args.dataset_dir = Path(Robot.benchmark_env.current_episode.scene_id).parent.parent
            Robot.memory.args.scene_name = current_scense
            Robot.memory.args.load_memory_path = memory_path
            Robot.memory.args.memory_save_path = memory_path

            args = Robot.memory.args

            if os.path.exists(memory_path):
                Robot.memory.load_memory(init_state=state)

            else:
                print("creating memory..., random_move_num:", Robot.memory.args.random_move_num)
                del Robot.memory
                Robot.memory = VoxelTokenMemory(args, init_state=state, build_map=True, memory_path=memory_path, preload_dino=dinov2, preload_yolo=yolow)
                Robot.memory.exploring_create_memory()

            # perform task
            Robot.reset(obs)
            # Robot.keyboard_explore()
            # print(f"find {Robot.benchmark_env.current_episode.object_category}")
            # episode_images, episode_topdowns = Robot.move2textprompt(f'a {habitat_benchmark_env.current_episode.object_category}')
            
            # for image,topdown in zip(episode_images,episode_topdowns):
            #     fps_writer.append_data(image)
            #     topdown_writer.append_data(topdown)

            evaluation_metrics = {
                                'id': args.scene_name,
                                'island': current_island,
                                'island_area': area_shape,
                                'long_memory_query': Robot.nav_log['long_memory_query'],
                                'working_memory_query': Robot.nav_log['working_memory_query'],
                                'search_point': Robot.nav_log['search_point']
                                    }
            print(habitat_benchmark_env.get_metrics())
            # write_metrics(evaluation_metrics) 
            
            del evaluation_metrics
            gc.collect()
            torch.cuda.empty_cache()
        
