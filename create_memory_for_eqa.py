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

import pickle
from diffusers import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from objnav_benchmark import *

try:
    from GES_vlnce.env_vlnce import *
except:
    pass

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"


if __name__ == "__main__":
    args = get_args()

    dinov2 = torch.hub.load('facebookresearch/dinov2', args.dino_size, source='github').to('cuda')
    yolow = YOLOWorld("yolov8x-worldv2.pt").to('cuda')
    yolow.set_classes(args.detect_classes)

    # memory = VoxelTokenMemory(args, build_map=False, preload_dino=dinov2, preload_yolo=yolow)
    
    paths = os.listdir("/home/orbit/桌面/Nav-2025/data_episode/eqa/data/frames/hm3d-v0")
    eqa_dataset = [path.split('-')[2] for path in paths]


    for i in tqdm(range(39, len(eqa_dataset))):
        torch.cuda.empty_cache()

        all_scense = [name for name in os.listdir("/home/orbit/桌面/Nav-2025/data/scene_datasets/hm3d/hm3d/val")]
        current_scense = [name for name in all_scense if name.endswith(eqa_dataset[i])][0]
        memory_path = f'{args.memory_path}/eqa/{current_scense}'

        
        with open(f"/home/orbit/桌面/Nav-2025/data_episode/eqa/data/frames/hm3d-v0/{paths[i]}/00000.pkl", 'rb') as file:
            state = pickle.load(file)['agent_state']

        args.scene_name = current_scense
        args.load_memory_path = memory_path
        args.memory_save_path = memory_path

        print("creating memory...")
        gc.collect()
        torch.cuda.empty_cache()
        memory = VoxelTokenMemory(args, init_state=state, build_map=True, memory_path=memory_path, preload_dino=dinov2, preload_yolo=yolow, need_diffusion=False)
        

        current_island = memory.Env.sims.pathfinder.get_island(state.position)
        area_shape = memory.Env.sims.pathfinder.island_area(current_island)
        memory.args.random_move_num = int(area_shape / 2) + 1
        
        memory.exploring_create_memory()