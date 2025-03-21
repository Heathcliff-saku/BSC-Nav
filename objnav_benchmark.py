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
from LLMAgent import *
from memory_2 import VoxelTokenMemory
from utils import keyboard_control_fast, adaptive_clustering
import time
import gc
import torch.nn.functional as F
import random
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

from diffusers import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

# 🙀1. 建图一次性把多个楼层建好 --> 根据初始位置（高度/半径）设定Voxel搜索范围 , 根据距离远近设定dict搜索顺序。

def write_metrics(metrics, path="objnav_hm3d_v1_results.csv"):
    if os.path.exists(path):
        with open(path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
            writer.writerow(metrics)
    else:
        with open(path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=metrics.keys())
            writer.writeheader()
            writer.writerow(metrics)

def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'],1024),cv2.COLOR_BGR2RGB)

def show_obs(obs, map_2d=None):
    bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
    if map_2d is not None:
        map_height = map_2d.shape[0]
        bgr = cv2.resize(bgr, (map_height, map_height))
        combined = np.hstack((bgr, map_2d))
        cv2.imshow("Navigation View", combined)
    else:
        cv2.imshow("RGB", bgr)

def Quantizing(model_id):
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16
        )

        diffusion = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        )
        diffusion.enable_model_cpu_offload()
        return diffusion

def get_start_episode(csv_path="objnav_mp3d_v1_results.csv"):
    start_episode = 0
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as csvfile:
            # 计算行数
            row_count = sum(1 for _ in csvfile)
            # 设置起始episode为行数-2 (减去标题行和最后一行)
            # 确保不会出现负数
            start_episode = max(0, row_count - 2)

    return start_episode
        

def load_qwen():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

    return model, processor

class TrajectoryDrawer:
    def __init__(self, rgb, pc, memory, args):
        self.rgb = rgb
        self.pc = pc
        self.memory = memory
        self.args = args
        state = self.memory.Env.agent.get_state()
        self.start_point, _ = self.state2grid_yaw(state)

        self.rgb_2d_map = self._get_rgb_2d_map(self.start_point[1] + int(self.args.sensor_height / self.memory.cs))

        self.PATH_COLOR = np.array([102, 102, 255])
        self.AGENT_COLOR = np.array([76, 0, 153])
        self.FOV_COLOR = np.array([160, 160, 160])

        self.fov = 90  # 视野角度
        self.radius = 30  # 视野半径

    def state2grid_yaw(self, state):
        x_base, z_base, y_base = state.position
        row = int(self.memory.gs / 2 - int(x_base / self.memory.cs))  
        col = int(self.memory.gs / 2 - int(y_base / self.memory.cs))
        h = int(z_base / self.memory.cs) - self.memory.minh
        loc = [row, col, h]
        r = R.from_quat([state.rotation.x, state.rotation.y, state.rotation.z, state.rotation.w])
        yaw = r.as_euler('xyz')[2]  # 获取yaw角(绕z轴的旋转)
        yaw = np.degrees(yaw)  # 将弧度转换为角度

        return loc, yaw

    def _get_rgb_2d_map(self, height):
        height_grid = np.floor((height / self.memory.cs) - self.memory.minh)
        
        rgb_2d_map = np.zeros((self.memory.gs, self.memory.gs, 3))
        height_map = np.full((self.memory.gs, self.memory.gs), -float('inf'))
        
        mask = self.pc[:, 2] <= height_grid
        valid_points = self.pc[mask]
        valid_rgb = self.rgb[mask]
        
        x_indices = valid_points[:, 0].astype(int)
        y_indices = valid_points[:, 1].astype(int)
        z_values = valid_points[:, 2]
        
        valid_mask = (x_indices >= 0) & (x_indices < self.memory.gs) & \
                     (y_indices >= 0) & (y_indices < self.memory.gs)
                    
        x_indices = x_indices[valid_mask]
        y_indices = y_indices[valid_mask]
        z_values = z_values[valid_mask]
        valid_rgb = valid_rgb[valid_mask]

        for i in range(len(x_indices)):
            x, y = x_indices[i], y_indices[i]
            if z_values[i] > height_map[x, y]:
                height_map[x, y] = z_values[i]
                rgb_2d_map[x, y] = valid_rgb[i]

        rgb_2d_map = rgb_2d_map.astype(np.uint8)
        rgb_2d_map = cv2.cvtColor(rgb_2d_map, cv2.COLOR_RGB2BGR)

        return rgb_2d_map

    def draw_line(self, start, end):
        start = np.array(start)
        end = np.array(end)
        start = start.astype(int)
        end = end.astype(int)

        # 在路径上绘制线条
        cv2.line(self.rgb_2d_map, (start[1], start[0]), (end[1], end[0]), 
                 self.PATH_COLOR.tolist(), thickness=2)

    def draw_agent(self, camera_position, camera_yaw):
        rgb_2d_map = self.rgb_2d_map.copy()
        x = int(camera_position[1])  # y坐标映射为x轴
        y = int(camera_position[0])  # x坐标映射为y轴

        # 绘制agent
        cv2.circle(rgb_2d_map, (x, y), 5, self.AGENT_COLOR.tolist(), -1)

        # 计算视野
        start_angle = camera_yaw - self.fov / 2
        end_angle = camera_yaw + self.fov / 2
        
        angles = np.linspace(start_angle, end_angle, 100)
        pts = np.array([[y, x]])  # 转换为 (y, x)
        for angle in angles:
            rad = np.radians(angle)
            px = x + self.radius * np.cos(rad)  # 使用cos和sin正确映射
            py = y + self.radius * np.sin(rad)
            pts = np.append(pts, [[int(px), int(py)]], axis=0)
            
        # 绘制视野范围
        cv2.fillPoly(rgb_2d_map, [pts.astype(np.int32)], self.FOV_COLOR.tolist(), lineType=cv2.LINE_AA)
        
        return rgb_2d_map
    
    def get_map(self, state):
        loc, yaw = self.state2grid_yaw(state)
        self.draw_line(self.start_point, loc)
        self.start_point = loc
        rgb_2d_map = self.draw_agent(loc, yaw) 
        return rgb_2d_map.copy()
    

class GESObjectNavRobot:
    def __init__(self, nav_memory, benchmark_env, task='objnav', load_local_qwen=False):

        self.memory = nav_memory
        self.benchmark_env = benchmark_env 
        self.task = task
        self.client = OpenAI(
                # base_url='https://api.nuwaapi.com/v1',
                # api_key='sk-A2HvPdqB5NN2Dj1AiRk1Z0085Q6PzJ3ls0nbt2BQTcfvWagG'
                base_url='https://xiaoai.plus/v1',
                api_key='sk-maKQVsTB0OwWx8puEfFjP0Bncq6KZ9LtuLl0bjKoz0zBPxEk'
            )
        
        self.api_key_pool = [
            'sk-maKQVsTB0OwWx8puEfFjP0Bncq6KZ9LtuLl0bjKoz0zBPxEk',
            'sk-67UvJf8exciXyfF2SItVenxPgdSo9Zg9thHptAe4tBIWkP7d',
            'sk-Q9wIUZ0mSNQKRZLcCa6t387rRPR5QJrqvMWWzYK2ngBWmwvz',
            'sk-kXvfyQkjctI3cziIytRVSxs2bUHfzEXgQh6C2vnGIbauofvt',
            'sk-m6viotUq3PMIR6ROBjDdODQK9W2Ur6sYc6ujzvBegNMvpsj1',
            'sk-9h5cah2084Mr5xLm9zHGt7MzHRgXD17Y5E6SGcOiOMubG30Q'
        ]
        self.api_id = 0

        self.pattern_loc = re.compile(r'Nav Loc:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]')
        self.pattern_unable = re.compile(r'Nav Loc:\s*Unable to find', re.IGNORECASE)
        self.pattern_unsuccess = re.compile(r'success:\s*(yes|no)', re.IGNORECASE)
        self.pattern_need_forward = re.compile(r'need forward:\s*(yes|no)', re.IGNORECASE)
        self.pattern_id = re.compile(r'best_img_id:\s*(\d+)', re.IGNORECASE)
        
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H/14', pretrained='metaclip_fullcc')
        self.clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.clip_model.to(self.memory.device)

        if load_local_qwen:
            self.qwen_model, self.qwen_processor = load_qwen()

        self.nav_log = {'long_memory_query':0, 'working_memory_query':0, 'search_point':0, 'success':0}

        self.obss = []

    def reset(self, obs=None):
        # 清理之前的观察数据
        if hasattr(self, 'episode_images'):
            del self.episode_images
        if hasattr(self, 'episode_topdowns'):
            del self.episode_topdowns
        if hasattr(self, 'curr_obs'):
            del self.curr_obs
        
        # 重新初始化观察数据
        self.curr_obs = obs
        self.episode_images = [self.curr_obs['rgb']]
        self.episode_topdowns = [adjust_topdown(self.benchmark_env.get_metrics())]
        
        # 重置导航日志
        self.nav_log = {'long_memory_query':0, 'working_memory_query':0, 'search_point':0, 'success':0}

        # 强制进行垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

    def _grid2loc(self, grid_id):
        
        row, col, height = grid_id[0], grid_id[1], grid_id[2]
        
        initial_position = self.memory.Env.original_state.position  # [x, z, y]
        initial_x, initial_z, initial_y = initial_position

        actual_y = initial_y + (row - self.memory.gs // 2) * self.memory.cs
        actual_x = initial_x + (col - self.memory.gs // 2) * self.memory.cs
        # actual_x = initial_z + 0.2
        actual_z = self.benchmark_env.sim.agents[0].get_state().position[1] + 0.2
        # actual_z = (height + self.memory.minh) * self.memory.cs
        
        return np.array([actual_x, actual_z, actual_y])
    
    def _loc2grid(self, loc):
        x, z, y = loc
        initial_position = self.memory.Env.original_state.position
        initial_x, initial_z, initial_y = initial_position
        
        row = int((y - initial_y) / self.memory.cs + self.memory.gs // 2)
        col = int((x - initial_x) / self.memory.cs + self.memory.gs // 2)
        h = int(z / self.memory.cs) - self.memory.minh
        return np.array([row, col, h])
    
    def weighted_cluster_centers(self, top_k_positions, top_k_similarity, eps=10, min_samples=5):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(top_k_positions)
        unique_labels = [lbl for lbl in set(labels) if lbl != -1]

        cluster_info = []
        for lbl in unique_labels:
            cluster_mask = (labels == lbl)
            cluster_points = top_k_positions[cluster_mask]
            cluster_weights = top_k_similarity[cluster_mask]
            weighted_center = np.average(cluster_points, axis=0, weights=cluster_weights)
            avg_similarity = np.mean(cluster_weights)
            cluster_info.append((avg_similarity, weighted_center, np.sum(cluster_mask)))

        cluster_info.sort(key=lambda x: x[0], reverse=True)
        cluster_centers = np.array([info[1] for info in cluster_info])
        cluster_sizes = [info[2] for info in cluster_info]
        
        return cluster_centers, labels, cluster_sizes

    def long_term_memory_retrival(self, text_prompts):
        print("search long memory...")
        while True:
            memory_dict = self.memory.long_memory_filter()
            answer = long_memory_localized(self.client, text_prompts, memory_dict)

            print(answer)
            if self.pattern_unable.search(answer):
                return None
            
            matches = re.findall(r'\*\*Result\*\*: \((.*?)\)', answer)
            if matches:
                # Extract all Nav Loc N: [x,y,z] patterns from within the Result
                loc_matches = re.findall(r'Nav Loc \d+: \[(\d+),\s*(\d+),\s*(\d+)\]', matches[0])
                if loc_matches:
                    # Convert matches to numpy array of coordinates
                    locs = np.array([[int(x), int(y), int(z)] for x,y,z in loc_matches])
                    print("Extracted Loc Array using long memory:", locs)
                    return locs
            # If no matches found, continue loop
            continue

    def long_term_memory_retrival_v2(self, text_prompts):
        print("search long memory...")
        memory_dict = self.memory.long_memory_filter()
        
        if not memory_dict:  # Check if memory_dict is empty
            print("No memories found in long-term memory")
            return None
            
        label_data = {}
        for item in memory_dict:
            label = item['label']
            if label not in label_data:
                label_data[label] = {'locs': [], 'confidences': []}
            label_data[label]['locs'].append(item['loc'])
            label_data[label]['confidences'].append(item['confidence'])
            
        if not label_data:  # Check if label_data is empty
            print("No valid labels found in memory")
            return None
            
        text_inputs = open_clip.tokenize([text_prompts]).to(self.memory.device)
        label_inputs = open_clip.tokenize(list(label_data.keys())).to(self.memory.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            label_features = self.clip_model.encode_text(label_inputs)
            
            text_features = F.normalize(text_features, p=2, dim=-1)
            label_features = F.normalize(label_features, p=2, dim=-1)
            
            similarities = torch.matmul(text_features, label_features.T).squeeze(0)
            
            if similarities.numel() == 0:  # Check if similarities tensor is empty
                print("No similarities computed")
                return None
                
            best_label_idx = torch.argmax(similarities, dim=0).item()
            
        best_label = list(label_data.keys())[best_label_idx]
        best_locs = np.array(label_data[best_label]['locs'])

        np.save(self.memory.memory_save_path + f"/best_locs_{text_prompts}.npy", best_locs)
        best_confidences = np.array(label_data[best_label]['confidences'])

        agent_loc = self._loc2grid(self.benchmark_env.sim.agents[0].get_state().position)
        
        # Calculate distances to agent
        distances = np.linalg.norm(best_locs - agent_loc, axis=1)
        norm_distances = (distances - distances.min()) / (distances.max() - distances.min() + 1e-6)
        norm_confidences = (best_confidences - best_confidences.min()) / (best_confidences.max() - best_confidences.min() + 1e-6)
        scores = 0.2 * (1 - norm_distances) + 0.8 * norm_confidences

        top_k = min(3, len(best_locs))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return best_locs[top_indices]

        
        # cluster_centers, _, _ = self.weighted_cluster_centers(best_locs, best_confidences, eps=8, min_samples=3)
        
        # if len(cluster_centers) >= 3:
        #     return cluster_centers[:3]
        # else:
        #     return cluster_centers

            
    def working_memory_retrival(self, prompts, vis_aug=False):

        if vis_aug:
            path = ['turn_left'] * int(360 / self.memory.args.turn_left)
            self.execute_path(path, save_img_list=True)
            vis = self.obss[::9]
        else:
            vis = None
        if isinstance(prompts, str):
            print("search voxel memory...")
            if vis:
                while True:
                    try:
                        answer = imagenary_helper_visaug(self.client, prompts, vis)
                        print("aug_prompt:", answer)
                    except:
                        print("error, try again .....................................................")
                        time.sleep(5)
                        continue
                    pattern = r"\*\*Enhancement Description\*\*:\s*(.*?)(?=\n|\Z)"
                    match = re.search(pattern, answer, re.DOTALL)
                    if match:
                        text_prompt_extend = match.group(1).strip()
                        break
                    else:
                        continue
            else: 
                while True:
                    try:
                        text_prompt_extend = imagenary_helper(self.client, prompts)
                        break
                    except:
                        print("error, try again .....................................................")
                        time.sleep(5)
                        continue
            
            best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(text_prompt_extend)

        else:
            print("search voxel memory using image observ...")
            best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(prompts)
            
        cluster_centers, _, _ = self.weighted_cluster_centers(top_k_positions, top_k_similarity)
        print("Extracted Loc Array using voxel memory:", cluster_centers)  

        if isinstance(prompts, str):
            np.save(self.memory.memory_save_path + f"/best_pos_topK_{prompts}.npy", np.array(top_k_positions))
            np.save(self.memory.memory_save_path + f"/best_pos_centers_{prompts}.npy", np.array(cluster_centers))

        else:
            if self.memory.args.benchmark_dataset == 'hm3d':
                np.save(self.memory.memory_save_path + f"/best_pos_topK_{self.benchmark_env.current_episode.object_category}.npy", np.array(top_k_positions))
                np.save(self.memory.memory_save_path + f"/best_pos_centers_{self.benchmark_env.current_episode.object_category}.npy", np.array(cluster_centers)) 
        
        best_pos = np.array([cluster_centers])
        return best_pos
    

    def touching_goal(self, text, obss, max_steps=3):
        # self.qwen_model, self.qwen_processor = load_qwen()
        
        current_obss = obss
        for i in range(max_steps):
            while True:
                # succeed_answer = touching_helper(self.client, text, current_obss, self.qwen_model, self.qwen_processor)
                succeed_answer = touching_helper(self.client, text, current_obss)
                print(succeed_answer)
                strategy_match = re.search(r"\*\*Strategy\*\*:\s*'([^']*)'", succeed_answer)
                if strategy_match:
                    current_strategy = strategy_match.group(1)
                    if current_strategy in ['move_forward', 'turn_left', 'turn_right', 'look_up', 'look_down', 'finish_task']:
                        break
                    else:
                        continue
                else:
                    continue

            if current_strategy == 'finish_task':
                break
            else:
                self.execute_path([current_strategy]*4, save_img_list=True)
                current_obss = [self.obss[-1]]
                continue

        # del self.qwen_model
        # del self.qwen_processor
        # gc.collect()
        # torch.cuda.empty_cache()
        # return

    # 🙀 定位到execute_path使用save_img_list=True出现内存泄露
    def check_around(self, prompt, max_around=2):
        if self.task == 'imgnav':
            max_around = 1 # imgnav don't use look up down
        for j in range(max_around):
            action_around = ['turn_left'] * int(360 / self.memory.args.turn_left)
            self.execute_path(action_around, save_img_list=True)
            
            with torch.no_grad():
                obss_tensor = torch.stack([self.preprocess(obs) for obs in self.obss]).to(self.memory.device)
                if isinstance(prompt, str):
                    text_inputs = open_clip.tokenize([prompt]).to(self.memory.device)
                    
                    image_features = self.clip_model.encode_image(obss_tensor)
                    text_features = self.clip_model.encode_text(text_inputs)
                    
                    image_features = F.normalize(image_features, p=2, dim=-1) 
                    text_features = F.normalize(text_features, p=2, dim=-1)
                    
                    similarities = torch.matmul(image_features, text_features.T).squeeze(1)
                    similarities = F.softmax(similarities, dim=0)
                    max_val, max_idx = torch.max(similarities, dim=0)

                else:
                    image_features = self.clip_model.encode_image(obss_tensor)

                    goal_tensor = torch.stack([self.preprocess(prompt)]).to(self.memory.device)
                    goal_features = self.clip_model.encode_image(goal_tensor)

                    image_features = F.normalize(image_features, p=2, dim=-1) 
                    goal_features = F.normalize(goal_features, p=2, dim=-1)
                    
                    similarities = torch.matmul(image_features, goal_features.T).squeeze(1)
                    similarities = F.softmax(similarities, dim=0)
                    max_val, max_idx = torch.max(similarities, dim=0)

            match_obs = [self.obss[max_idx]]

            num_turns = int(360 / self.memory.args.turn_left)
            if max_idx >= num_turns:
                max_idx -= num_turns
            
            target_angle = max_idx.item() * self.memory.args.turn_left
            if target_angle <= 180:
                actions = ['turn_left'] * max_idx.item()
            else:
                total_steps = int(360 / self.memory.args.turn_left)
                actions = ['turn_right'] * (total_steps - max_idx.item())
            
            self.execute_path(actions)
            success, need_forward = self.handle_succeed_check(prompt, match_obs)

            if success:
                self.task_over = True
                print("----------------------")
                print('detect goal, task success')
                if need_forward:
                    self.execute_path(['move_forward']*5)
                # self.touching_goal(text, match_obs, max_steps=3)
                break
            else:
                if j < max_around - 1:
                    print("Target not found in current round. Executing 'look_down' and retrying around.")
                    self.execute_path(['look_down'])
                else:
                    print("Max around attempts reached, target not found.")
                    path = ['look_up']*(max_around-1)
                    if len(path) != 0:
                        self.execute_path(path)


    def handle_succeed_check(self, prompt, obss):
        while True:
            self.api_id += 1 
            self.client.api_key = self.api_key_pool[self.api_id%6]
            try:
                if isinstance(prompt, str):
                    succeed_answer = succeed_determine_singleview(self.client, prompt, obss) 
                else:
                    succeed_answer = succeed_determine_singleview_with_imggoal(self.client, prompt, obss) 
            except:
                print("error, try again .....................................................")
                time.sleep(5)
                continue

            print(succeed_answer)
            match = self.pattern_unsuccess.search(succeed_answer)
            
            if match:
                status = match.group(1).lower()
                if status == 'no':
                    print("No match found.")
                    return False, False
                elif status == 'yes':
                    print(f"Match found.")
                    match = self.pattern_need_forward.search(succeed_answer)
                    if match:
                        need_forward = match.group(1).lower()
                        if need_forward == 'yes':
                            print("Need forward.")
                            return True, True
                        else:
                            print("No need forward.")
                            return True,False
            else:
                print("No valid 'Success' status found in the response. Continuing the loop...")
                continue        

    
    def execute_path(self, path, save_img_list=False):  
        
        if len(self.obss) != 0:
            self.obss = []

        for action in path:
            if not self.memory.args.no_vis:
                # map_2d = self.trajectory_drawer.get_map(self.benchmark_env.sim.agents[0].get_state())
                show_obs(self.curr_obs)
                cv2.waitKey(1)
            if not self.memory.args.quite:
                print("action:", action)
            self.curr_obs = self.benchmark_env.step(action)
            # if action not in ['stop', 'look_up', 'look_down']:
            #     obs = self.memory.Env.sims.step(action)
            if not self.memory.args.no_record:
                self.episode_images.append(self.curr_obs["rgb"].copy())
                self.episode_topdowns.append(adjust_topdown(self.benchmark_env.get_metrics().copy()))

            if save_img_list:
                img = Image.fromarray(self.curr_obs["rgb"][:, :, :3])
                self.obss.append(img)
     
        agent_state = self.benchmark_env.sim.agents[0].get_state()
        self.memory.Env.agent.set_state(agent_state)

    def move2textprompt(self, text_prompt):
        
        self.curr_obs = self.benchmark_env.sim.get_sensor_observations(0)
        self.task_over = False
        # Step 1: Attempt long-term memory retrieval
        best_poses = self.long_term_memory_retrival_v2(text_prompt)
        if best_poses is not None:
            for best_pos in best_poses:
                self.nav_log['long_memory_query'] += 1
                self.nav_log['search_point'] += 1
                best_pos = self._grid2loc(best_pos)
                try:
                    path, self.goal = self.memory.Env.move2point(best_pos)
                    if len(path) > 2000:
                        print("path too long, skip")
                        continue
                    else:
                        self.execute_path(path[:-1])
                except Exception as e:
                    print(f"move2point failed: {e}")
                    continue

                self.check_around(text_prompt) 

                if self.task_over:
                    self.execute_path(['stop'])
                    return self.episode_images, self.episode_topdowns

        # Step 2: Attempt working memory retrieval if long-term retrieval fails
        best_poses = self.working_memory_retrival(text_prompt)
        query_num = min(len(best_poses[0]), 3)
        if best_poses is not None:
            for best_pos in best_poses[0][:query_num]:
                self.nav_log['working_memory_query'] += 1
                self.nav_log['search_point'] += 1
                best_pos = self._grid2loc(best_pos)
                try:
                    path, self.goal = self.memory.Env.move2point(best_pos)
                    if len(path) > 2000:
                        print("path too long, skip")
                        continue
                    else:
                        self.execute_path(path[:-1])
                except Exception as e:
                    print(f"move2point failed: {e}")
                    continue
                
                self.check_around(text_prompt)

                if self.task_over:
                    self.execute_path(['stop'])
                    return self.episode_images, self.episode_topdowns
                else:
                    continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        self.execute_path(['stop'])
        
        return self.episode_images, self.episode_topdowns
    
    def move2imgprompt(self, obs):
        self.curr_obs = self.benchmark_env.sim.get_sensor_observations(0)
        self.task_over = False

        best_poses = self.working_memory_retrival(obs)
        query_num = min(len(best_poses[0]), 3)
        if best_poses is not None:
            for best_pos in best_poses[0][:query_num]:
                self.nav_log['working_memory_query'] += 1
                self.nav_log['search_point'] += 1
                best_pos = self._grid2loc(best_pos)
                try:
                    path, self.goal = self.memory.Env.move2point(best_pos)
                    if len(path) > 2000:
                        print("path too long, skip")
                        continue
                    else:
                        self.execute_path(path[:-1])
                except Exception as e:
                    print(f"move2point failed: {e}")
                    continue
                
                self.check_around(obs)

                if self.task_over:
                    self.execute_path(['stop'])
                    return self.episode_images, self.episode_topdowns
                else:
                    continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        self.execute_path(['stop'])
        
        return self.episode_images, self.episode_topdowns
    

    def move2NaturalLanguageprompt(self, text_prompt):
        self.curr_obs = self.benchmark_env.sim.get_sensor_observations(0)
        self.task_over = False
        
        best_poses = self.working_memory_retrival(text_prompt, vis_aug=False)
        query_num = min(len(best_poses[0]), 5)
        if best_poses is not None:
            for best_pos in best_poses[0][:query_num]:
                self.nav_log['working_memory_query'] += 1
                self.nav_log['search_point'] += 1
                best_pos = self._grid2loc(best_pos)
                try:
                    path, self.goal = self.memory.Env.move2point(best_pos)
                    if len(path) > 2000:
                        print("path too long, skip")
                        continue
                    else:
                        self.execute_path(path[:-1])
                except Exception as e:
                    print(f"move2point failed: {e}")
                    continue
                
                self.check_around(text_prompt)

                if self.task_over:
                    self.execute_path(['stop'])
                    return self.episode_images, self.episode_topdowns
                else:
                    continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        self.execute_path(['stop'])
        
        return self.episode_images, self.episode_topdowns


    def keyboard_explore(self):
        last_action = None
        release_count = 0
        obs = self.benchmark_env.sim.get_sensor_observations(0)
        while True:
            # map_2d = self.trajectory_drawer.get_map(self.benchmark_env.sim.agents[0].get_state())
            show_obs(obs)
            k, action = keyboard_control_fast()
            if k != -1:
                if action == "stop":
                    break
                if action == "record":
                    init_agent_state = self.sims.get_agent(0).get_state()
                    actions_list = []
                    continue
                last_action = action
                release_count = 0
            else:
                if last_action is None:
                    time.sleep(0.01)
                    continue
                else:
                    release_count += 1
                    if release_count > 1:
                        # print("stop after release")
                        last_action = None
                        release_count = 0
                        continue
                    action = last_action

            obs = self.benchmark_env.step(action)
            agent_state = self.benchmark_env.sim.agents[0].get_state()
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

            state = Robot.benchmark_env.sim.agents[0].state
            current_island = Robot.benchmark_env.sim.pathfinder.get_island(state.position)
            print("current_island:", current_island)



if __name__ == "__main__":

    csv_path = "objnav_hm3d_v1_results.csv"
    args = get_args()

    dinov2 = torch.hub.load('facebookresearch/dinov2', args.dino_size, source='github').to('cuda')
    yolow = YOLOWorld("yolov8x-worldv2.pt").to('cuda')
    yolow.set_classes(args.detect_classes)
    # diffusion = Quantizing(args.diffusion_id).to('cuda')

    habitat_benchmark_env = get_objnav_env(args)
    # habitat_benchmark_env.sim.episode_iterator.set
    # goal = [habitat_benchmark_env.episode_iterator.episodes[i].object_category for i in range(200)]
    # goal = list(set(goal))
    # print(goal)

    memory = VoxelTokenMemory(args, build_map=False, preload_dino=dinov2, preload_yolo=yolow)
    Robot = GESObjectNavRobot(memory, habitat_benchmark_env)

    start_episode = get_start_episode(csv_path)


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

        memory_path = f'{args.memory_path}/objectnav/{args.benchmark_dataset}_v1/{current_scense}_island_{current_island}'
        
        
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
        Robot.reset(obs)
        # Robot.keyboard_explore()
        print(f"find {Robot.benchmark_env.current_episode.object_category}")
        episode_images, episode_topdowns = Robot.move2textprompt(f'a {habitat_benchmark_env.current_episode.object_category}')
        
        for image,topdown in zip(episode_images,episode_topdowns):
            fps_writer.append_data(image)
            topdown_writer.append_data(topdown)
        fps_writer.close()
        topdown_writer.close()
        evaluation_metrics = {'success':habitat_benchmark_env.get_metrics()['success'],
                               'spl':habitat_benchmark_env.get_metrics()['spl'],
                               'distance_to_goal':habitat_benchmark_env.get_metrics()['distance_to_goal'],
                               'object_goal':habitat_benchmark_env.current_episode.object_category,
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
