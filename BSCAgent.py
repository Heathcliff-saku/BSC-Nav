import habitat
import os
import argparse
# from args import get_args
from env import *
from tqdm import tqdm
import imageio
import cv2
import csv
from pathlib import Path
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
from vlnce_maps import colorize_draw_agent_and_fit_to_height_vlnce
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

def adjust_topdown(metrics, args):
    if args.nav_task != 'vlnce':
        return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'],1024),cv2.COLOR_BGR2RGB)
    else:
        return cv2.cvtColor(colorize_draw_agent_and_fit_to_height_vlnce(metrics['top_down_map_vlnce'],1024),cv2.COLOR_BGR2RGB)

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

def create_simple_navigation_video(episode_images, episode_topdowns, action_hist, 
                                 output_path='navigation_video.mp4', fps=5):
    """
    创建简单的导航视频，拼接高度一致，保持原始比例
    """
    # 确保动作列表长度匹配
    if len(action_hist) < len(episode_images):
        last_action = action_hist[-1] if action_hist else "unknown"
        action_hist = action_hist + [last_action] * (len(episode_images) - len(action_hist))
    elif len(action_hist) > len(episode_images):
        action_hist = action_hist[:len(episode_images)]
    
    writer = imageio.get_writer(output_path, fps=fps)
    
    for idx, (rgb_img, topdown_img, action) in enumerate(zip(episode_images, episode_topdowns, action_hist)):
        # 确保图像格式正确
        if rgb_img.dtype != np.uint8:
            rgb_img = (rgb_img * 255).astype(np.uint8)
        if topdown_img.dtype != np.uint8:
            topdown_img = (topdown_img * 255).astype(np.uint8)
            
        # 获取原始尺寸
        rgb_h, rgb_w = rgb_img.shape[:2]
        topdown_h, topdown_w = topdown_img.shape[:2]
        
        # 使用RGB图像的高度作为目标高度
        target_height = rgb_h
        
        # 如果topdown高度不同，按比例缩放
        if topdown_h != target_height:
            scale = target_height / topdown_h
            new_width = int(topdown_w * scale)
            topdown_img = cv2.resize(topdown_img, (new_width, target_height))
        
        # 横向拼接（现在高度相同）
        combined = np.hstack((rgb_img, topdown_img))
        
        # 添加文本区域
        text_height = 80
        text_area = np.ones((text_height, combined.shape[1], 3), dtype=np.uint8) * 255
        frame = np.vstack((combined, text_area))
        
        # 添加居中的动作文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame {idx+1}: Action: {action}"
        font_scale = 0.8
        thickness = 2
        
        # 计算文本大小和位置
        (text_width, text_height_size) = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x_position = (frame.shape[1] - text_width) // 2
        y_position = combined.shape[0] + 50
        
        cv2.putText(frame, text, (x_position, y_position), 
                   font, font_scale, (0, 0, 0), thickness)
        
        writer.append_data(frame)
    
    writer.close()
    print(f"Video saved to: {output_path}")
    return output_path
        

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
        self.client = OpenAI(
                # base_url='https://api.nuwaapi.com/v1',
                # api_key='sk-A2HvPdqB5NN2Dj1AiRk1Z0085Q6PzJ3ls0nbt2BQTcfvWagG'
                base_url='https://xiaoai.plus/v1',
                api_key='sk-XalCM6C0Wocy4amFS1RNj8KJMqcNKJbD95Uhgm0rzWkYg15Q'
                # api_key='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'
                
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
        self.state_hist = []
        self.action_hist = []
        self.agent_response_log = []
        self.loc_hist = {'long_memory':[], 'working_memory':[]}

        self.log_dir = "./tmp/trajectory_0"
        os.makedirs(self.log_dir, exist_ok=True)

        self.obss = []

        self.curr_obs = self.benchmark_env.sims.step("turn_right")

    def reset(self, obs=None, log_dir=None):

        # 清理之前的观察数据
        if hasattr(self, 'episode_images'):
            del self.episode_images
        if hasattr(self, 'episode_topdowns'):
            del self.episode_topdowns
        if hasattr(self, 'curr_obs'):
            del self.curr_obs
        
        # 重新初始化观察数据
        self.curr_obs = obs

        if obs is not None:
            self.episode_images = [self.curr_obs['rgb']]
            self.episode_topdowns = [adjust_topdown(self.benchmark_env.get_metrics(), self.memory.args)]
        else:
            self.episode_images = []
            self.episode_topdowns = []

        self.curr_action = ''
        self.curr_agent_response = ''
        self.episode_vedio = []
        # self.episode_vedio = self.draw_vedio_frame()
        
        # 重置导航日志
        self.nav_log = {'long_memory_query':0, 'working_memory_query':0, 'search_point':0, 'success':0}

        self.state_hist = []
        self.agent_response_log = []
        self.loc_hist = {'long_memory':[], 'working_memory':[]}
        if log_dir is not None:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)

        # 强制进行垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

    def draw_vedio_frame(self):
                           
        def put_wrapped_text(img, text, position, font, font_scale, color, thickness, line_spacing=5):
            max_width = img.shape[1] - 40  # 左右各留20像素边距
            words = text.split(' ')
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                # 获取当前行的宽度
                (line_width, _) = cv2.getTextSize(' '.join(current_line), font, font_scale, thickness)[0]
                
                if line_width > max_width:
                    # 如果超过最大宽度，将最后一个词移到下一行
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            # 添加最后一行
            if current_line:
                lines.append(' '.join(current_line))
            
            y = position[1]
            for line in lines:
                cv2.putText(img, line, (position[0], y), font, font_scale, color, thickness)
                y += int((_[1] + line_spacing) * 1.5)  # 行间距
            
            return y  # 返回最后一行的y坐标
        
        if self.memory.args.nav_task in ['objnav', 'ovon']:
            task_goal = self.benchmark_env.current_episode.object_category
            curr_rgb = self.curr_obs['rgb']
            curr_topdown = adjust_topdown(self.benchmark_env.get_metrics(), self.memory.args)
            action = self.curr_action
            response = self.curr_agent_response
        
            # 调整topdown地图大小
            topdown_size = (curr_rgb.shape[0]//2, curr_rgb.shape[1]//2)
            curr_topdown = cv2.resize(curr_topdown, topdown_size)
            
            # 创建底部空白区域用于文本
            text_area_height = 300  # 增加文本区域高度以适应换行
            canvas_width = curr_rgb.shape[1]
            text_area = np.ones((text_area_height, canvas_width, 3), dtype=np.uint8) * 255
            
            # 将RGB图像和缩小后的topdown地图水平连接
            top_row = np.hstack((curr_rgb, cv2.resize(curr_topdown, (curr_rgb.shape[1], curr_rgb.shape[0]))))
            
            # 将图像和文本区域垂直连接
            final_image = np.vstack((top_row, text_area))
            
            # 设置文本参数
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (0, 0, 0)
            thickness = 2
            padding = 20
            
            # 添加任务目标
            next_y = put_wrapped_text(
                final_image,
                f"Task Goal: {task_goal}",
                (padding, top_row.shape[0] + 40),
                font, font_scale, font_color, thickness
            )
            
            # 添加当前动作
            next_y = put_wrapped_text(
                final_image,
                f"Action: {action}",
                (padding, next_y + 20),  # 在上一个文本下方20像素
                font, font_scale, font_color, thickness
            )
            
            # 添加响应
            next_y = put_wrapped_text(
                final_image,
                f"Response: {response}",
                (padding, next_y + 20),
                font, font_scale, font_color, thickness
            )
            
            return final_image


    def _grid2loc(self, grid_id):
        
        row, col, height = grid_id[0], grid_id[1], grid_id[2]
        
        initial_position = self.memory.Env.original_state.position  # [x, z, y]
        initial_x, initial_z, initial_y = initial_position

        actual_y = initial_y + (row - self.memory.gs // 2) * self.memory.cs
        actual_x = initial_x + (col - self.memory.gs // 2) * self.memory.cs
        # actual_x = initial_z + 0.2
        actual_z = self.benchmark_env.sims.agents[0].get_state().position[1] + 0.2
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

        agent_loc = self._loc2grid(self.benchmark_env.sims.agents[0].get_state().position)
        
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

            
    def working_memory_retrival(self, prompts, vis_aug=False, text_aug=True, region_radius=np.inf, curr_grid=None):
        
        if curr_grid is None and region_radius != np.inf:
            curr_state = self.benchmark_env.sims.agents[0].get_state().position
            curr_grid = self._loc2grid(curr_state)

        if vis_aug:
            path = ['turn_left'] * int(360 / self.memory.args.turn_left)
            self.execute_path(path, save_img_list=True)
            vis = self.obss[::9]
        else:
            vis = None
        if isinstance(prompts, str):
            print("search voxel memory...")
            if text_aug:
                if vis:
                    while True:
                        try:
                            answer = imagenary_helper_visaug(self.client, prompts, vis)
                            print("aug_prompt:", answer)
                        except:
                            print("error, try again .....................................................")
                            time.sleep(50)
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
                            time.sleep(50)
                            continue
            else:
                text_prompt_extend = prompts
            
            best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(text_prompt_extend, region_radius=region_radius, curr_grid=curr_grid)

        elif isinstance(prompts, list):
            print("search voxel memory...")
            while True:
                try:
                    text_prompt_extend = imagenary_helper_long_text(self.client, prompts)
                    break
                except:
                    print("error, try again .....................................................")
                    time.sleep(50)
                    continue
                
            best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(text_prompt_extend, region_radius=region_radius, curr_grid=curr_grid)

        else:
            print("search voxel memory using image observ...")
            best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(prompts, region_radius=region_radius, curr_grid=curr_grid)
            
        cluster_centers, _, _ = self.weighted_cluster_centers(top_k_positions, top_k_similarity)
        print("Extracted Loc Array using voxel memory:", cluster_centers)  

        if isinstance(prompts, str):
            if len(prompts) > 64:
                prompts = prompts[:64]
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
                time.sleep(50)
                continue
            
            self.agent_response_log.append(succeed_answer)
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
                # map_2d = self.trajectory_drawer.get_map(self.benchmark_env.sims.agents[0].get_state())
                show_obs(self.curr_obs)
                cv2.waitKey(1)
            if not self.memory.args.quite:
                print("action:", action)
            self.action_hist.append(action)
            self.state_hist.append(self.benchmark_env.sims.agents[0].get_state())
            self.curr_obs = self.benchmark_env.sims.step(action)
            # if action not in ['stop', 'look_up', 'look_down']:
            #     obs = self.memory.Env.sims.step(action)
            if save_img_list:
                img = Image.fromarray(self.curr_obs["rgb"][:, :, :3])
                self.obss.append(img)
     
        agent_state = self.benchmark_env.sims.agents[0].get_state()
        self.memory.Env.agent.set_state(agent_state)

    def save_log(self):
        
        # 由于 AgentState 类型无法直接被 JSON 序列化，这里需要将 state_hist 和 agent_response_log 中的每个元素转换为可序列化的格式
        def to_serializable(obj):
            # 针对 AgentState 类型，尝试提取其属性为 dict
            if hasattr(obj, '__dict__'):
                # 只保留常见的可序列化字段
                return {k: to_serializable(v) for k, v in obj.__dict__.items() if not callable(v) and not k.startswith('_')}
            elif isinstance(obj, (list, tuple)):
                return [to_serializable(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            elif hasattr(obj, 'tolist'):  # 处理 numpy/tensor
                return obj.tolist()
            else:
                return str(obj)  # 兜底转为字符串

        serializable_state_hist = to_serializable(self.state_hist)
        serializable_agent_response_log = to_serializable(self.agent_response_log)
        serializable_loc_hist = to_serializable(self.loc_hist)

        save_data = {
            'state_hist': serializable_state_hist,
            'agent_response_log': serializable_agent_response_log,
            'loc_hist': serializable_loc_hist
        }
        save_path = os.path.join(self.log_dir, 'log_data.json')

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=4)
    
        # for i in range(len(self.episode_images)):
        #     cv2.imwrite(os.path.join(self.log_dir+'_images', f'episode_image_{i}.png'), self.episode_images[i])
        #     cv2.imwrite(os.path.join(self.log_dir+'_topdowns', f'episode_topdown_{i}.png'), self.episode_topdowns[i])


    def move2textprompt(self, text_prompt):
        
        self.curr_obs = self.benchmark_env.sims.get_sensor_observations(0)
        self.task_over = False
        # Step 1: Attempt long-term memory retrieval
        if not self.memory.args.use_only_working_memory:
            best_poses = self.long_term_memory_retrival_v2(text_prompt)
            if best_poses is not None:
                self.loc_hist['long_memory'].extend(best_poses)
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
                        
                        self.save_log()
                        return

        # Step 2: Attempt working memory retrieval if long-term retrieval fails
        best_poses = self.working_memory_retrival(text_prompt)
        query_num = min(len(best_poses[0]), 3)
        if best_poses is not None:
            self.loc_hist['working_memory'].extend(best_poses[0][:query_num])
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
                    
                    self.save_log()
                    return
                else:
                    continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        
        self.save_log()
        return
    
    def move2imgprompt(self, obs):
        self.curr_obs = self.benchmark_env.sims.get_sensor_observations(0)
        self.task_over = False

        best_poses = self.working_memory_retrival(obs)
        query_num = min(len(best_poses[0]), 3)
        if best_poses is not None:
            self.loc_hist['working_memory'].extend(best_poses[0][:query_num])
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
                    
                    self.save_log()
                    return
                else:
                    continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        
        self.save_log()
        
        return
    

    def move2NaturalLanguageprompt(self, text_prompt):
        self.curr_obs = self.benchmark_env.sims.get_sensor_observations(0)
        self.task_over = False
        
        best_poses = self.working_memory_retrival(text_prompt, vis_aug=False)
        query_num = min(len(best_poses[0]), 5)
        if best_poses is not None:
            self.loc_hist['working_memory'].extend(best_poses[0][:query_num])
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
                    
                    self.save_log()
                    return
                else:
                    continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        
        
        self.save_log()
        
        return

    def move2text_attributes_prompt(self, goal_text_intrinsic, goal_text_extrinsic):
        self.curr_obs = self.benchmark_env.sims.get_sensor_observations(0)
        self.task_over = False
        text_prompt = [goal_text_intrinsic, goal_text_extrinsic]
        self.agent_response_log.append(text_prompt)
        
        best_poses = self.working_memory_retrival(text_prompt, vis_aug=False)
        query_num = min(len(best_poses[0]), 5)
        if best_poses is not None:
            self.loc_hist['working_memory'].extend(best_poses[0][:query_num])
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
                
                self.check_around(text_prompt[0])

                if self.task_over:
                    
                    self.save_log()
                    return
                else:
                    continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        
        self.save_log()
        return


    def move2subgoal(self, best_poses, text_prompt):
        query_num = min(len(best_poses[0]), 2)
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
                    return True
                else:
                    continue
        return False

    def move2textprompt_adaptive_region(self, text_prompt, text_aug=False, ridus=30):
        self.task_over = False

        curr_state = self.benchmark_env.sims.agents[0].get_state().position
        curr_grid = self._loc2grid(curr_state)

        for i in range(3):
            best_poses = self.working_memory_retrival(text_prompt, region_radius=ridus, text_aug=text_aug, curr_grid=curr_grid)
            # 按照与curr_grid的距离对best_poses排序，最近的排在最前面
            if best_poses is not None and len(best_poses) > 1:
                distances = np.linalg.norm(best_poses - np.array(curr_grid), axis=1)
                sorted_indices = np.argsort(distances)
                best_poses = best_poses[sorted_indices]
                
            success = self.move2subgoal(best_poses, text_prompt)
            if success:
                return True
            else:
                ridus += 10
            
        return False
        
    def move2VLNprompt(self, text_prompt):
        self.curr_obs = self.benchmark_env.sims.get_sensor_observations(0)
        self.task_over = False
        self.agent_response_log.append(text_prompt)
        subgoals = None
        while subgoals is None:
            subgoals = vln_subgoal_planner_with_obs(self.client, text_prompt)
        # subgoal = '1. Move to {the stairs at the end of the hallway}  \n2. Move to {the bed in the bedroom}  \n3. Move to {the closet}  \n4. Move to {the toilet in the bathroom}  '
        # we need text in {}
        self.agent_response_log.append(subgoals)
        subgoals_list = []
        for subgoal in subgoals.split('\n'):
            subgoal = subgoal.split('.')[1].strip()
            subgoal = subgoal.split('{')[1].split('}')[0].strip()
            subgoals_list.append(subgoal)
        print(subgoals_list)
        for subgoal in subgoals_list:
            print("move to subgoal:", subgoal)
            self.execute_path(['turn_left']*(360 // self.memory.args.turn_left), save_img_list=True)
            # chosen self.obss each 9 images
            # obss = self.obss[::9]
            anchor = vln_anchor_planner_v2(self.client, subgoal, self.obss)
            self.agent_response_log.append(anchor)
            # anchor = anchor.split('Anchor Object:')[1].strip()
            print("anchor:", anchor)
            success = self.move2textprompt_adaptive_region(anchor, text_aug=False, ridus=50)
            if success:
                continue
            else:
                print("failed to move to subgoal")
                # try:
                #     best_poses = self.working_memory_retrival(subgoals_list[-1], region_radius=np.inf, text_aug=False)
                #     success = self.move2subgoal(best_poses, subgoals_list[-1])
                #     break
                # except:
                #     print("move2subgoal failed, try again")
                #     break
        
        
        self.save_log()
        return
    

    def move2VLNprompt_v2(self, text_prompt):
        self.curr_obs = self.benchmark_env.sims.get_sensor_observations(0)
        self.task_over = False

        self.execute_path(['turn_left']*(360 // self.memory.args.turn_left), save_img_list=True)
        obss = self.obss[::9]
        anchor = vln_anchor_planner_v2(self.client, text_prompt, obss)
        anchor = anchor.split('Anchor Object:')[1].strip()
        print(anchor)
        success = self.move2textprompt_adaptive_region(anchor, text_aug=False, ridus=70)
        # best_poses = self.working_memory_retrival(anchor, region_radius=np.inf, vis_aug=False, text_aug=False)
        # query_num = min(len(best_poses[0]), 5)
        # if best_poses is not None:
        #     for best_pos in best_poses[0][:query_num]:
        #         self.nav_log['working_memory_query'] += 1
        #         self.nav_log['search_point'] += 1
        #         best_pos = self._grid2loc(best_pos)
        #         try:
        #             path, self.goal = self.memory.Env.move2point(best_pos)
        #             if len(path) > 2000:
        #                 print("path too long, skip")
        #                 continue
        #             else:
        #                 self.execute_path(path[:-1])
        #         except Exception as e:
        #             print(f"move2point failed: {e}")
        #             continue
                
        #         self.check_around(anchor)

        #         if self.task_over:
        #             
        #             return self.episode_images, self.episode_topdowns, self.episode_vedio
        #         else:
        #             continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        
        
        return

    def keyboard_explore(self):
        last_action = None
        release_count = 0
        obs = self.benchmark_env.sims.get_sensor_observations(0)
        while True:
            # map_2d = self.trajectory_drawer.get_map(self.benchmark_env.sims.agents[0].get_state())
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

            obs = self.benchmark_env.sims.step(action)
            agent_state = self.benchmark_env.sims.agents[0].get_state()
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

            state = self.benchmark_env.sims.agents[0].state
            current_island = self.benchmark_env.sims.pathfinder.get_island(state.position)
            print("current_island:", current_island)

            self.episode_images.append(obs["rgb"].copy())
            self.episode_topdowns.append(adjust_topdown(self.benchmark_env.get_metrics().copy(), self.memory.args))

        
        return self.episode_images, self.episode_topdowns, self.episode_vedio
        
    
