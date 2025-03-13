import numpy as np
import pyarrow.feather as feather
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import random
import json
from math import fabs
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import cv2
import os
import h5py
import time
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from args import get_args
from utils import *
from env import NavEnv
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLOWorld
import math
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from collections import deque
from typing import List, Tuple
from itertools import product
from collections import defaultdict
import re
import math
from tqdm import tqdm
import gc
from sklearn.cluster import DBSCAN

class VoxelTokenMemory():
    def __init__(self, args, memory_path=None, init_state=None, build_map=False, preload_dino=None, preload_yolo=None, preload_diffusion=None):
        self.args = args
        self.device = 'cuda'
        if not preload_dino:
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', self.args.dino_size, source='github').to(self.device)
        else:
            self.dinov2 = preload_dino
        # self.gdino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        # self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
        if not preload_yolo:
            self.yolow = YOLOWorld("yolov8x-worldv2.pt").to(self.device)
            self.yolow.set_classes(self.args.detect_classes)
        else:
            self.yolow = preload_yolo

        # if not preload_diffusion:
        #     self.Quantizing(self.args.diffusion_id)
        #     self.diffusion = self.diffusion.to(self.device)
        # else:
        #     self.diffusion = preload_diffusion
        
        
        if memory_path:
            self.memory_save_path = memory_path
        else:
            self.memory_save_path = os.path.join(self.args.memory_path, self.args.scene_name)

        self.transform = transforms.Compose([
                    transforms.Resize((self.args.query_height, self.args.query_width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
                ])
        self.transform_ = transforms.Compose([
                    transforms.Resize((self.args.query_height, self.args.query_width)),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
        ])
        
        self.Env = NavEnv(self.args, init_state, build_map)

        self.inv_init_base_tf = []
        
        self.patch_h = 14
        self.patch_w = 14
        self.n_patch_w = self.args.query_width // self.patch_w
        self.n_patch_h = self.args.query_height // self.patch_h
        
        self.base_transform = np.eye(4)
        self.base_transform[0, :3] = self.args.base_forward_axis
        self.base_transform[1, :3] = self.args.base_left_axis
        self.base_transform[2, :3] = self.args.base_up_axis
        
        self.base2cam_tf = np.eye(4)
        self.base2cam_tf[:3, :3] = np.array([self.args.base2cam_rot]).reshape((3, 3))
        self.base2cam_tf[1, 3] = self.args.sensor_height

        self.cs = self.args.cell_size
        self.gs = self.args.grid_size
        self.depth_sample_rate = self.args.depth_sample_rate
        
        self.cv_map = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.max_height = np.full((self.gs, self.gs), -np.inf)
        self.FrontierMap = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        
        self.calib_mat = get_sim_cam_mat_with_fov(self.args.height, self.args.width, fov=90)
        # self.calib_mat = np.array(self.args.cam_calib_mat).reshape((3, 3))
        self.min_depth = self.args.min_depth
        self.max_depth = self.args.max_depth
        
        self.token_dim = 1024
        # update memory when reach iter_size 
        self.iter_size = 50000
        # the max token of each voxel
        self.cache_size = 10
        
        self.neighbor_radius = 1
        self.boring_threshold = 0.95
        self.surprise_threshold = 0.5
        
        self.camera_height = self.args.sensor_height
        
        self.floor_height = self.args.floor_height
        self.map_height = self.args.map_height
        
        self.maxh = int(self.map_height / self.cs) 
        self.minh = int(self.floor_height / self.cs)
        
        (
         self.grid_feat, 
         self.grid_feat_pos, 
         self.grid_rgb_pos,
         self.grid_feat_dis,
         self.weight, 
         self.occupied_ids, 
         self.grid_rgb,
         self.max_id,
         self.iter_id,

         self.base_height
        ) = self._init_cache()
        
        # self.gdino_results = []
        self.yolow_results = []
        self.long_memory_dict = []

    def _clear_memory(self):
        """清理内存中的大型数组和对象"""
        if hasattr(self, 'grid_feat'):
            del self.grid_feat
        if hasattr(self, 'grid_feat_pos'):
            del self.grid_feat_pos
        if hasattr(self, 'grid_rgb_pos'): 
            del self.grid_rgb_pos
        if hasattr(self, 'grid_feat_dis'):
            del self.grid_feat_dis
        if hasattr(self, 'weight'):
            del self.weight
        if hasattr(self, 'occupied_ids'):
            del self.occupied_ids
        if hasattr(self, 'grid_rgb'):
            del self.grid_rgb
        gc.collect()
        torch.cuda.empty_cache()

    def load_memory(self, init_state=None, build_map=False):
        # 先清理之前的内存
        self._clear_memory()
        self.Env.reset(self.args, init_state=init_state, build_map=build_map)

        # 重新分配内存
        (
         self.grid_feat, 
         self.grid_feat_pos, 
         self.grid_rgb_pos,
         self.grid_feat_dis,
         self.weight, 
         self.occupied_ids, 
         self.grid_rgb,
         self.max_id,
         self.iter_id,
         self.base_height
        ) = self._init_cache()

        self.memory_save_path = self.args.load_memory_path

        if not build_map:
        
            self.feat_path = self.memory_save_path+"/feat.h5df"
            self.max_id = int(np.load(self.memory_save_path+"/max_id.npy"))
            self.grid_rgb_pos[:self.max_id] = np.load(self.memory_save_path+"/grid_rgb_pos.npy")
            self.grid_rgb[:self.max_id] = np.load(self.memory_save_path+"/grid_rgb.npy")
            self.weight[:self.max_id] = np.load(self.memory_save_path+"/weight.npy")
            self.occupied_ids[:self.max_id] = np.load(self.memory_save_path+"/occupied_ids.npy")
            self.Env.original_state.position = np.load(self.memory_save_path+"/original_pos.npy") 

            with open(self.memory_save_path+"/long_memory.json", "r") as f:
                self.long_memory_dict = json.load(f)

            self.minh, self.maxh = np.load(self.memory_save_path+"/map_height.npy")

            if self.args.load_single_floor:    
                self.base_height = np.load(self.memory_save_path+"/base_height.npy")
                
                base_height_array = np.array(self.base_height).reshape(-1, 1)
                min_samples = len(self.base_height)//5 if len(self.base_height)//5 > 0 else 1
                clustering = DBSCAN(eps=0.4, min_samples=min_samples).fit(base_height_array)
                
                floor_heights = []
                for label in set(clustering.labels_):
                    if label != -1:
                        mask = clustering.labels_ == label
                        floor_height = np.mean(base_height_array[mask])
                        floor_heights.append(floor_height)
                self.floor_heights = sorted(floor_heights)
                self.num_floors = len(self.floor_heights)
                print(f"detect {self.num_floors} floors, heights: {self.floor_heights}\n")
                current_height = self.Env.agent.get_state().position[1]
                current_floor = np.argmin(np.abs(np.array(self.floor_heights) - current_height))
                print(f"current floor: {current_floor}\n")

                pos_range = [self.grid_rgb_pos[:self.max_id][:, 2].min(), self.grid_rgb_pos[:self.max_id][:, 2].max()]
                
                if self.num_floors == 1:
                    current_floor_range = pos_range
                else:
                    floor_ranges = []
                    for i in range(self.num_floors):
                        if i == 0:
                            # 最低层，下限使用pos_range最小值
                            floor_min = pos_range[0]
                            floor_max = pos_range[0] + (self.floor_heights[1] - self.floor_heights[0])/self.cs
                        elif i == self.num_floors - 1:
                            # 最高层，上限使用pos_range最大值
                            floor_min = pos_range[0] + (self.floor_heights[i] - self.floor_heights[0])/self.cs
                            floor_max = pos_range[1]
                        else:
                            floor_min = pos_range[0] + (self.floor_heights[i] - self.floor_heights[0])/self.cs
                            floor_max = pos_range[0] + (self.floor_heights[i+1] - self.floor_heights[0])/self.cs
                        floor_ranges.append([int(floor_min)+1, int(floor_max)-1])
                    current_floor_range = floor_ranges[current_floor]
                print(f"current_floor_range: {current_floor_range}\n")

                self.floor_min_height = current_floor_range[0]
                self.floor_max_height = current_floor_range[1]

                height_mask = np.logical_and(self.grid_rgb_pos[:self.max_id, 2] >= self.floor_min_height,
                                        self.grid_rgb_pos[:self.max_id, 2] <= self.floor_max_height)
                temp_pos = self.grid_rgb_pos[:self.max_id][height_mask]
                temp_rgb = self.grid_rgb[:self.max_id][height_mask]
                np.save(self.memory_save_path+f"/grid_rgb_pos_floor_{current_floor}.npy", temp_pos)
                np.save(self.memory_save_path+f"/grid_rgb_floor_{current_floor}.npy", temp_rgb)
                
                del temp_pos
                del temp_rgb
                gc.collect()

    def imaginary(self, text_prompts, vis=False):

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            images = self.diffusion(
                text_prompts,
                prompt_3=text_prompts,
                # negative_prompt="",
                height=self.args.gen_height,
                width=self.args.gen_width,
                num_inference_steps=28,
                guidance_scale=7.0,
                max_sequence_length=512,
                num_images_per_prompt=self.args.imagenary_num
            )
        for i in range(self.args.imagenary_num):
            images.images[i].save(f'test_{i}.png')
        # image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

        return images
    
    def matching2D(self, obs_path, text_prompts):
        ref_img = Image.open(obs_path).convert('RGB')
        ref_img_tensor = self.transform(ref_img).unsqueeze(0).cuda()    
        
        query_img = self.imaginary(text_prompts)
        query_img_tensor = self.transform(query_img).unsqueeze(0).cuda()
        
        with torch.no_grad():
            ref_features_dict = self.dinov2.forward_features(ref_img_tensor)
            ref_features = ref_features_dict['x_norm_patchtokens'].squeeze(0)
            
            query_features_dict = self.dinov2.forward_features(query_img_tensor)
            query_features = query_features_dict['x_norm_patchtokens'].squeeze(0).mean(dim=0)
            
            similarities = F.cosine_similarity(ref_features.unsqueeze(1), query_features.unsqueeze(0), dim=-1).squeeze(1)
            similarities_2d = similarities.reshape(self.args.height // 14, self.args.width // 14)
            
            plot_token_matching(query_img, ref_img, similarities_2d)
            
            
    def initial_memory(self):
        count = 1
        while os.path.exists(self.memory_save_path):
            self.memory_save_path = self.memory_save_path.split("_")[0]
            self.memory_save_path = f"{self.memory_save_path}_{count}"
            count +=1  
        
        os.makedirs(self.memory_save_path)
        self.feat_path = self.memory_save_path+"/feat.h5df"
        # with h5py.File(self.memory_save_path, 'w') as f:
        #     f.create_group('voxels')
        print("memory init at:", self.memory_save_path)

    
    def get_total_token_count(self):
        total_tokens = 0
        group_pattern = re.compile(r'^grid_\d+_\d+_\d+$')

        with h5py.File(self.feat_path, 'r') as h5f:
            for group_name in h5f:
                if group_pattern.match(group_name):
                    group = h5f[group_name]
                    if 'features' in group:
                        num_tokens = group['features'].shape[0]
                        total_tokens += num_tokens
        print("total_tokens:", total_tokens)
    
    
    def update_memory_dist_base(self):
        print("updating memory...")
        t1 = time.time()

        with h5py.File(self.feat_path, 'a') as h5f:
            for i in range(self.grid_feat.shape[0]):
                grid_id = f'{self.grid_feat_pos[i][0]}_{self.grid_feat_pos[i][1]}_{self.grid_feat_pos[i][2]}'
                group_name = f'grid_{grid_id}'
                
                if group_name not in h5f:
                    group = h5f.create_group(group_name)
                    group.create_dataset('features', data=self.grid_feat[i:i+1], maxshape=(None, self.grid_feat.shape[1]), chunks=True)
                    group.create_dataset('distances', data=self.grid_feat_dis[i:i+1], maxshape=(None,), chunks=True)
                
                else:
                    group = h5f[group_name]
                    features = group['features']
                    distances = group['distances']    
                
                    if features.shape[0] < self.cache_size:
                        features.resize((features.shape[0] + 1, features.shape[1]))
                        distances.resize((distances.shape[0] + 1,))
                        features[-1] = self.grid_feat[i]
                        distances[-1] = self.grid_feat_dis[i]
                        
                    else:
                        remove_idx = random.choice(range(distances.shape[0]))
                        features[remove_idx] = self.grid_feat[i]
                        distances[remove_idx] = self.grid_feat_dis[i]
       
        print(f"finish updating, time:{time.time() - t1}")
        self.get_total_token_count()
        self._reinit_cache()
        
        
        
    
    
    def update_memory_surp_base(self):
        
        def get_neighbor_positions(pos):
            neighbors = []
            offsets = list(product(range(-self.neighbor_radius, self.neighbor_radius + 1),repeat=3))
            offsets.remove((0, 0, 0))
            for offset in offsets:
                neighbor_pos = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
                neighbors.append(neighbor_pos)
            return neighbors
        
        def compute_surprise(new_token, surrounding_tokens):
            if surrounding_tokens.numel() == 0:
                return float('inf')
            new_token_norm = new_token / new_token.norm()
            surrounding_tokens_norm = surrounding_tokens / surrounding_tokens.norm(dim=1, keepdim=True)
            cosine_sim = torch.mm(surrounding_tokens_norm, new_token_norm.unsqueeze(1)).squeeze(1)
            cosine_dist = 1 - cosine_sim

            surprise = cosine_dist.min().item()
            return surprise
        
        def forgetting_strategy(features, distances):
            if features.size(0) <= 1:
                    return features, distances

            # 归一化特征
            features_norm = features / features.norm(dim=1, keepdim=True)
            # 计算相似度矩阵
            similarity = torch.mm(features_norm, features_norm.t())
            # 排除自身的相似度
            mask = torch.ones_like(similarity, dtype=torch.bool)
            mask.fill_diagonal_(False)

            # 找到相似度高于阈值的所有对
            redundant = (similarity > self.boring_threshold).nonzero(as_tuple=False)

            # 使用并查集（Union-Find）算法来找到所有相似的组
            parent = list(range(features.size(0)))

            def find(u):
                while parent[u] != u:
                    parent[u] = parent[parent[u]]
                    u = parent[u]
                return u

            def union(u, v):
                pu, pv = find(u), find(v)
                if pu != pv:
                    parent[pv] = pu

            for i, j in redundant:
                union(i.item(), j.item())

            # 分组
            groups = defaultdict(list)
            for idx in range(features.size(0)):
                root = find(idx)
                groups[root].append(idx)

            # 计算每组的平均特征和距离
            new_features = []
            new_distances = []
            processed = set()

            for group in groups.values():
                if len(group) == 1:
                    idx = group[0]
                    new_features.append(features[idx])
                    new_distances.append(distances[idx])
                else:
                    # 计算特征的平均值
                    avg_feature = torch.mean(features[group], dim=0)
                    new_features.append(avg_feature)
                    # 选择距离的平均值（或其他策略，如最小值、最大值等）
                    avg_distance = torch.mean(distances[group], dim=0)
                    new_distances.append(avg_distance)

            # 将结果转换为张量
            new_features = torch.stack(new_features).to(self.device)
            new_distances = torch.stack(new_distances).to(self.device)

            return new_features, new_distances
        
        def add_or_replace_token(group, token, distance):
            features = torch.tensor(group['features'][:], device=self.device)  # 形状: (M, D)
            distances = torch.tensor(group['distances'][:], device=self.device)  # 形状: (M,)

            if features.shape[0] < self.cache_size:
                # 扩展数据集
                group['features'].resize((features.shape[0] + 1, features.shape[1]))
                group['distances'].resize((distances.shape[0] + 1,))
                group['features'][-1] = token
                group['distances'][-1] = distance
            else:
                # 替换最不惊奇的令牌（最小距离）
                features_norm = features / features.norm(dim=1, keepdim=True)
                token_tensor = torch.tensor(token, device=self.device, dtype=torch.float32)
                token_norm = token_tensor / token_tensor.norm()
                cosine_sim = torch.mm(features_norm, token_norm.unsqueeze(1)).squeeze(1)
                cosine_dist = 1 - cosine_sim
                replace_idx = torch.argmin(cosine_dist).item()
                group['features'][replace_idx] = token
                group['distances'][replace_idx] = distance
        
        
        def process_single_token(h5f, token, distance, pos):
            group_name = f'grid_{pos[0]}_{pos[1]}_{pos[2]}'
            if group_name not in h5f:
                group = h5f.create_group(group_name)
                group.create_dataset('features', data=token.reshape(1, -1),
                                    maxshape=(None, token.shape[0]), chunks=True)
                group.create_dataset('distances', data=np.array([distance]),
                                    maxshape=(None,), chunks=True)
                return  # 新创建的体素组已初始化，无需进一步处理

            group = h5f[group_name]

            # 获取邻近体素的位置
            neighbors = get_neighbor_positions(pos)
            surrounding_tokens = []

            for neighbor_pos in neighbors:
                neighbor_group_name = f'grid_{neighbor_pos[0]}_{neighbor_pos[1]}_{neighbor_pos[2]}'
                if neighbor_group_name in h5f:
                    neighbor_features = torch.tensor(h5f[neighbor_group_name]['features'][:], device=self.device)
                    surrounding_tokens.append(neighbor_features)

            if surrounding_tokens:
                surrounding_tokens = torch.cat(surrounding_tokens, dim=0)  # 形状: (M', D)
            else:
                surrounding_tokens = torch.empty((0, token.shape[0]), device=self.device)

            # 计算惊奇度
            surprise = compute_surprise(torch.tensor(token, device=self.device), surrounding_tokens)

            if surprise > self.surprise_threshold:
                # 添加或替换令牌
                add_or_replace_token(group, token, distance)

                # 重新加载更新后的特征和距离
                updated_features = torch.tensor(group['features'][:], device=self.device)
                updated_distances = torch.tensor(group['distances'][:], device=self.device)

                # 应用遗忘策略
                updated_features, updated_distances = forgetting_strategy(updated_features, updated_distances)

                # 截断至 cache_size
                if updated_features.size(0) > self.cache_size:
                    updated_features = updated_features[:self.cache_size]
                    updated_distances = updated_distances[:self.cache_size]

                # 更新数据集
                group['features'].resize((updated_features.size(0), updated_features.size(1)))
                group['features'][:] = updated_features.cpu().numpy()
                group['distances'].resize((updated_distances.size(0),))
                group['distances'][:] = updated_distances.cpu().numpy()
        
        print("updating memory...")
        t1 = time.time()

        B, D = self.grid_feat.shape

        with h5py.File(self.feat_path, 'a') as h5f:
            for i in range(B):
                token = self.grid_feat[i]
                distance = self.grid_feat_dis[i].item()
                pos = tuple(self.grid_feat_pos[i].tolist())
                process_single_token(h5f, token, distance, pos)

        print(f"finish updating, time:{time.time() - t1}")
        self.get_total_token_count()
        self._reinit_cache()
        
    
    def read_memory(self):
        pass
    
    def Quantizing(self, model_id):
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

        self.diffusion = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            transformer=model_nf4,
            torch_dtype=torch.bfloat16
        )
        self.diffusion.enable_model_cpu_offload()
        
    
    def voxel_localized(self, text_prompt, K=100, batch_size=300):

        # self.diffusion = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
        self.Quantizing(self.args.diffusion_id)
        self.diffusion = self.diffusion.to(self.device)

        gen_results = self.imaginary(text_prompt)  # 修正为单个 text_prompt
        query_imgs = [gen_results.images[i] for i in range(self.args.imagenary_num)]
                
        del self.diffusion
        gc.collect()
        torch.cuda.empty_cache()
        
        query_img_tensors = torch.stack([self.transform(img) for img in query_imgs]).to(self.device)  # [batch_size, C, H, W]
        
        candidates = []
        
        print("localizing...")
        t1 = time.time()
        
        with torch.no_grad():
            # query_features_dict = self.dinov2.forward_features(query_img_tensors)
            # query_features = query_features_dict['x_norm_patchtokens'].squeeze(0).mean(dim=0)
            # query_features = query_features.unsqueeze(0)
            query_features_dict = self.dinov2.forward_features(query_img_tensors)
            tokens = query_features_dict['x_norm_patchtokens']  # 假设形状为 [batch_size, num_tokens, feature_dim]
            batch_size, num_tokens, feature_dim = tokens.size()
            grid_size = int(math.sqrt(num_tokens))
            
            
            xs = torch.arange(grid_size, device=self.device).repeat(grid_size).view(1, num_tokens)
            ys = torch.arange(grid_size, device=self.device).repeat_interleave(grid_size).view(1, num_tokens)
            center = (grid_size - 1) / 2
            distances = (xs - center) ** 2 + (ys - center) ** 2  # [1, num_tokens]
            sigma = (grid_size / 2) ** 2
            weights = torch.exp(-distances / (2 * sigma))  # [1, num_tokens]
            weights = weights / weights.sum(dim=1, keepdim=True)  # [1, num_tokens]
            weights = weights.unsqueeze(-1)  # [1, num_tokens, 1]
            
            weighted_tokens = tokens * weights  # [batch_size, num_tokens, feature_dim]
            weighted_sum = weighted_tokens.sum(dim=1)  # [batch_size, feature_dim]
            query_features = weighted_sum.mean(dim=0).unsqueeze(0)
            
            # with h5py.File(self.feat_path, 'r') as h5f:
            #     for group_name in h5f.keys():
            #         group = h5f[group_name]
            #         grid_feat = torch.from_numpy(group['features'][:]).to(self.device)
            #         similarities = F.cosine_similarity(query_features, grid_feat, dim=1)
            #         batch_max_similarity, max_idx = torch.max(similarities, dim=0)
            #         batch_max_similarity = batch_max_similarity.item()
                    
            #         pos = [int(group_name.split('_')[1]), 
            #                int(group_name.split('_')[2]), 
            #                int(group_name.split('_')[3])]
            #         candidates.append((batch_max_similarity, pos))
            
            with h5py.File(self.feat_path, 'r') as h5f:
                group_names = list(h5f.keys())

                if self.args.load_single_floor:      
                    print(f'filter height from {self.floor_min_height} to {self.floor_max_height}')  
                    filtered_group_names = []
                    for group_name in group_names:
                        _, x, y, z = group_name.split('_')  # 假设 group_name 形如 "group_x_y_z"
                        if self.floor_min_height <= int(z) <= self.floor_max_height:
                            filtered_group_names.append(group_name)
                    group_names = filtered_group_names

                for i in range(0, len(group_names), batch_size):
                    batch_group_names = group_names[i:i + batch_size]

                    all_features = []
                    group_positions = []

                    for group_name in batch_group_names:
                        group = h5f[group_name]
                        grid_feat = torch.from_numpy(group['features'][:]).to(self.device)
                        all_features.append(grid_feat)
                        position = tuple(map(int, group_name.split('_')[1:4]))
                        group_positions.append((position, grid_feat.shape[0]))

                    all_features = torch.cat(all_features, dim=0)
                    similarities = F.cosine_similarity(query_features, all_features, dim=1)

                    start_idx = 0
                    for position, count in group_positions:
                        group_similarities = similarities[start_idx:start_idx + count]
                        max_similarity, _ = torch.max(group_similarities, dim=0)
                        candidates.append((max_similarity.item(), position))
                        start_idx += count
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_k_positions = [pos for _, pos in candidates[:K]]
        top_k_similarity = [sim for sim, _ in candidates[:K]]
        
        print(f"finish localizing, time:{time.time() - t1}")
        
        return np.array([top_k_positions[0]]), np.array(top_k_positions), np.array(top_k_similarity)
        
        # grid_feat = torch.from_numpy(self.grid_feat[:self.max_id]).to(self.device)
        
        # query_img = self.imaginary(text_prompts)
        # query_img_tensor = self.transform(query_img).unsqueeze(0).cuda()
        
        # with torch.no_grad():
        #     query_features_dict = self.dinov2.forward_features(query_img_tensor)
        #     query_features = query_features_dict['x_norm_patchtokens'].squeeze(0).mean(dim=0)
            
        #     query_features = query_features.unsqueeze(0)
        #     similarities = F.cosine_similarity(query_features, grid_feat, dim=1)
        #     max_index = torch.argmax(similarities).item()
        #     best_pos = self.grid_pos[max_index]
        #     best_rgb = self.grid_rgb[max_index]
        
        
        # np.save("grid_pos.npy", self.grid_pos[:self.max_id])
        # np.save("grid_rgb.npy", self.grid_rgb[:self.max_id])
        # np.save("best_pos.npy", np.array([best_pos]))
        
    def long_memory_filter(self):
        
        if self.args.load_single_floor:      
            print(f'filter height from {self.floor_min_height} to {self.floor_max_height}') 

            filter_long_memory = [
                obj for obj in self.long_memory_dict
                if self.floor_min_height <= obj["loc"][2] <= self.floor_max_height
            ]

            return filter_long_memory
        else:
            return self.long_memory_dict

    
    def _init_cache(self):

        grid_feat = np.zeros((self.iter_size, self.token_dim), dtype=np.float32)
        grid_feat_pos = np.zeros((self.iter_size, 3), dtype=np.int32)
        grid_feat_dis = np.zeros((self.iter_size), dtype=np.float32)
        iter_id = 0
        
        grid_rgb_pos = np.zeros((self.gs * self.gs, 3), dtype=np.int32)
        occupied_ids = -1 * np.ones((self.gs, self.gs, self.maxh-self.minh), dtype=np.int32)
        weight = np.zeros((self.gs * self.gs), dtype=np.float32)
        grid_rgb = np.zeros((self.gs * self.gs, 3), dtype=np.uint8)
        max_id = 0

        base_height = []
        return grid_feat, grid_feat_pos, grid_rgb_pos, grid_feat_dis, weight, occupied_ids, grid_rgb, max_id, iter_id, base_height
    
    def _reinit_cache(self):
        
        self.grid_feat = np.zeros((self.iter_size, self.token_dim), dtype=np.float32)
        self.grid_feat_pos = np.zeros((self.iter_size, 3), dtype=np.int32)
        self.grid_feat_dis = np.zeros((self.iter_size), dtype=np.float32)
        self.iter_id = 0
    
    
    def _get_patch_token(self, img):
        img = torch.from_numpy(img).to(self.device).unsqueeze(0)
        img = img.permute(0, 3, 1, 2).float() / 255
        img = self.transform_(img)
        
        with torch.no_grad():
            img_features_dict = self.dinov2.forward_features(img)
            patch_token = img_features_dict['x_norm_patchtokens'].squeeze(0)
            patch_token = patch_token.reshape(self.n_patch_w, self.n_patch_h, -1)
            
        return patch_token
    
    def _backproject_depth(self, depth):
        
        pc, mask = depth2pc(depth, intr_mat=self.calib_mat, min_depth=self.min_depth, max_depth=self.max_depth)  # (3, N)
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::self.depth_sample_rate]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pc = pc[:, mask]
        return pc
    
    def _out_of_range(self, row, col, height):        
        return col >= self.gs or row >= self.gs or height >= self.maxh or col < 0 or row < 0 or height < self.minh
    
    def draw_detect(self, bgr_):
        # 随机生成颜色
        num_boxes = sum(len(result['boxes']) for result in self.gdino_results)
        colors = np.random.randint(0, 255, size=(num_boxes, 3), dtype=np.uint8)
        color_index = 0

        for result in self.gdino_results:
            scores = result['scores']
            labels = result['labels']
            boxes = result['boxes']

            for score, label, box in zip(scores, labels, boxes):
                # 获取边界框坐标
                x1, y1, x2, y2 = map(int, box)

                # 选择颜色
                color = tuple(map(int, colors[color_index]))
                color_index += 1

                # 绘制边界框
                cv2.rectangle(bgr_, (x1, y1), (x2, y2), color, 2)

                # 绘制标签和置信度
                text = f"{label}: {score:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(bgr_, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
                cv2.putText(bgr_, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return bgr_  
    
    def _show_map_online(self, obs):
        if self.args.no_vis:
            return
        # 获取 RGB 图像并转换为 BGR 格式
        bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)

        # 获取深度图像，并归一化到 0-255 范围以便显示
        depth = obs["depth"]
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # 获取语义图像，并将其转换为彩色图像
        semantic_obs = obs["semantic"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_colored = np.array(semantic_img.convert("RGB"))
        semantic_colored = cv2.cvtColor(semantic_colored, cv2.COLOR_RGB2BGR)
        # cv map
        
        cv_map = cv2.cvtColor(self.cv_map, cv2.COLOR_RGB2BGR)
        FrontierMap = cv2.cvtColor(self.FrontierMap, cv2.COLOR_RGB2BGR)
        # detect result
        # detect = bgr.copy()
        # if len(self.gdino_results) != 0:
        #     detect = self.draw_detect(detect)
        if len(self.yolow_results) != 0:
            detect = self.yolow_results[0].plot()
            detect = cv2.cvtColor(detect, cv2.COLOR_RGB2BGR)
        else:
            detect = bgr.copy()

        # 将三种图像调整为相同大小
        h, w = bgr.shape[:2]
        depth_resized = cv2.resize(depth_normalized, (w, h))
        semantic_resized = cv2.resize(semantic_colored, (w, h))
        cv_map_resized = cv2.resize(cv_map, (w, h))
        FrontierMap_resized = cv2.resize(FrontierMap, (w, h))
        detect_resized = cv2.resize(detect, (w, h))
        
        # 合并 RGB、深度和语义图像
        combined_image = np.hstack((bgr, cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR), cv_map_resized, FrontierMap_resized, detect_resized))

        # 调整窗口大小，使其小于屏幕宽度
        screen_width = 6200  # 假设屏幕宽度为 800 像素
        scale_factor = min(screen_width / combined_image.shape[1], 1.0)
        new_width = int(combined_image.shape[1] * scale_factor)
        new_height = int(combined_image.shape[0] * scale_factor)
        combined_image_resized = cv2.resize(combined_image, (new_width, new_height))
        
        # 显示拼接图像
        cv2.imshow("RGB | Depth | Semantic | cvmap | FrontierMap | detect", combined_image_resized)
        
    
    def obs2voxeltoken(self, obs, pose):
        
        if len(self.inv_init_base_tf) == 0:
            self.init_base_tf = cvt_pose_vec2tf(pose)
            self.init_base_tf = self.base_transform @ self.init_base_tf @ np.linalg.inv(self.base_transform)
            self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        
        habitat_base_pose = cvt_pose_vec2tf(pose)
        base_pose = self.base_transform @ habitat_base_pose @ np.linalg.inv(self.base_transform)
        self.tf = self.inv_init_base_tf @ base_pose
        
        rgb = np.array(obs["rgb"][:,:,:3])
        depth = np.array(obs["depth"])
        
        patch_tokens = self._get_patch_token(rgb)
        patch_tokens_intr = get_sim_cam_mat(patch_tokens.shape[0], patch_tokens.shape[1])
        
        pc = self._backproject_depth(depth)
        pc_transform = self.tf @ self.base_transform @ self.base2cam_tf
        pc_global = transform_pc(pc, pc_transform) 
        
        for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
            row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p[0], p[1], p[2])
            if self._out_of_range(row, col, height):
                continue
            height = height - self.minh
            
            px, py, pz = project_point(self.calib_mat, p_local)
            rgb_v = rgb[py, px, :]
            px, py, pz = project_point(patch_tokens_intr, p_local)
            
            radial_dist_sq = np.sum(np.square(p_local))
            sigma_sq = 0.6
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))
                
            
            if not (px < 0 or py < 0 or px >= patch_tokens.shape[1] or py >= patch_tokens.shape[0]):
                
                if self.iter_id >= self.iter_size:
                    self.update_memory_dist_base()
                else:
                    self.grid_feat[self.iter_id, :] = patch_tokens[py, px, :].cpu().numpy()
                    self.grid_feat_pos[self.iter_id, :] = [row, col, height]
                    self.grid_feat_dis[self.iter_id] = radial_dist_sq
                    self.iter_id += 1
                
                occupied_id = self.occupied_ids[row, col, height]
                if occupied_id == -1:
                    self.occupied_ids[row, col, height] = self.max_id
                    self.grid_rgb[self.max_id] = rgb_v
                    self.weight[self.max_id] += alpha
                    self.grid_rgb_pos[self.max_id] = [row, col, height]
                    self.max_id += 1
                else:
                    self.grid_rgb[occupied_id] = (self.grid_rgb[occupied_id] * self.weight[occupied_id] + rgb_v * alpha) / (
                            self.weight[occupied_id] + alpha
                        )
                    self.weight[occupied_id] += alpha

                if height >= self.max_height[row, col]:
                    self.max_height[row, col] = height
                    self.cv_map[row, col] = rgb_v
                
    def long_memory(self, obs):
        
        self.yolow_results = self.yolow.predict(Image.fromarray(obs["rgb"][:,:,:3]), conf=self.args.detect_conf)
        if len(self.yolow_results[0].boxes) > 0:
            depth = np.array(obs["depth"])
            pc, mask = depth2pc(depth, intr_mat=self.calib_mat, min_depth=self.min_depth, max_depth=self.max_depth)
            center = []
            confs = []
            labels = []
            
            for i in range(len(self.yolow_results[0].boxes.conf)):
                xyxy = self.yolow_results[0].boxes.xyxy[i].cpu().numpy()
                col = int((xyxy[0] + xyxy[2])/2)
                row = int((xyxy[1] + xyxy[3])/2)
                index = row * self.args.width + col
                
                if mask[index]:
                    center.append(index)
                    confs.append(self.yolow_results[0].boxes.conf[i].item())
                    cls = int(self.yolow_results[0].boxes.cls[i].item())
                    labels.append(self.args.detect_classes[cls])
                    
            if len(center) != 0:
                center = np.array(center)
                pc = pc[:, center]
                pc_transform = self.tf @ self.base_transform @ self.base2cam_tf
                pc_global = transform_pc(pc, pc_transform) 
                
                for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
                    row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p[0], p[1], p[2])
                    if self._out_of_range(row, col, height):
                        continue
                    height = height - self.minh
                    instance = {
                        'label': labels[i],
                        'loc': [row, col, height],
                        'confidence': confs[i]
                        }
                    self.long_memory_dict.append(instance)
                    
        self.long_memory_integration()
        
        # inputs = self.gdino_processor(images=Image.fromarray(obs["rgb"][:,:,:3]), text=self.args.detect_classes, return_tensors="pt").to(self.device)
        # with torch.no_grad():
        #     outputs = self.gdino(**inputs)

        # self.gdino_results = self.gdino_processor.post_process_grounded_object_detection(
        #     outputs,
        #     inputs.input_ids,
        #     box_threshold=0.5,
        #     text_threshold=0.3,
        #     target_sizes=[Image.fromarray(obs["rgb"][:,:,:3]).size[::-1]]
        # )
        
        # if len(self.gdino_results[0]['labels']) > 0:
        #     depth = np.array(obs["depth"])
        #     pc, mask = depth2pc(depth, intr_mat=self.calib_mat, min_depth=self.min_depth, max_depth=self.max_depth)
        #     center = []
        #     for result in self.gdino_results:
        #         scores = result['scores']
        #         labels = result['labels']
        #         boxes = result['boxes']
                
        #         for score, label, box in zip(scores, labels, boxes):
        #             x1, y1, x2, y2 = map(int, box)
        #             col = int((x1 + x2)/2)
        #             row = int((y1 + y2)/2)
        #             index = row * self.args.width + col
        #             if mask[index]:
        #                 center.append(row * self.args.width + col)

        #     if len(center) != 0:
        #         center = np.array(center)
        #         pc = pc[:, center]
        #         pc_transform = self.tf @ self.base_transform @ self.base2cam_tf
        #         pc_global = transform_pc(pc, pc_transform) 
                
        #         for i, (p, p_local) in enumerate(zip(pc_global.T, pc.T)):
        #             row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p[0], p[1], p[2])
        #             if self._out_of_range(row, col, height):
        #                 continue
        #             instance = {
        #                 'label': labels[i],
        #                 'loc': [row, col, height],
        #                 'confidence': scores[i].item()
        #                 }
        #             self.long_memory_dict.append(instance)
    
    def long_memory_integration(self, threshold=3):
        
        def l1_distance(loc1, loc2):
            return sum(abs(a - b) for a, b in zip(loc1, loc2))
        # 根据label分组
        label_groups = {}
        for item in self.long_memory_dict:
            label = item["label"]
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(item)

        final_results = []

        for label, items in label_groups.items():
            # 用来存放筛选后的条目
            filtered = []
            for itm in items:
                merged = False
                for f in filtered:
                    # 判断是否在距离阈值以内
                    if l1_distance(f['loc'], itm['loc']) <= threshold:
                        # 如果在距离阈值以内，保留 confidence 最大的条目
                        if itm['confidence'] > f['confidence']:
                            f['loc'] = itm['loc']
                            f['confidence'] = itm['confidence']
                        merged = True
                        break
                if not merged:
                    filtered.append(itm)

            final_results.extend(filtered)
        self.long_memory_dict = final_results

    def create_memory(self):
        self.initial_memory()

        last_action = None
        release_count = 0
        obs = self.Env.sims.get_sensor_observations(0)
        step_num = 0
        while True:
            self._show_map_online(obs)
            k, action = keyboard_control_fast()
            if k != -1:
                if action == "stop":
                    break
                if action == "record":
                    init_agent_state = self.sim.get_agent(0).get_state()
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

            step_num += 1
            obs = self.Env.sims.step(action)
            agent_state = self.Env.agent.get_state()
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
            if step_num % 10 == 0:
                self.base_height.append(agent_state.position[1])

            pos, rot = agent_state.position, agent_state.rotation
            pose = np.array([pos[0], pos[1], pos[2], rot.x, rot.y, rot.z, rot.w])
            
            self.obs2voxeltoken(obs, pose)
            self.long_memory(obs)
            
        cv2.destroyAllWindows()
        # self._visualize_rgb_map_3d(self.grid_pos, self.grid_rgb)
        np.save(self.memory_save_path+"/grid_rgb_pos.npy", self.grid_rgb_pos[:self.max_id])
        np.save(self.memory_save_path+"/grid_rgb.npy", self.grid_rgb[:self.max_id])
        np.save(self.memory_save_path+"/weight.npy", self.weight[:self.max_id])
        np.save(self.memory_save_path+"/occupied_ids.npy", self.occupied_ids)
        np.save(self.memory_save_path+"/max_id.npy", np.array(self.max_id))
        np.save(self.memory_save_path+"/original_pos.npy", self.Env.original_state.position)
        np.save(self.memory_save_path+"/map_height.npy", np.array([self.minh, self.maxh]))
        np.save(self.memory_save_path+"/base_height.npy", np.array(self.base_height))
        with open(self.memory_save_path+"/long_memory.json", 'w') as f:
            json.dump(self.long_memory_dict, f, indent=4)

    
    def excute(self, obs, actions):
        for action in actions:
            if action != "stop":
                self._show_map_online(obs)
                obs = self.Env.sims.step(action)
                agent_state = self.Env.agent.get_state()
                print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
                pos, rot = agent_state.position, agent_state.rotation
                pose = np.array([pos[0], pos[1], pos[2], rot.x, rot.y, rot.z, rot.w])
                
                self.obs2voxeltoken(obs, pose)
                self.long_memory(obs)

                cv2.waitKey(1)
        
        return obs
    
    
    def exploring_create_memory(self):
        self.initial_memory()
        obs = self.Env.sims.get_sensor_observations(0)
        agent_state = self.Env.agent.get_state()
        self.init_height = agent_state.position[1]
        # random_move
        for _ in tqdm(range(self.args.random_move_num)):
            
            subgoal = self.Env.plnner.pathfinder.get_random_navigable_point()
            island_goal = self.Env.plnner.pathfinder.get_island(subgoal)
            island_begin = self.Env.plnner.pathfinder.get_island(self.Env.agent.get_state().position)

            while (not self.Env.plnner.pathfinder.is_navigable(subgoal)) or (island_goal != island_begin):
                subgoal = self.Env.plnner.pathfinder.get_random_navigable_point()
                island_goal = self.Env.plnner.pathfinder.get_island(subgoal)

            try:
                path, goal = self.Env.move2point(subgoal)
                obs = self.excute(obs, path) 
                self.base_height.append(self.Env.agent.get_state().position[1])
                action_around = ['turn_left'] * int(360 / self.args.turn_left)
                obs = self.excute(obs, action_around)        
            except Exception as e:
                print(f"移动失败: {e}")
                continue
  
            
                    
        
        cv2.destroyAllWindows()
        # self._visualize_rgb_map_3d(self.grid_pos, self.grid_rgb)
        self.update_memory_dist_base()
        np.save(self.memory_save_path+"/grid_rgb_pos.npy", self.grid_rgb_pos[:self.max_id])
        np.save(self.memory_save_path+"/grid_rgb.npy", self.grid_rgb[:self.max_id])
        np.save(self.memory_save_path+"/weight.npy", self.weight[:self.max_id])
        np.save(self.memory_save_path+"/occupied_ids.npy", self.occupied_ids)
        np.save(self.memory_save_path+"/max_id.npy", np.array(self.max_id))
        np.save(self.memory_save_path+"/original_pos.npy", self.Env.original_state.position)
        np.save(self.memory_save_path+"/map_height.npy", np.array([self.minh, self.maxh]))
        np.save(self.memory_save_path+"/base_height.npy", np.array(self.base_height))
        with open(self.memory_save_path+"/long_memory.json", 'w') as f:
            json.dump(self.long_memory_dict, f, indent=4)
    
    # FrontierExplorer
    def grid2loc_2d(self, x, y):
        row, col = x, y
        initial_position = self.Env.original_state.position  # [x, z, y]
        initial_x, initial_z, initial_y = initial_position

        actual_y = initial_y + (row - self.gs // 2) * self.cs
        actual_x = initial_x + (col - self.gs // 2) * self.cs
        actual_z = initial_z
        
        # 返回实际坐标
        return np.array([actual_x, actual_z, actual_y])
    
    def loc2grid_2d(self, x_base, y_base):
        row = int(self.gs / 2 - int(x_base / self.cs))
        col = int(self.gs / 2 - int(y_base / self.cs))
        return row, col
    
    def is_unknown(self, x: int, y: int) -> bool:
        return (self.cv_map[x, y].sum() == 0)   
    def is_known(self, x: int, y: int) -> bool:
        return not self.is_unknown(x, y)     
    def in_bounds(self, x: int, y: int) -> bool:
        return (0 <= x < self.gs and 0 <= y < self.gs)
    def is_navigabale(self, x: int, y: int) -> bool:
        return self.Env.plnner.pathfinder.is_navigable(self.grid2loc_2d(x,y))    
        
    def build_navigable_mask(self) -> np.ndarray:
        """
        根据 self.map_3d 构建一个 [gs, gs] 的布尔数组，表示哪些网格可导航。
        """
        navigable_mask = np.zeros((self.gs, self.gs), dtype=bool)
        for x in range(self.gs):
            for y in range(self.gs):
                # 如果某格是已知，则认为可导航
                if self.is_known(x, y) and self.is_navigabale(x, y):
                    navigable_mask[x, y] = True
        return navigable_mask
    
    def find_frontiers(self, navigable_mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        寻找当前地图中的所有前沿点(网格坐标)。
        前沿点定义：已知+可导航，且与至少一个未知邻居相邻。
        :return: list of (x, y)，表示前沿点的网格坐标
        """
        frontiers = []
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for x in range(self.gs):
            for y in range(self.gs):
                if not navigable_mask[x, y]:
                    continue
                if self.is_known(x, y):
                    # 判断是否与未知邻居相邻
                    neighbors_unknown = False
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if self.in_bounds(nx, ny) and self.is_unknown(nx, ny):
                            neighbors_unknown = True
                            break
                    if neighbors_unknown:
                        frontiers.append((x, y))
        return frontiers    
    
    def cluster_frontiers(self, frontiers: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        对前沿点进行连通性聚类，返回若干个簇，每个簇是一组前沿点 (x, y)。
        使用4邻域BFS。
        """
        if not frontiers:
            return []

        frontier_set = set(frontiers)
        visited = set()
        clusters = []

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for f in frontiers:
            if f in visited:
                continue
            # BFS
            queue = deque([f])
            cluster = []
            visited.add(f)

            while queue:
                cx, cy = queue.popleft()
                cluster.append((cx, cy))
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if (nx, ny) in frontier_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

            clusters.append(cluster)
            
        filtered_clusters = []
        for c in clusters:
            if len(c) >= self.min_cluster_size:
                filtered_clusters.append(c)

        return filtered_clusters

    
    def compute_cluster_center(self, cluster: List[Tuple[int, int]]) -> Tuple[float, float]:
        """
        计算前沿簇的中心（例如质心），并返回 (cx, cy) 浮点数网格坐标。
        """
        cx = sum([p[0] for p in cluster]) / len(cluster)
        cy = sum([p[1] for p in cluster]) / len(cluster)
        return (cx, cy)
    
    def compute_information_gain(self, center_x: float, center_y: float) -> float:
        """
        在网格坐标系下，以 (center_x, center_y) 为中心，查看半径 ig_radius 范围内未知格子数量，
        作为信息增益近似值。
        """
        cx = int(round(center_x))
        cy = int(round(center_y))

        unknown_count = 0
        radius = self.ig_radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx = cx + dx
                ny = cy + dy
                if not self.in_bounds(nx, ny):
                    continue
                if self.is_unknown(nx, ny):
                    unknown_count += 1

        return float(unknown_count)
    
    def select_best_cluster_center_by_ig(
        self,
        frontier_clusters: List[List[Tuple[int, int]]]
    ) -> Tuple[float, float]:
        """
        基于信息增益来选择最优的前沿簇中心。
        具体做法：
          1. 对每个簇，先计算其网格质心 (cx, cy)
          2. 估算信息增益: compute_information_gain(cx, cy)
          3. 选出信息增益最高的簇中心。
        
        如果所有簇都0信息增益,可以返回None表示没有值得去的前沿点。
        """
        best_center = None
        best_ig = 0.0

        for cluster in frontier_clusters:
            cx, cy = self.compute_cluster_center(cluster)
            ig = self.compute_information_gain(cx, cy)
            if ig > best_ig:
                best_ig = ig
                best_center = (cx, cy)

        # 如果信息增益全是 0，则返回 None 代表没有价值的探索目标
        if best_center is None or best_ig == 0:
            return None

        return best_center

    def update_frontier_map(self, frontiers, frontier_clusters, target_center_map):
        frontier_img = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)

        # 1) 已知/未知/可导航 上色
        for x in range(self.gs):
            for y in range(self.gs):
                if self.is_unknown(x, y):
                    frontier_img[x, y] = (0, 0, 0)           # 黑:未知
                else:
                    if self.is_navigabale(x, y):
                        frontier_img[x, y] = (255, 255, 255) # 白:已知可导航
                    else:
                        frontier_img[x, y] = (100, 100, 100) # 灰:已知不可导航
        
        # 2) 标记前沿点 (红色)
        for (fx, fy) in frontiers:
            frontier_img[fx, fy] = (255, 0, 0)

        # # 3) 随机颜色标记前沿簇
        # for cluster in frontier_clusters:
        #     color = (
        #         random.randint(50, 255),
        #         random.randint(50, 255),
        #         random.randint(50, 255)
        #     )
        #     for (cx, cy) in cluster:
        #         frontier_img[cx, cy] = color

        # 4) 标记目标点(绿色)
        if target_center_map is not None:
            tx = int(round(target_center_map[0]))
            ty = int(round(target_center_map[1]))
            if 0 <= tx < self.gs and 0 <= ty < self.gs:
                cv2.circle(frontier_img, (ty, tx), 5, (0, 255, 0), -1)

        self.FrontierMap = frontier_img
    
    
    def explore_entire_space(self, max_iterations=30):
        """
        前沿点探索的主方法：
        不断寻找前沿点 -> 聚类 -> 选目标 -> 导航，直到前沿点耗尽或达到最大迭代次数。
        """
        iteration_count = 0
        self.min_cluster_size = 10 
        self.ig_radius = 5
        
        self.initial_memory()
        obs = self.Env.sims.get_sensor_observations(0)

        while iteration_count < max_iterations:
            iteration_count += 1
            
            action_around = ['turn_left'] * int(360 / self.args.turn_left)
            obs = self.excute(obs, action_around)
            
            navigable_mask = self.build_navigable_mask()
            frontiers = self.find_frontiers(navigable_mask)
            if not frontiers:
                break

            frontier_clusters = self.cluster_frontiers(frontiers)
            if not frontier_clusters:
                break
            
            agent_state = self.Env.agent.get_state()
            robot_pos_world = [agent_state.position[0],  agent_state.position[2]]
            target_center_map = self.select_best_cluster_center_by_ig(frontier_clusters)
            if target_center_map is None:
                break
            self.update_frontier_map(frontiers, frontier_clusters, target_center_map)
            
            
            subgoal = self.grid2loc_2d(target_center_map[0], target_center_map[1])
            subgoal = self.Env.get_random_navigable_point_near(subgoal)
            
            path, goal = self.Env.move2point(subgoal)
            obs = self.excute(obs, path)        
            
            
     
        
if __name__ == "__main__":
    
    args = get_args()
    memory = VoxelTokenMemory(args)
    # memory.matching2D(obs_path='/home/orbit-new/桌面/orbit/shouwei_GES/VLDMap/Vlmaps/DistributionMap_dynamic/vlmaps/data/vlmaps_dataset/5q7pvUzZiYa_1/rgb/000307.png',
    #     text_prompts="A potted plant featuring a round blue vase topped with a white bouquet of a handful of green leaves.")
    memory.create_memory()
    # memory.exploring_create_memory()
    # memory.explore_entire_space()
    
    # memory.localized(text_prompts='A large oven, in the center of the cabinet.')
    # A purple vase with a bouquet of flowers on a kitchen countertop
    # A TV screen which is located in the cabinet.
    # A full watermelon on a blank background
    # grid_pos = np.load("grid_pos.npy")
    # grid_rgb = np.load("grid_rgb.npy")
    
    # memory._visualize_rgb_map_3d(grid_pos, grid_rgb)
        