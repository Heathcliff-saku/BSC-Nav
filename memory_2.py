import numpy as np
import pyarrow.feather as feather
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline
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

class VoxelTokenMemory():
    def __init__(self, args):
        self.args = args
        self.device = 'cuda'
        self.dinov2 = torch.hub.load('facebookresearch/dinov2', self.args.dino_size, source='github').to(self.device)

        self.memory_save_path = os.path.join(self.args.memory_path, self.args.scene_name)
        self.transform = transforms.Compose([
                    transforms.Resize((self.args.height, self.args.width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
                ])
        self.transform_ = transforms.Compose([
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet的均值和标准差
        ])
       
        self.Env = NavEnv(self.args)
        self.inv_init_base_tf = []
        
        self.patch_h = 14
        self.patch_w = 14
        self.n_patch_w = self.args.width // self.patch_w
        self.n_patch_h = self.args.height // self.patch_h
        
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
        
        self.calib_mat = get_sim_cam_mat_with_fov(self.args.height, self.args.width, fov=90)
        # self.calib_mat = np.array(self.args.cam_calib_mat).reshape((3, 3))
        self.min_depth = self.args.min_depth
        self.max_depth = self.args.max_depth
        
        self.token_dim = 1024
        # update memory when reach iter_size 
        self.iter_size = 10000
        # the max token of each voxel
        self.cache_size = 30
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
         self.iter_id
        ) = self._init_cache()
        
        
    def imaginary(self, text_prompts, vis=False):
        self.diffusion = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16)
        self.diffusion = self.diffusion.to(self.device)
        
        image = self.diffusion(
            text_prompts,
            negative_prompt="",
            height=224,
            width=224,
            num_inference_steps=28,
            guidance_scale=7.0
        ).images[0]
        image.save('test.png')
        # image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        
        del self.diffusion
        torch.cuda.empty_cache()
        return image
    
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

    
    def update_memory(self):
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
        self._reinit_cache()
    
    
    
    def read_memory(self):
        pass
    
    
    def localized(self, text_prompts):
        print("localizing...")
        t1 = time.time()
        
        max_similarity = -1
        best_pos = None
    
        query_img = self.imaginary(text_prompts)
        query_img_tensor = self.transform(query_img).unsqueeze(0).cuda()
        
        with torch.no_grad():
            query_features_dict = self.dinov2.forward_features(query_img_tensor)
            query_features = query_features_dict['x_norm_patchtokens'].squeeze(0).mean(dim=0)
            query_features = query_features.unsqueeze(0)
            
            with h5py.File(self.feat_path, 'r') as h5f:
                for group_name in h5f.keys():
                    group = h5f[group_name]
                    grid_feat = torch.from_numpy(group['features'][:]).to(self.device)
                    similarities = F.cosine_similarity(query_features, grid_feat, dim=1)
                    batch_max_similarity = torch.max(similarities).item()
                    
                    if batch_max_similarity > max_similarity:
                        max_similarity = batch_max_similarity
                        best_pos = [int(group_name.split('_')[1]), int(group_name.split('_')[2]), int(group_name.split('_')[3])]
        
        np.save(self.memory_save_path+"/best_pos.npy", np.array([best_pos]))
        print(f"finish localizing, time:{time.time() - t1}")
        
        return np.array([best_pos])
        
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
        return grid_feat, grid_feat_pos, grid_rgb_pos, grid_feat_dis, weight, occupied_ids, grid_rgb, max_id, iter_id
    
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
    
    def _show_map_online(self, obs):
        # 获取 RGB 图像并转换为 BGR 格式
        bgr = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_RGB2BGR)

        # 获取深度图像，并归一化到 0-255 范围以便显示
        depth = obs["depth_sensor"]
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # 获取语义图像，并将其转换为彩色图像
        semantic_obs = obs["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_colored = np.array(semantic_img.convert("RGB"))
        semantic_colored = cv2.cvtColor(semantic_colored, cv2.COLOR_RGB2BGR)
        
        # cv map
        
        cv_map = cv2.cvtColor(self.cv_map, cv2.COLOR_RGB2BGR)

        # 将三种图像调整为相同大小
        h, w = bgr.shape[:2]
        depth_resized = cv2.resize(depth_normalized, (w, h))
        semantic_resized = cv2.resize(semantic_colored, (w, h))
        cv_map_resized = cv2.resize(cv_map, (w, h))

        # 合并 RGB、深度和语义图像
        combined_image = np.hstack((bgr, cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR), semantic_resized, cv_map_resized))

        # 调整窗口大小，使其小于屏幕宽度
        screen_width = 6200  # 假设屏幕宽度为 800 像素
        scale_factor = min(screen_width / combined_image.shape[1], 1.0)
        new_width = int(combined_image.shape[1] * scale_factor)
        new_height = int(combined_image.shape[0] * scale_factor)
        combined_image_resized = cv2.resize(combined_image, (new_width, new_height))
        
        # 显示拼接图像
        cv2.imshow("RGB | Depth | Semantic | cvmap", combined_image_resized)
    
    
    def obs2voxeltoken(self, obs, pose):
        
        if len(self.inv_init_base_tf) == 0:
            self.init_base_tf = cvt_pose_vec2tf(pose)
            self.init_base_tf = self.base_transform @ self.init_base_tf @ np.linalg.inv(self.base_transform)
            self.inv_init_base_tf = np.linalg.inv(self.init_base_tf)
        
        habitat_base_pose = cvt_pose_vec2tf(pose)
        base_pose = self.base_transform @ habitat_base_pose @ np.linalg.inv(self.base_transform)
        tf = self.inv_init_base_tf @ base_pose
        
        rgb = np.array(obs["color_sensor"][:,:,:3])
        depth = np.array(obs["depth_sensor"])
        
        patch_tokens = self._get_patch_token(rgb)
        patch_tokens_intr = get_sim_cam_mat(patch_tokens.shape[0], patch_tokens.shape[1])
        
        pc = self._backproject_depth(depth)
        pc_transform = tf @ self.base_transform @ self.base2cam_tf
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
                    self.update_memory()
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
                
        
    def load_memory(self):
        print(f"using memory in {self.args.load_memory_path}")
        self.memory_save_path = self.args.load_memory_path
        self.feat_path = self.memory_save_path+"/feat.h5df"
        
        self.max_id = int(np.load(self.memory_save_path+"/max_id.npy"))
        self.grid_rgb_pos[:self.max_id] =  np.load(self.memory_save_path+"/grid_rgb_pos.npy")
        self.grid_rgb[:self.max_id] =  np.load(self.memory_save_path+"/grid_rgb.npy")
        self.weight[:self.max_id] =  np.load(self.memory_save_path+"/weight.npy")
        self.occupied_ids[:self.max_id] =  np.load(self.memory_save_path+"/occupied_ids.npy")
        self.Env.original_state.position = np.load(self.memory_save_path+"/original_pos.npy")
        
    def create_memory(self):
        self.initial_memory()

        last_action = None
        release_count = 0
        obs = self.Env.sim.get_sensor_observations(0)
        
        
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
                    
            obs = self.Env.sim.step(action)
            agent_state = self.Env.agent.get_state()
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
            
            pos, rot = agent_state.position, agent_state.rotation
            pose = np.array([pos[0], pos[1], pos[2], rot.x, rot.y, rot.z, rot.w])
            
            self.obs2voxeltoken(obs, pose)
            
        cv2.destroyAllWindows()
        # self._visualize_rgb_map_3d(self.grid_pos, self.grid_rgb)
        np.save(self.memory_save_path+"/grid_rgb_pos.npy", self.grid_rgb_pos[:self.max_id])
        np.save(self.memory_save_path+"/grid_rgb.npy", self.grid_rgb[:self.max_id])
        np.save(self.memory_save_path+"/weight.npy", self.weight[:self.max_id])
        np.save(self.memory_save_path+"/occupied_ids.npy", self.occupied_ids)
        np.save(self.memory_save_path+"/max_id.npy", np.array(self.max_id))
        np.save(self.memory_save_path+"/original_pos.npy", self.Env.original_state.position)

    
            
        
if __name__ == "__main__":
    
    args = get_args()
    memory = VoxelTokenMemory(args)
    # memory.matching2D(obs_path='/home/orbit-new/桌面/orbit/shouwei_GES/VLDMap/Vlmaps/DistributionMap_dynamic/vlmaps/data/vlmaps_dataset/5q7pvUzZiYa_1/rgb/000307.png',
    #     text_prompts="A potted plant featuring a round blue vase topped with a white bouquet of a handful of green leaves.")
    
    memory.create_memory()
    memory.localized(text_prompts='A large oven, in the center of the cabinet.')
    # A purple vase with a bouquet of flowers on a kitchen countertop
    # A TV screen which is located in the cabinet.
    # A full watermelon on a blank background
    # grid_pos = np.load("grid_pos.npy")
    # grid_rgb = np.load("grid_rgb.npy")
    
    # memory._visualize_rgb_map_3d(grid_pos, grid_rgb)
        