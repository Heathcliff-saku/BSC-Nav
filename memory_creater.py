import os
import json
import cv2
import numpy as np
import time
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLOWorld
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from tqdm import tqdm
import gc
from collections import defaultdict
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy.spatial.transform import Rotation as R

# Import utility functions
from utils import depth2pc, transform_pc, base_pos2grid_id_3d, project_point, get_sim_cam_mat, get_sim_cam_mat_with_fov, grid_id_3d2base_pos

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class CameraIntrinsics:
    """Camera intrinsics parameters"""
    fx: float
    fy: float
    ppx: float
    ppy: float
    width: int
    height: int
    fov: Optional[float] = None  # Field of view in degrees (if known)

class PhysicalCognitiveMapBuilder:
    """
    Physical environment cognitive map and long-term memory builder
    Processes static RGBD data to build 3D feature grids and object detection results
    """
    
    def __init__(self, 
                 camera_intrinsics: CameraIntrinsics,
                 camera_to_base_transform: Dict,
                 save_path: str,
                 device: str = 'cuda:0',
                 config: Optional[Dict] = None):
        """
        Initialize the cognitive map builder
        
        Args:
            camera_intrinsics: Camera intrinsic parameters
            camera_to_base_transform: Transform from camera to robot base_link
            save_path: Path to save the cognitive map
            device: Computing device ('cuda' or 'cpu')
            config: Additional configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        self.camera_intrinsics = camera_intrinsics
        self.camera_to_base_transform = camera_to_base_transform
        self.save_path = save_path
        

        # dongsheng04 is 0.1 / 2000 / 100
        # Default configuration
        self.config = {
            'cell_size': 0.2,  # Grid cell size in meters
            'grid_size': 2000,  # Grid dimension
            'depth_range': {'min': 0.3, 'max': 8.0},
            'depth_sample_rate': 200,
            'query_height': 252,
            'query_width': 448,
            'patch_size': 14,
            'dino_model': 'dinov2_vitl14_reg',
            'yolo_model': 'yolov8x-worldv2.pt',
            'detect_classes':  ['chair', 'table', 'sink', 'couch', 'plant', 'vending machine', 'trash bin', 'tv', 'kitchen island', 'book', 'lamp'],
            'detect_conf': 0.5,
            'token_dim': 1024,
            'cache_size': 10,
            # 修改1: 扩大高度范围，不再限制floor height
            'floor_height': -15.0,  # 允许负高度
            'map_height': 15,
            'base_forward_axis': [1, 0, 0],
            'base_left_axis': [0, 1, 0],
            'base_up_axis': [0, 0, 1],
            'sensor_height': 1.5,
            'visualize_interval': 1  # Visualize every N frames
        }
        
        # Update config with provided parameters
        if config:
            self.config.update(config)
        
        # Initialize models
        self._init_models()
        
        # Initialize cache and storage
        self._init_storage()
        
        # Setup transforms
        self._init_transforms()
        
        # Initialize visualization
        self._init_visualization()
        
    def _init_models(self):
        """Initialize DINOv2 and YOLO models"""
        self.logger.info("Loading DINOv2 model...")
        # self.dinov2 = torch.hub.load('facebookresearch/dinov2', self.config['dino_model'], source='github').to(self.device)
        self.dinov2 = torch.hub.load('/home/agilex/.cache/torch/hub/facebookresearch_dinov2_main', self.config['dino_model'], trust_repo=True, source='local').to(self.device)
        self.dinov2.eval()
        
        self.logger.info("Loading YOLO-World model...")
        self.yolow = YOLOWorld(self.config['yolo_model']).to(self.device)
        self.yolow.set_classes(self.config['detect_classes'])
        
    def _init_storage(self):
        """Initialize storage structures"""
        # Create save directory
        os.makedirs(self.save_path, exist_ok=True)
        self.feat_path = os.path.join(self.save_path, "feat.h5df")
        
        # Grid parameters
        self.cs = self.config['cell_size']
        self.gs = self.config['grid_size']
        self.maxh = int(self.config['map_height'] / self.cs)
        self.minh = int(self.config['floor_height'] / self.cs)
        
        # Initialize grid structures
        self.grid_rgb_pos = np.zeros((self.gs * self.gs, 3), dtype=np.int32)
        self.occupied_ids = -1 * np.ones((self.gs, self.gs, self.maxh - self.minh), dtype=np.int32)
        self.weight = np.zeros((self.gs * self.gs), dtype=np.float32)
        self.grid_rgb = np.zeros((self.gs * self.gs, 3), dtype=np.uint8)
        self.max_id = 0
        
        # Temporary feature cache
        self.grid_feat = np.zeros((50000, self.config['token_dim']), dtype=np.float32)
        self.grid_feat_pos = np.zeros((50000, 3), dtype=np.int32)
        self.grid_feat_dis = np.zeros((50000,), dtype=np.float32)
        self.iter_id = 0
        
        # 2D map for visualization
        self.cv_map = np.zeros((self.gs, self.gs, 3), dtype=np.uint8)
        self.max_height = np.full((self.gs, self.gs), -np.inf)
        
        # Long-term memory
        self.long_memory_dict = []
        
        # Camera matrix - handle both FOV-based and direct intrinsics
        if self.camera_intrinsics.fov is not None:
            # Use FOV to calculate intrinsics (like in simulator)
            self.calib_mat = get_sim_cam_mat_with_fov(
                self.camera_intrinsics.height, 
                self.camera_intrinsics.width, 
                self.camera_intrinsics.fov
            )
            self.logger.info(f"Using FOV-based camera intrinsics with FOV={self.camera_intrinsics.fov}°")
        else:
            # Use provided intrinsics directly
            self.calib_mat = np.array([
                [self.camera_intrinsics.fx, 0, self.camera_intrinsics.ppx],
                [0, self.camera_intrinsics.fy, self.camera_intrinsics.ppy],
                [0, 0, 1]
            ])
            # Calculate FOV from fx for reference
            calculated_fov = 2 * np.arctan(self.camera_intrinsics.width / (2 * self.camera_intrinsics.fx))
            calculated_fov_deg = np.rad2deg(calculated_fov)
            self.logger.info(f"Using direct camera intrinsics (calculated FOV≈{calculated_fov_deg:.1f}°)")
        
        # Patch dimensions
        self.n_patch_h = self.config['query_height'] // self.config['patch_size']
        self.n_patch_w = self.config['query_width'] // self.config['patch_size']
        
    def _init_transforms(self):
        """Initialize image transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((self.config['query_height'], self.config['query_width'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_ = transforms.Compose([
            transforms.Resize((self.config['query_height'], self.config['query_width'])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def _init_visualization(self):
        """Initialize visualization components"""
        self.viz_fig = None
        self.viz_axes = None
        self.frame_count = 0
        plt.ion()  # Enable interactive mode
        
    def _visualize_frame(self, color_img: np.ndarray, depth_img: np.ndarray, 
                        robot_pose: Dict, detection_results: Optional[List] = None):
        """
        Visualize current frame processing results
        
        Args:
            color_img: RGB image
            depth_img: Depth image in millimeters
            robot_pose: Robot pose information
            detection_results: YOLO detection results (optional)
        """
        # Initialize figure on first call
        if self.viz_fig is None:
            self.viz_fig, self.viz_axes = plt.subplots(2, 2, figsize=(15, 12))
            self.viz_fig.suptitle('Cognitive Map Building Progress', fontsize=16)
            
        # Clear previous plots
        for ax in self.viz_axes.flat:
            ax.clear()
        
        # 1. RGB Image with detections
        ax_rgb = self.viz_axes[0, 0]
        rgb_display = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        ax_rgb.imshow(rgb_display)
        ax_rgb.set_title(f'RGB Image (Frame {self.frame_count})')
        ax_rgb.axis('off')
        
        # Add detection boxes if available
        if detection_results and len(detection_results[0].boxes) > 0:
            for i in range(len(detection_results[0].boxes.conf)):
                xyxy = detection_results[0].boxes.xyxy[i].cpu().numpy()
                cls = int(detection_results[0].boxes.cls[i].item())
                conf = detection_results[0].boxes.conf[i].item()
                label = self.config['detect_classes'][cls]
                
                # Draw bounding box
                rect = Rectangle((xyxy[0], xyxy[1]), xyxy[2]-xyxy[0], xyxy[3]-xyxy[1],
                               linewidth=2, edgecolor='red', facecolor='none')
                ax_rgb.add_patch(rect)
                ax_rgb.text(xyxy[0], xyxy[1]-5, f'{label} {conf:.2f}', 
                           color='red', fontsize=10, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        # 2. Depth Image
        ax_depth = self.viz_axes[0, 1]
        depth_display = depth_img.astype(np.float32) / 1000.0  # Convert to meters
        depth_colormap = plt.cm.viridis(depth_display / depth_display.max())
        ax_depth.imshow(depth_colormap)
        ax_depth.set_title('Depth Image')
        ax_depth.axis('off')
        
        # Add colorbar for depth
        # im = ax_depth.imshow(depth_display, cmap='viridis')
        # cbar = plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)
        # cbar.set_label('Depth (m)', rotation=270, labelpad=15)
        
        # 3. Current CV Map (zoomed in around robot position)
        ax_map = self.viz_axes[1, 0]
        # Get robot grid position
        robot_x = robot_pose['pose']['position']['x']
        robot_y = robot_pose['pose']['position']['y']
        robot_grid_x = int(robot_x / self.cs + self.gs // 2)
        robot_grid_y = int(robot_y / self.cs + self.gs // 2)
        
        # Define zoom window
        window_size = 200  # Grid cells to show
        x_min = max(0, robot_grid_x - window_size // 2)
        x_max = min(self.gs, robot_grid_x + window_size // 2)
        y_min = max(0, robot_grid_y - window_size // 2)
        y_max = min(self.gs, robot_grid_y + window_size // 2)
        
        # Extract and display map region
        map_region = self.cv_map[y_min:y_max, x_min:x_max]
        ax_map.imshow(map_region)
        ax_map.set_title(f'Top-down Map (Cell size: {self.cs}m)')
        ax_map.axis('off')
        
        # Add robot position marker
        robot_local_x = robot_grid_x - x_min
        robot_local_y = robot_grid_y - y_min
        ax_map.plot(robot_local_x, robot_local_y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # Add robot orientation
        robot_quat = [
            robot_pose['pose']['orientation']['x'],
            robot_pose['pose']['orientation']['y'],
            robot_pose['pose']['orientation']['z'],
            robot_pose['pose']['orientation']['w']
        ]
        # Simple yaw extraction from quaternion
        yaw = np.arctan2(2 * (robot_quat[3] * robot_quat[2] + robot_quat[0] * robot_quat[1]),
                        1 - 2 * (robot_quat[1]**2 + robot_quat[2]**2))
        arrow_length = 20
        ax_map.arrow(robot_local_x, robot_local_y, 
                    arrow_length * np.cos(yaw), arrow_length * np.sin(yaw),
                    head_width=5, head_length=5, fc='red', ec='red')
        
        # 4. Statistics and Info
        ax_info = self.viz_axes[1, 1]
        ax_info.axis('off')
        
        # Prepare statistics text
        info_text = f"Frame: {self.frame_count}\n"
        info_text += f"Robot Position: ({robot_x:.2f}, {robot_y:.2f})\n"
        info_text += f"Grid Cells Occupied: {self.max_id}\n"
        info_text += f"Features Cached: {self.iter_id}\n"
        info_text += f"Objects Detected: {len(self.long_memory_dict)}\n\n"
        
        # Add object statistics
        object_counts = defaultdict(int)
        for obj in self.long_memory_dict:
            object_counts[obj['label']] += 1
        
        if object_counts:
            info_text += "Detected Objects:\n"
            for label, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
                info_text += f"  {label}: {count}\n"
        
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                    fontsize=12, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        # Update display
        plt.tight_layout()
        plt.pause(0.001)  # Small pause to allow display update
        
    def _create_transformation_matrix(self, translation: List[float], rotation_quat: List[float]) -> np.ndarray:
        """Create 4x4 transformation matrix from translation and quaternion"""
        # Normalize quaternion
        q = np.array(rotation_quat)
        q = q / np.linalg.norm(q)
        x, y, z, w = q
        
        # Quaternion to rotation matrix
        R = np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
            [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
            [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
        ])
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        
        return T
    
    def _get_patch_token(self, img: np.ndarray) -> torch.Tensor:
        """Extract patch-level features using DINOv2"""
        img = torch.from_numpy(img).to(self.device).unsqueeze(0)
        img = img.permute(0, 3, 1, 2).float() / 255
        img = self.transform_(img)
        
        with torch.no_grad():
            img_features_dict = self.dinov2.forward_features(img)
            patch_token = img_features_dict['x_norm_patchtokens'].squeeze(0)
            patch_token = patch_token.reshape(self.n_patch_h, self.n_patch_w, -1)
            
        return patch_token
    
    def _backproject_depth(self, depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backproject depth image to point cloud
        修改：返回对应的像素坐标，以便正确获取RGB值
        """
        pc, mask = depth2pc(
            depth, 
            intr_mat=self.calib_mat, 
            min_depth=self.config['depth_range']['min'], 
            max_depth=self.config['depth_range']['max']
        )
        
        # 获取有效点的像素坐标
        h, w = depth.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        pixel_coords = np.stack([x.flatten(), y.flatten()], axis=0)
        
        # Downsample - 保持像素坐标的对应关系
        shuffle_mask = np.arange(pc.shape[1])
        np.random.shuffle(shuffle_mask)
        shuffle_mask = shuffle_mask[::self.config['depth_sample_rate']]
        mask = mask[shuffle_mask]
        pc = pc[:, shuffle_mask]
        pixel_coords = pixel_coords[:, shuffle_mask]
        
        # 只保留有效点
        pc = pc[:, mask]
        pixel_coords = pixel_coords[:, mask]
        
        return pc, mask, pixel_coords
    
    def _out_of_range(self, row: int, col: int, height: int) -> bool:
        """
        Check if grid coordinates are out of range
        修改：放宽高度限制
        """
        # 只检查水平范围和高度上限，不限制下限（允许负高度）
        return col >= self.gs or row >= self.gs or col < 0 or row < 0 or height >= self.maxh
    
    def process_frame(self, color_img: np.ndarray, depth_img: np.ndarray, robot_pose: Dict, visualize: bool = True):
        """
        Process a single RGBD frame
        修改：改进坐标变换和RGB颜色获取逻辑
        """
        # Convert depth to meters
        depth = depth_img.astype(np.float32) / 1000.0
        
        # Extract patch features
        patch_tokens = self._get_patch_token(color_img)
        # Create patch token intrinsics matrix for projection
        patch_tokens_intr = get_sim_cam_mat(patch_tokens.shape[0], patch_tokens.shape[1])
        
        # Backproject depth to point cloud - 获取像素坐标
        pc, _, pixel_coords = self._backproject_depth(depth)
        
        # Transform to world coordinates
        # 1. Camera to base_link
        T_base_cam = self._create_transformation_matrix(
            self.camera_to_base_transform['translation'],
            self.camera_to_base_transform['rotation']
        )
        
        # 2. Base_link to world
        T_world_base = self._create_transformation_matrix(
            [robot_pose['pose']['position']['x'],
             robot_pose['pose']['position']['y'],
             robot_pose['pose']['position']['z']],
            [robot_pose['pose']['orientation']['x'],
             robot_pose['pose']['orientation']['y'],
             robot_pose['pose']['orientation']['z'],
             robot_pose['pose']['orientation']['w']]
        )
        
        # Combined transform
        pc_transform = T_world_base @ T_base_cam
        pc_global = transform_pc(pc, pc_transform)
        
        # Process each point
        for i, (p, p_local, pix_coord) in enumerate(zip(pc_global.T, pc.T, pixel_coords.T)):
            row, col, height = base_pos2grid_id_3d(self.gs, self.cs, p[0], p[1], p[2])
            
            # 修改：允许负高度，调整height索引
            if self._out_of_range(row, col, height):
                continue
            height_idx = height - self.minh  # 转换为数组索引
            if height_idx < 0:  # 确保索引非负
                continue
                
            # 修改：直接使用保存的像素坐标获取RGB值
            px, py = int(pix_coord[0]), int(pix_coord[1])
            if 0 <= px < color_img.shape[1] and 0 <= py < color_img.shape[0]:
                # 注意：OpenCV读取的是BGR格式，但这里保持原始格式
                rgb_v = color_img[py, px, :]
            else:
                continue
            
            # Get patch token - 使用相机坐标系下的点投影到patch空间
            px_patch, py_patch, _ = project_point(patch_tokens_intr, p_local)
            
            # Calculate weight based on distance (for weighted averaging)
            radial_dist_sq = np.sum(np.square(p_local))
            sigma_sq = 0.6
            alpha = np.exp(-radial_dist_sq / (2 * sigma_sq))
            
            # Store patch features if within bounds
            if 0 <= px_patch < patch_tokens.shape[1] and 0 <= py_patch < patch_tokens.shape[0]:
                if self.iter_id < self.grid_feat.shape[0]:
                    self.grid_feat[self.iter_id, :] = patch_tokens[py_patch, px_patch, :].cpu().numpy()
                    self.grid_feat_pos[self.iter_id, :] = [row, col, height_idx]
                    self.grid_feat_dis[self.iter_id] = radial_dist_sq
                    self.iter_id += 1
                else:
                    # Update memory when cache is full
                    self._update_memory()
            
            # Update RGB grid
            occupied_id = self.occupied_ids[row, col, height_idx]
            if occupied_id == -1:
                self.occupied_ids[row, col, height_idx] = self.max_id
                self.grid_rgb[self.max_id] = rgb_v
                self.weight[self.max_id] = alpha
                self.grid_rgb_pos[self.max_id] = [row, col, height_idx]
                self.max_id += 1
            else:
                # Weighted average for RGB values
                self.grid_rgb[occupied_id] = (
                    self.grid_rgb[occupied_id] * self.weight[occupied_id] + rgb_v * alpha
                ) / (self.weight[occupied_id] + alpha)
                self.weight[occupied_id] += alpha
            
            # Update 2D map (top-down view) - 使用实际高度而不是索引
            if height >= self.max_height[row, col]:
                self.max_height[row, col] = height
                self.cv_map[row, col] = rgb_v
        
        # Object detection for long-term memory
        detection_results = self._process_detection(color_img, depth, pc_transform)
        
        # Visualize if requested
        if visualize and self.frame_count % self.config['visualize_interval'] == 0:
            self._visualize_frame(color_img, depth_img, robot_pose, detection_results)
        
        self.frame_count += 1
        
        return detection_results
    
    def _process_detection(self, color_img: np.ndarray, depth: np.ndarray, pc_transform: np.ndarray):
        """Process YOLO detection for long-term memory"""
        # 修改：转换BGR到RGB用于YOLO检测
        rgb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        results = self.yolow.predict(Image.fromarray(rgb_img), conf=self.config['detect_conf'])
        
        if len(results[0].boxes) > 0:
            pc, mask = depth2pc(
                depth, 
                intr_mat=self.calib_mat,
                min_depth=self.config['depth_range']['min'],
                max_depth=self.config['depth_range']['max']
            )
            
            for i in range(len(results[0].boxes.conf)):
                xyxy = results[0].boxes.xyxy[i].cpu().numpy()
                col = int((xyxy[0] + xyxy[2]) / 2)
                row = int((xyxy[1] + xyxy[3]) / 2)
                index = row * self.camera_intrinsics.width + col
                
                if index < len(mask) and mask[index]:
                    # Transform to world coordinates
                    p_local = pc[:, index]
                    p_global = (pc_transform @ np.append(p_local, 1))[:3]
                    
                    grid_row, grid_col, grid_height = base_pos2grid_id_3d(
                        self.gs, self.cs, p_global[0], p_global[1], p_global[2]
                    )
                    
                    # 修改：调整高度检查
                    if not self._out_of_range(grid_row, grid_col, grid_height):
                        grid_height_idx = grid_height - self.minh
                        if grid_height_idx >= 0:
                            cls = int(results[0].boxes.cls[i].item())
                            instance = {
                                'label': self.config['detect_classes'][cls],
                                'loc': [grid_row, grid_col, grid_height_idx],
                                'confidence': results[0].boxes.conf[i].item()
                            }
                            self.long_memory_dict.append(instance)
        
        return results
    
    def _update_memory(self):
        """Update HDF5 memory storage"""
        self.logger.info("Updating memory storage...")
        
        with h5py.File(self.feat_path, 'a') as h5f:
            for i in range(self.iter_id):
                grid_id = f'{self.grid_feat_pos[i][0]}_{self.grid_feat_pos[i][1]}_{self.grid_feat_pos[i][2]}'
                group_name = f'grid_{grid_id}'
                
                if group_name not in h5f:
                    group = h5f.create_group(group_name)
                    group.create_dataset('features', data=self.grid_feat[i:i+1], 
                                       maxshape=(None, self.grid_feat.shape[1]), chunks=True)
                    group.create_dataset('distances', data=self.grid_feat_dis[i:i+1], 
                                       maxshape=(None,), chunks=True)
                else:
                    group = h5f[group_name]
                    features = group['features']
                    distances = group['distances']
                    
                    if features.shape[0] < self.config['cache_size']:
                        features.resize((features.shape[0] + 1, features.shape[1]))
                        distances.resize((distances.shape[0] + 1,))
                        features[-1] = self.grid_feat[i]
                        distances[-1] = self.grid_feat_dis[i]
                    else:
                        # Replace random token
                        replace_idx = np.random.choice(features.shape[0])
                        features[replace_idx] = self.grid_feat[i]
                        distances[replace_idx] = self.grid_feat_dis[i]
        
        # Reset cache
        self.iter_id = 0
        self.grid_feat.fill(0)
        self.grid_feat_pos.fill(0)
        self.grid_feat_dis.fill(0)
    
    def build_from_session(self, session_dir: str, visualize: bool = True, visualize_during_build: bool = True):
        """
        Build cognitive map from session data
        
        Args:
            session_dir: Directory containing session data
            visualize: Whether to visualize the final result
            visualize_during_build: Whether to visualize during building process
        """
        # Load data index
        index_path = os.path.join(session_dir, "data_index_filtered.json")
        with open(index_path, 'r') as f:
            data_entries = json.load(f)
        
        self.logger.info(f"Processing {len(data_entries)} frames...")
        
        # Process each frame
        for entry in tqdm(data_entries):
            try:
                # Load images
                color_img = cv2.imread(entry['color_image_path'])
                depth_img = cv2.imread(entry['depth_image_path'], cv2.IMREAD_UNCHANGED)
                
                if color_img is None or depth_img is None:
                    self.logger.warning(f"Failed to load images: {entry['color_image_path']}")
                    continue
                
                # Process frame with visualization
                self.process_frame(color_img, depth_img, entry['robot_pose'], visualize=visualize_during_build)
                
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
                continue
        
        # Final memory update
        if self.iter_id > 0:
            self._update_memory()
        
        # Integrate long-term memory
        self._integrate_long_memory()
        
        # Save results
        self.save_results()
        
        # Save final cv_map as PNG
        self._save_cv_map_png()
        
        # Close visualization window if open
        if self.viz_fig is not None:
            plt.close(self.viz_fig)
            plt.ioff()
        
        # Visualize final results if requested
        if visualize:
            self.visualize_results()
    
    def _integrate_long_memory(self, threshold: int = 3):
        """Integrate long-term memory by merging nearby detections"""
        def l1_distance(loc1, loc2):
            return sum(abs(a - b) for a, b in zip(loc1, loc2))
        
        # Group by label
        label_groups = defaultdict(list)
        for item in self.long_memory_dict:
            label_groups[item["label"]].append(item)
        
        # Filter and merge
        final_results = []
        for label, items in label_groups.items():
            filtered = []
            for item in items:
                merged = False
                for f in filtered:
                    if l1_distance(f['loc'], item['loc']) <= threshold:
                        if item['confidence'] > f['confidence']:
                            f['loc'] = item['loc']
                            f['confidence'] = item['confidence']
                        merged = True
                        break
                if not merged:
                    filtered.append(item)
            final_results.extend(filtered)
        
        self.long_memory_dict = final_results
    
    def _save_cv_map_png(self):
        """Save the final cv_map as PNG image"""
        self.logger.info("Saving cv_map as PNG...")
        
        # Find the bounding box of non-zero areas in cv_map
        non_zero_mask = np.any(self.cv_map != 0, axis=2)
        if np.any(non_zero_mask):
            rows = np.any(non_zero_mask, axis=1)
            cols = np.any(non_zero_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add some padding
            padding = 50
            rmin = max(0, rmin - padding)
            rmax = min(self.gs - 1, rmax + padding)
            cmin = max(0, cmin - padding)
            cmax = min(self.gs - 1, cmax + padding)
            
            # Extract the region
            cv_map_cropped = self.cv_map[rmin:rmax+1, cmin:cmax+1]
            
            # Save full map
            cv2.imwrite(os.path.join(self.save_path, "cv_map_full.png"), self.cv_map)
            self.logger.info(f"Full cv_map saved: {self.cv_map.shape}")
            
            # Save cropped map
            cv2.imwrite(os.path.join(self.save_path, "cv_map_cropped.png"), cv_map_cropped)
            self.logger.info(f"Cropped cv_map saved: {cv_map_cropped.shape}")
            
            # Create a visualization with grid lines and scale
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Convert BGR to RGB for matplotlib
            cv_map_rgb = cv2.cvtColor(cv_map_cropped, cv2.COLOR_BGR2RGB)
            im = ax.imshow(cv_map_rgb)
            
            # Add grid
            grid_spacing = 50  # Grid lines every 50 cells
            for i in range(0, cv_map_cropped.shape[0], grid_spacing):
                ax.axhline(i, color='gray', alpha=0.3, linewidth=0.5)
            for i in range(0, cv_map_cropped.shape[1], grid_spacing):
                ax.axvline(i, color='gray', alpha=0.3, linewidth=0.5)
            
            # Add scale information
            scale_text = f"1 pixel = {self.cs}m, Grid spacing = {grid_spacing * self.cs}m"
            ax.text(0.02, 0.98, scale_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Add detected objects as markers
            for obj in self.long_memory_dict:
                obj_row, obj_col, _ = obj['loc']
                # Check if object is within the cropped region
                if rmin <= obj_row <= rmax and cmin <= obj_col <= cmax:
                    # Adjust coordinates to cropped region
                    marker_row = obj_row - rmin
                    marker_col = obj_col - cmin
                    ax.plot(marker_col, marker_row, 'r*', markersize=8)
                    ax.text(marker_col + 2, marker_row - 2, obj['label'], 
                           fontsize=8, color='red',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
            
            ax.set_title('Top-down Cognitive Map', fontsize=16)
            ax.set_xlabel('X (grid cells)')
            ax.set_ylabel('Y (grid cells)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, "cv_map_annotated.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info("Annotated cv_map saved")
        else:
            self.logger.warning("cv_map is empty, skipping PNG save")
    
    def save_results(self):
        """Save all results"""
        self.logger.info("Saving results...")
        
        # Save grid data
        np.save(os.path.join(self.save_path, "grid_rgb_pos.npy"), self.grid_rgb_pos[:self.max_id])
        np.save(os.path.join(self.save_path, "grid_rgb.npy"), self.grid_rgb[:self.max_id])
        np.save(os.path.join(self.save_path, "weight.npy"), self.weight[:self.max_id])
        np.save(os.path.join(self.save_path, "occupied_ids.npy"), self.occupied_ids)
        np.save(os.path.join(self.save_path, "max_id.npy"), np.array(self.max_id))
        np.save(os.path.join(self.save_path, "cv_map.npy"), self.cv_map)
        
        # Save long-term memory
        with open(os.path.join(self.save_path, "long_memory.json"), 'w') as f:
            json.dump(self.long_memory_dict, f, indent=4)
        
        # Save point cloud
        self._save_point_cloud()
        
        self.logger.info(f"Results saved to {self.save_path}")
    
    def _save_point_cloud(self):
        """Save RGB point cloud"""
        # Create point cloud
        valid_points = self.grid_rgb_pos[:self.max_id]
        valid_colors = self.grid_rgb[:self.max_id]
        
        # Convert grid coordinates to world coordinates
        points_world = []
        for i in range(self.max_id):
            row, col, height_idx = valid_points[i]
            x = (col - self.gs // 2) * self.cs
            y = (row - self.gs // 2) * self.cs
            # 修改：使用实际高度而不是索引
            z = (height_idx + self.minh) * self.cs
            points_world.append([x, y, z])
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_world))
        # 修改：如果是BGR格式，转换为RGB
        if valid_colors.shape[1] == 3:
            # BGR to RGB
            rgb_colors = valid_colors[:, [2, 1, 0]]
            pcd.colors = o3d.utility.Vector3dVector(rgb_colors / 255.0)
        else:
            pcd.colors = o3d.utility.Vector3dVector(valid_colors / 255.0)
        
        # Save
        o3d.io.write_point_cloud(os.path.join(self.save_path, "cognitive_map.ply"), pcd)
    
    def visualize_results(self):
        """Visualize the built map"""
        # Load point cloud
        pcd = o3d.io.read_point_cloud(os.path.join(self.save_path, "cognitive_map.ply"))
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        # Visualize
        o3d.visualization.draw_geometries([pcd, coordinate_frame], window_name="Cognitive Map")
    
    def voxel_localize(self, query: str, K: int = 100, region_radius: float = np.inf, 
                      curr_grid: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Localize voxels based on text query
        
        Args:
            query: Text query or PIL Image
            K: Number of top matches to return
            region_radius: Search radius constraint
            curr_grid: Current grid position for local search
            
        Returns:
            best_pos: Best matching position
            top_k_positions: Top K positions
            top_k_similarities: Corresponding similarities
        """
        # Get query features
        if isinstance(query, str):
            # For text queries, you would need to implement text-to-image generation
            # For now, this is a placeholder
            raise NotImplementedError("Text queries require image generation model")
        else:
            # Direct image query
            query_tensor = self.transform(query).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract query features
            query_features_dict = self.dinov2.forward_features(query_tensor)
            tokens = query_features_dict['x_norm_patchtokens']
            
            # Weighted average of patches (center-weighted)
            batch_size, num_tokens, feature_dim = tokens.size()
            grid_size = int(np.sqrt(num_tokens))
            
            xs = torch.arange(grid_size, device=self.device).repeat(grid_size).view(1, num_tokens)
            ys = torch.arange(grid_size, device=self.device).repeat_interleave(grid_size).view(1, num_tokens)
            center = (grid_size - 1) / 2
            distances = (xs - center) ** 2 + (ys - center) ** 2
            sigma = (grid_size / 2) ** 2
            weights = torch.exp(-distances / (2 * sigma))
            weights = weights / weights.sum(dim=1, keepdim=True)
            weights = weights.unsqueeze(-1)
            
            weighted_tokens = tokens * weights
            weighted_sum = weighted_tokens.sum(dim=1)
            query_features = weighted_sum.mean(dim=0).unsqueeze(0)
            
            # Search in HDF5
            candidates = []
            with h5py.File(self.feat_path, 'r') as h5f:
                group_names = list(h5f.keys())
                
                # Apply region constraint if specified
                if region_radius != np.inf and curr_grid is not None:
                    filtered_names = []
                    for name in group_names:
                        _, x, y, z = name.split('_')
                        dist_sq = (int(x) - curr_grid[0])**2 + (int(y) - curr_grid[1])**2 + (int(z) - curr_grid[2])**2
                        if dist_sq <= region_radius**2:
                            filtered_names.append(name)
                    group_names = filtered_names
                
                # Batch process
                for name in group_names:
                    group = h5f[name]
                    features = torch.from_numpy(group['features'][:]).to(self.device)
                    similarities = F.cosine_similarity(query_features, features, dim=1)
                    max_sim = similarities.max().item()
                    
                    pos = tuple(map(int, name.split('_')[1:4]))
                    candidates.append((max_sim, pos))
            
            # Sort and return top K
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_k_positions = [pos for _, pos in candidates[:K]]
            top_k_similarities = [sim for sim, _ in candidates[:K]]
            
            return np.array(top_k_positions[0]), np.array(top_k_positions), np.array(top_k_similarities)
    
    def get_long_memory_objects(self, label_filter: Optional[List[str]] = None) -> List[Dict]:
        """
        Get long-term memory objects with optional label filtering
        
        Args:
            label_filter: List of labels to filter by
            
        Returns:
            List of detected objects
        """
        if label_filter is None:
            return self.long_memory_dict
        
        return [obj for obj in self.long_memory_dict if obj['label'] in label_filter]
    
    def get_2d_map(self) -> np.ndarray:
        """Get 2D occupancy map"""
        return self.cv_map






def main():
    """Example usage"""
    # Camera intrinsics (example for RealSense)
    camera_intrinsics = CameraIntrinsics(
        fx=607.96533203125,
        fy=607.874755859375,
        ppx=428.05804443359375,
        ppy=245.64642333984375,
        width=848,
        height=480
    )
    
    # Camera to base_link transform
    pitch_angle = 0  # 向下倾斜20度（根据实际情况调整）
    camera_rotation = R.from_euler('y', pitch_angle, degrees=True)
    camera_quat = camera_rotation.as_quat()  # [x, y, z, w]

    # 更新相机变换
    camera_transform = {
        'translation': [-0.1, 0.0, -1.35],  # 可根据实际调整
        'rotation': camera_quat.tolist()
    }

    optical_to_ros = R.from_euler('xyz', [-90, 0, -90], degrees=True)
    combined_rotation = camera_rotation * optical_to_ros
    camera_transform['rotation'] = combined_rotation.as_quat().tolist()
    
    # Configuration
    config = {
        # 'cell_size': 0.2,
        # 'grid_size': 2000,
        # 'detect_classes': ['chair', 'table', 'sink', 'couch', 'plant', 'vending machine', 'trash bin', 'tv', 'kitchen island', 'book', 'lamp'],
        'detect_classes': ['chair', 'sofa', 'potted plant', 'table', 'sink', 'vending machine', 'trash bin'],
        'visualize_interval': 1  # Visualize every 5 frames
    }
    
    # Create builder
    builder = PhysicalCognitiveMapBuilder(
        camera_intrinsics=camera_intrinsics,
        camera_to_base_transform=camera_transform,
        save_path="cognitive_map_output",
        config=config
    )
    
    # Build from session data with visualization
    session_dir = "/home/agilex/桌面/nav2025-phy/realtime_mapping_data/realtime_session_20250711_092642"
    builder.build_from_session(
        session_dir, 
        visualize=True,  # Visualize final 3D result
        visualize_during_build=True  # Visualize during building process
    )
    
    # Query example
    # results = builder.voxel_localize("a red chair", K=10)
    
    # Get detected objects
    objects = builder.get_long_memory_objects(label_filter=['chair', 'table'])
    print(f"Found {len(objects)} objects")
    
    # The cv_map has been automatically saved as PNG images in the output directory
    print(f"CV map saved to: {builder.save_path}")


if __name__ == "__main__":
    main()