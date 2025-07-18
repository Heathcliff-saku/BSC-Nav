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
import open3d as o3d
from typing import Dict, List, Tuple, Optional, Union
import logging
import gc
import re
import matplotlib.pyplot as plt
import math
from sklearn.cluster import DBSCAN
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from utils import depth2pc, transform_pc, base_pos2grid_id_3d, project_point, get_sim_cam_mat, get_sim_cam_mat_with_fov, grid_id_3d2base_pos

# Import the base builder class
from memory_creater import PhysicalCognitiveMapBuilder, CameraIntrinsics

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class Memory(PhysicalCognitiveMapBuilder):
    """
    Extended memory system with retrieval capabilities
    Inherits all building functionality from PhysicalCognitiveMapBuilder
    Adds loading, retrieval, and visualization features
    """
    
    def __init__(self, 
                 camera_intrinsics: CameraIntrinsics,
                 camera_to_base_transform: Dict,
                 save_path: str,
                 device: str = 'cuda:0',
                 config: Optional[Dict] = None,
                 need_diffusion: bool = True,
                 gpt_client=None,
                 preload_features: bool = True):
        """
        Initialize the memory system with retrieval capabilities
        
        Args:
            camera_intrinsics: Camera intrinsic parameters
            camera_to_base_transform: Transform from camera to robot base_link
            save_path: Path to save/load the memory
            device: Computing device ('cuda' or 'cpu')
            config: Additional configuration parameters
            need_diffusion: Whether to load diffusion model for text-to-image generation
            gpt_client: GPT client for enhanced long-term memory retrieval
            preload_features: Whether to preload all features from HDF5 to GPU memory
        """
        # Initialize parent class
        super().__init__(camera_intrinsics, camera_to_base_transform, save_path, device, config)
        
        # Additional attributes for retrieval
        self.memory_loaded = False
        self.gpt_client = gpt_client
        self.diffusion = None
        self.preload_features = preload_features
        
        # Preloaded features storage
        self.preloaded_feature_data = None
        self.feature_index_map = {}  # Maps group_name to (start_idx, end_idx) in preloaded tensor
        self.grid_positions = []  # Store grid positions corresponding to feature groups
        
        # Pattern for parsing GPT responses
        self.pattern_unable = re.compile(r'unable|cannot|no.*found', re.IGNORECASE)
        
        # Extended config for retrieval
        if config is None:
            config = {}
        retrieval_config = {
            'diffusion_id': 'stabilityai/stable-diffusion-3.5-medium',
            'gen_height': 512,
            'gen_width': 512,
            'imaginary_num': 3,  # Number of images to generate for text queries
        }
        retrieval_config.update(config)
        self.config.update(retrieval_config)
        
        # Initialize diffusion model if needed
        if need_diffusion:
            self.logger.info("Loading Diffusion model for text-to-image generation...")
            self._quantize_diffusion(self.config['diffusion_id'])
    
    def _quantize_diffusion(self, model_id):
        """Load quantized diffusion model for memory efficiency"""
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
            torch_dtype=torch.bfloat16,
            local_files_only = True
        )

        self.diffusion.enable_model_cpu_offload()
    
    def _preload_all_features(self):
        """
        Preload all features from HDF5 file to GPU memory
        This significantly speeds up retrieval at the cost of GPU memory
        """
        if not os.path.exists(self.feat_path):
            self.logger.error(f"Feature database not found at {self.feat_path}")
            return
        
        self.logger.info("Preloading all features to GPU memory...")
        start_time = time.time()
        
        try:
            with h5py.File(self.feat_path, 'r') as h5f:
                # Single pass: collect features and build index map
                feature_list = []
                feature_dim = None
                current_idx = 0
                self.grid_positions = []
                
                # Sort group names to ensure consistent ordering
                group_names = sorted([name for name in h5f.keys() if name.startswith('grid_')])
                
                for group_name in group_names:
                    group = h5f[group_name]
                    if 'features' in group:
                        # Read features
                        features_np = group['features'][:]
                        num_features = features_np.shape[0]
                        
                        # Get feature dimension from first valid group
                        if feature_dim is None:
                            feature_dim = features_np.shape[1]
                        
                        # Convert to tensor and add to list
                        features_tensor = torch.from_numpy(features_np).to(self.device, dtype=torch.float32)
                        feature_list.append(features_tensor)
                        
                        # Extract position from group name
                        _, x, y, z = group_name.split('_')
                        position = (int(x), int(y), int(z))
                        
                        # Store index mapping
                        end_idx = current_idx + num_features
                        self.feature_index_map[group_name] = (current_idx, end_idx, position)
                        self.grid_positions.append(position)
                        current_idx = end_idx
                
                if not feature_list:
                    self.logger.warning("No features found in HDF5 file")
                    return
                
                # Concatenate all features into one tensor
                self.preloaded_feature_data = torch.cat(feature_list, dim=0)
                total_features = self.preloaded_feature_data.shape[0]
                
                # Convert grid positions to numpy array
                self.grid_positions = np.array(self.grid_positions)
                
                # Calculate memory usage
                memory_usage_gb = (self.preloaded_feature_data.element_size() * 
                                 self.preloaded_feature_data.nelement()) / (1024**3)
                
                self.logger.info(f"Feature preloading completed in {time.time() - start_time:.2f}s")
                self.logger.info(f"Total features loaded: {total_features}")
                self.logger.info(f"GPU memory usage: {memory_usage_gb:.2f} GB")
                self.logger.info(f"Number of grid cells: {len(self.feature_index_map)}")
                
        except Exception as e:
            self.logger.error(f"Failed to preload features: {e}")
            self.preloaded_feature_data = None
            self.feature_index_map = {}
            self.grid_positions = []
    
    def load_memory(self, memory_path: Optional[str] = None):
        """
        Load existing memory from disk
        
        Args:
            memory_path: Path to memory directory. If None, uses self.save_path
        """
        if memory_path:
            self.save_path = memory_path
        
        self.logger.info(f"Loading memory from {self.save_path}")
        
        try:
            # Load feature database path
            self.feat_path = os.path.join(self.save_path, "feat.h5df")
            
            # Load grid data
            self.max_id = int(np.load(os.path.join(self.save_path, "max_id.npy")))
            self.grid_rgb_pos[:self.max_id] = np.load(os.path.join(self.save_path, "grid_rgb_pos.npy"))
            self.grid_rgb[:self.max_id] = np.load(os.path.join(self.save_path, "grid_rgb.npy"))
            self.weight[:self.max_id] = np.load(os.path.join(self.save_path, "weight.npy"))
            self.occupied_ids = np.load(os.path.join(self.save_path, "occupied_ids.npy"))
            
            # Load cv_map if exists
            cv_map_path = os.path.join(self.save_path, "cv_map.npy")
            if os.path.exists(cv_map_path):
                self.cv_map = np.load(cv_map_path)
            
            # Load long-term memory
            with open(os.path.join(self.save_path, "long_memory.json"), "r") as f:
                self.long_memory_dict = json.load(f)
            
            self.memory_loaded = True
            self.logger.info(f"Memory loaded successfully. Total voxels: {self.max_id}, Objects: {len(self.long_memory_dict)}")
            
            # Preload features if requested
            if self.preload_features:
                self._preload_all_features()
            
        except Exception as e:
            self.logger.error(f"Failed to load memory: {e}")
            raise
    
    def imaginary(self, text_prompts: str, vis: bool = True) -> List[Image.Image]:
        """
        Generate images from text prompts using diffusion model
        
        Args:
            text_prompts: Text description
            vis: Whether to save generated images for visualization
            
        Returns:
            List of generated PIL images
        """
        if self.diffusion is None:
            self.logger.warning("Diffusion model not loaded. Loading now...")
            self._quantize_diffusion(self.config['diffusion_id'])
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            results = self.diffusion(
                text_prompts,
                prompt_3=text_prompts,
                height=self.config['gen_height'],
                width=self.config['gen_width'],
                num_inference_steps=28,
                guidance_scale=7.0,
                max_sequence_length=512,
                num_images_per_prompt=self.config['imaginary_num']
            )
        
        if vis:
            for i in range(self.config['imaginary_num']):
                results.images[i].save(f'generated_{i}.png')
        
        return results.images
    
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
    
    def voxel_localize(self, query: Union[str, Image.Image], K: int = 100, 
                    region_radius: float = np.inf, curr_grid: Optional[Tuple[int, int, int]] = None,
                    batch_size: int = 100000, vis: bool = False,
                    filter_height: bool = False,
                    floor_margin: int = 15,
                    ceiling_margin: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Localize voxels based on text or image query
        
        Args:
            query: Text query (str) or PIL Image
            K: Number of top matches to return
            region_radius: Search radius constraint
            curr_grid: Current grid position for local search
            batch_size: Batch size for processing
            vis: Whether to visualize results
            filter_height: Whether to filter out floor and ceiling points
            floor_margin: Number of grid cells above floor to exclude
            ceiling_margin: Number of grid cells below ceiling to exclude
            
        Returns:
            best_pos: Best matching position [x, y, z]
            top_k_positions: Top K positions
            top_k_similarities: Corresponding similarities
        """
        if not self.memory_loaded:
            self.logger.error("Memory not loaded. Please call load_memory() first.")
            return None, None, None
        
        # Initialize last_generated_image
        self.last_generated_image = None
        
        # Handle text queries by generating images
        if isinstance(query, str):
            self.logger.info(f"Generating images for text query: {query}")
            gen_images = self.imaginary(query)
            
            # Save the first generated image as last_generated_image
            if gen_images and len(gen_images) > 0:
                # Convert PIL image to OpenCV format
                first_image = np.array(gen_images[0])
                if first_image.shape[2] == 4:  # RGBA
                    first_image = cv2.cvtColor(first_image, cv2.COLOR_RGBA2BGR)
                elif first_image.shape[2] == 3:  # RGB
                    first_image = cv2.cvtColor(first_image, cv2.COLOR_RGB2BGR)
                self.last_generated_image = first_image
                self.logger.info(f"Saved generated image with shape: {self.last_generated_image.shape}")
            
            query_img_tensors = torch.stack([self.transform(img) for img in gen_images]).to(self.device)
            
            # Clean up diffusion model memory if needed
            if self.diffusion is not None:
                del self.diffusion
                self.diffusion = None
                gc.collect()
                torch.cuda.empty_cache()
        else:
            # Direct image query - save as last_generated_image
            query_np = np.array(query)
            if len(query_np.shape) == 3:
                if query_np.shape[2] == 4:  # RGBA
                    query_np = cv2.cvtColor(query_np, cv2.COLOR_RGBA2BGR)
                elif query_np.shape[2] == 3:  # RGB
                    query_np = cv2.cvtColor(query_np, cv2.COLOR_RGB2BGR)
            self.last_generated_image = query_np
            self.logger.info(f"Saved query image with shape: {self.last_generated_image.shape}")
            
            query_img_tensors = torch.stack([self.transform(query)]).to(self.device)
        
        candidates = []
        self.logger.info("Localizing in voxel memory...")
        
        # Calculate height bounds for filtering if enabled
        if filter_height:
            # Use numpy vectorized operation to get height range efficiently
            if self.max_id > 0:
                # Extract all z-coordinates (height_idx) from grid_rgb_pos
                all_heights = self.grid_rgb_pos[:self.max_id, 2]  # Get all height indices
                
                # Get min and max heights
                min_z = np.min(all_heights)
                max_z = np.max(all_heights)
                
                # Calculate filtering bounds
                z_lower_bound = min_z + floor_margin
                z_upper_bound = max_z - ceiling_margin
                
                # Ensure valid range
                if z_lower_bound >= z_upper_bound:
                    self.logger.warning(f"Invalid height filter range: [{z_lower_bound}, {z_upper_bound}]. Disabling height filtering.")
                    filter_height = False
                else:
                    self.logger.info(f"Height filtering enabled: z range [{min_z}, {max_z}], "
                                f"filtering range [{z_lower_bound}, {z_upper_bound}]")
            else:
                filter_height = False
                self.logger.warning("No voxels found (max_id=0), disabling height filtering")
            
        t1 = time.time()
                
        with torch.no_grad():
            # Extract query features with center-weighted pooling
            query_features_dict = self.dinov2.forward_features(query_img_tensors)
            tokens = query_features_dict['x_norm_patchtokens']
            batch_size_q, num_tokens, feature_dim = tokens.size()
            grid_size = int(math.sqrt(num_tokens))
            
            # Center-weighted averaging
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
            
            # Check if features are preloaded
            if self.preload_features and self.preloaded_feature_data is not None:
                # Use preloaded features for faster retrieval
                self.logger.info("Using preloaded features for batch similarity computation")
                
                # Compute all similarities at once
                all_similarities = F.cosine_similarity(query_features, self.preloaded_feature_data, dim=1)
                
                # Process each grid to find max similarity
                grid_max_similarities = []
                grid_positions = []
                
                for group_name, (start_idx, end_idx, position) in self.feature_index_map.items():
                    # Get max similarity for this grid
                    grid_similarities = all_similarities[start_idx:end_idx]
                    max_similarity = torch.max(grid_similarities).item()
                    
                    grid_max_similarities.append(max_similarity)
                    grid_positions.append(position)
                
                # Convert to numpy arrays for filtering
                grid_max_similarities = np.array(grid_max_similarities)
                grid_positions = np.array(grid_positions)
                
                # Apply filters after similarity computation
                valid_mask = np.ones(len(grid_positions), dtype=bool)
                
                # Height filtering
                if filter_height:
                    height_mask = (grid_positions[:, 2] >= z_lower_bound) & (grid_positions[:, 2] <= z_upper_bound)
                    valid_mask &= height_mask
                    self.logger.info(f"Height filter removed {np.sum(~height_mask)} grids")
                
                # Region radius filtering
                if region_radius != np.inf and curr_grid is not None:
                    dist_sq = ((grid_positions[:, 0] - curr_grid[0])**2 + 
                              (grid_positions[:, 1] - curr_grid[1])**2 + 
                              (grid_positions[:, 2] - curr_grid[2])**2)
                    radius_mask = dist_sq <= region_radius**2
                    valid_mask &= radius_mask
                    self.logger.info(f"Radius filter removed {np.sum(~radius_mask)} grids")
                
                # Filter results
                filtered_similarities = grid_max_similarities[valid_mask]
                filtered_positions = grid_positions[valid_mask]
                
                # Create candidates from filtered results
                candidates = [(sim, tuple(pos)) for sim, pos in zip(filtered_similarities, filtered_positions)]
                
            else:
                # Fall back to original HDF5-based retrieval
                self.logger.info("Using HDF5-based retrieval (features not preloaded)")
                
                with h5py.File(self.feat_path, 'r') as h5f:
                    # Get all group names
                    group_names = list(h5f.keys())
                    
                    # Apply filters
                    filtered_names = []
                    excluded_by_height = 0
                    excluded_by_radius = 0
                    
                    for name in group_names:
                        _, x, y, z = name.split('_')
                        x, y, z = int(x), int(y), int(z)
                        
                        # Height filtering
                        if filter_height:
                            if z < z_lower_bound or z > z_upper_bound:
                                excluded_by_height += 1
                                continue
                        
                        # Region constraint filtering
                        if region_radius != np.inf and curr_grid is not None:
                            dist_sq = (x - curr_grid[0])**2 + (y - curr_grid[1])**2 + (z - curr_grid[2])**2
                            if dist_sq > region_radius**2:
                                excluded_by_radius += 1
                                continue
                        
                        filtered_names.append(name)
                    
                    group_names = filtered_names
                    
                    self.logger.info(f"Filtering results: {len(group_names)} groups remaining "
                                f"(excluded {excluded_by_height} by height, {excluded_by_radius} by radius)")
                    
                    group_names.sort()

                    # Use larger batch size for fewer iterations
                    optimized_batch_size = min(batch_size, len(group_names))

                    for i in range(0, len(group_names), optimized_batch_size):
                        batch_group_names = group_names[i:i + optimized_batch_size]
                        
                        # Pre-allocate lists with estimated size
                        all_features = []
                        group_positions = []
                        total_features = 0
                        
                        # First pass: read all data from HDF5 (minimize I/O operations)
                        batch_data = []
                        for group_name in batch_group_names:
                            if group_name.startswith('grid_'):
                                group = h5f[group_name]
                                # Read features once
                                features_np = group['features'][:]
                                position = tuple(map(int, group_name.split('_')[1:4]))
                                batch_data.append((features_np, position))
                                total_features += len(features_np)
                        
                        # Second pass: convert to tensors and compute similarities
                        if batch_data:
                            # Pre-allocate tensor for all features
                            all_features_tensor = torch.empty((total_features, feature_dim), device=self.device)
                            
                            # Fill tensor and track positions
                            current_idx = 0
                            for features_np, position in batch_data:
                                num_feats = len(features_np)
                                # Direct copy to GPU tensor
                                all_features_tensor[current_idx:current_idx + num_feats] = torch.from_numpy(features_np).to(self.device)
                                group_positions.append((position, num_feats))
                                current_idx += num_feats
                            
                            # Compute all similarities at once
                            similarities = F.cosine_similarity(query_features, all_features_tensor, dim=1)
                            
                            # Extract max similarity per grid
                            start_idx = 0
                            for position, count in group_positions:
                                end_idx = start_idx + count
                                group_similarities = similarities[start_idx:end_idx]
                                max_similarity = torch.max(group_similarities).item()
                                candidates.append((max_similarity, position))
                                start_idx = end_idx

        import heapq
        top_k_candidates = heapq.nlargest(K, candidates, key=lambda x: x[0])
        
        top_k_positions = [pos for _, pos in top_k_candidates]
        top_k_similarities = [sim for sim, _ in top_k_candidates]
        
        self.logger.info(f"Localization completed in {time.time() - t1:.2f}s")
        
        if top_k_positions:
            best_poses, top_k_poses, top_k_similarity = np.array(top_k_positions[0]), np.array(top_k_positions), np.array(top_k_similarities)

            if vis:
                # Add height information to visualization
                if filter_height:
                    self.logger.info(f"Visualization: Height filter applied [z: {z_lower_bound} to {z_upper_bound}]")
                self.visualize_retrieval_results(top_k_poses, "voxel", show_top_k=5)
                self.visualize_2d_retrieval(top_k_poses, "voxel", save_path=self.save_path+'/2D.png')

            cluster_centers, _, _ = self.weighted_cluster_centers(top_k_poses, top_k_similarity)
            print("Extracted Loc Array using voxel memory:", cluster_centers)

            best_pos = np.array([cluster_centers])

            return best_pos, top_k_poses, top_k_similarity
        else:
            self.logger.warning("No matching voxels found after filtering")
            return None, None, None
    
    def clear_preloaded_features(self):
        """
        Clear preloaded features from GPU memory
        Useful when memory is needed for other operations
        """
        if self.preloaded_feature_data is not None:
            self.logger.info("Clearing preloaded features from GPU memory")
            del self.preloaded_feature_data
            self.preloaded_feature_data = None
            self.feature_index_map = {}
            self.grid_positions = []
            gc.collect()
            torch.cuda.empty_cache()
    
    def long_term_memory_retrieval(self, text_prompts: str) -> Optional[np.ndarray]:
        """
        Retrieve objects from long-term memory using GPT
        
        Args:
            text_prompts: Text description of the object
            
        Returns:
            Array of locations [[x, y, z], ...] or None if not found
        """
        if not self.memory_loaded:
            self.logger.error("Memory not loaded. Please call load_memory() first.")
            return None
        
        if self.gpt_client is None:
            self.logger.error("GPT client not provided. Cannot perform GPT-based retrieval.")
            # Fallback to simple keyword matching
            return self._simple_long_term_retrieval(text_prompts)
        
        self.logger.info("Searching long memory with GPT...")
        
        while True:
            # Filter memory if needed (in real robot, we use all features)
            memory_dict = self.long_memory_dict
            
            # Call the GPT-based retrieval function
            answer = long_memory_localized(self.gpt_client, text_prompts, memory_dict)
            print(answer)
            
            if self.pattern_unable.search(answer):
                return None
            
            matches = re.findall(r'\*\*Result\*\*: \((.*?)\)', answer)
            if matches:
                # Extract all Nav Loc N: [x,y,z] patterns from within the Result
                loc_matches = re.findall(r'Nav Loc \d+: \[(\d+),\s*(\d+),\s*(\d+)\]', matches[0])
                if loc_matches:
                    # Convert matches to numpy array of coordinates
                    locs = np.array([[int(x), int(y), int(z)] for x, y, z in loc_matches])
                    self.logger.info(f"Extracted Loc Array using long memory: {locs}")
                    return locs
            # If no matches found, continue loop
            continue
    
    def _simple_long_term_retrieval(self, text_prompt: str) -> Optional[np.ndarray]:
        """Fallback simple keyword-based retrieval"""
        self.logger.info(f"Using simple keyword matching for: {text_prompt}")
        
        keywords = text_prompt.lower().split()
        matching_objects = []
        
        for obj in self.long_memory_dict:
            label_lower = obj['label'].lower()
            if any(keyword in label_lower for keyword in keywords):
                matching_objects.append(obj)
        
        if matching_objects:
            matching_objects.sort(key=lambda x: x['confidence'], reverse=True)
            locs = np.array([obj['loc'] for obj in matching_objects])
            self.logger.info(f"Found {len(locs)} matching objects")
            return locs
        else:
            self.logger.info("No matching objects found in long-term memory")
            return None
    
    def visualize_retrieval_results(self, query_positions: np.ndarray, 
                                  retrieval_type: str = "voxel",
                                  show_top_k: int = 5,
                                  save_path: Optional[str] = None):
        """
        Visualize retrieval results in 3D point cloud
        
        Args:
            query_positions: Array of positions [[x, y, z], ...]
            retrieval_type: Type of retrieval ("voxel" or "long_term")
            show_top_k: Number of top results to highlight
            save_path: Path to save visualization
        """
        self.logger.info("Generating retrieval visualization...")
        
        # Create base point cloud
        valid_points = self.grid_rgb_pos[:self.max_id]
        valid_colors = self.grid_rgb[:self.max_id]
        
        # Convert grid coordinates to world coordinates
        points_world = []
        for i in range(self.max_id):
            row, col, height_idx = valid_points[i]
            x = (col - self.gs // 2) * self.cs
            y = (row - self.gs // 2) * self.cs
            z = (height_idx + self.minh) * self.cs
            points_world.append([x, y, z])
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points_world))
        
        # Convert BGR to RGB colors
        rgb_colors = valid_colors[:, [2, 1, 0]] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(rgb_colors)
        
        # Add retrieval results as highlighted spheres
        spheres = []
        colors = plt.cm.rainbow(np.linspace(0, 1, min(show_top_k, len(query_positions))))
        
        for i, pos in enumerate(query_positions[:show_top_k]):
            # Convert grid position to world coordinates
            row, col, height_idx = pos
            x = (col - self.gs // 2) * self.cs
            y = (row - self.gs // 2) * self.cs
            z = (height_idx + self.minh) * self.cs
            
            # Create sphere at retrieval position
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
            sphere.translate([x, y, z])
            
            # Color based on ranking
            if i == 0:  # Best match in bright red
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
            else:
                sphere.paint_uniform_color(colors[i][:3])
            
            spheres.append(sphere)
        
        # Create coordinate frame
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        
        # Combine geometries
        geometries = [pcd, coordinate_frame] + spheres
        
        # Visualize
        window_name = f"{retrieval_type.capitalize()} Retrieval Results"
        o3d.visualization.draw_geometries(geometries, window_name=window_name)
        
        # Save if requested
        if save_path:
            # Combine all geometries into one
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd += pcd
            
            # Add sphere points
            for sphere in spheres:
                sphere_pcd = sphere.sample_points_uniformly(number_of_points=1000)
                combined_pcd += sphere_pcd
            
            o3d.io.write_point_cloud(save_path, combined_pcd)
            self.logger.info(f"Visualization saved to {save_path}")
    
    def visualize_2d_retrieval(self, query_positions: np.ndarray, 
                              retrieval_type: str = "voxel",
                              save_path: Optional[str] = None):
        """
        Visualize retrieval results on 2D map
        
        Args:
            query_positions: Array of positions [[x, y, z], ...]
            retrieval_type: Type of retrieval ("voxel" or "long_term")
            save_path: Path to save visualization
        """
        # Create a copy of cv_map for visualization
        vis_map = self.cv_map.copy()
        
        # Convert to RGB for matplotlib
        if len(vis_map.shape) == 3:
            vis_map_rgb = cv2.cvtColor(vis_map, cv2.COLOR_BGR2RGB)
        else:
            vis_map_rgb = vis_map
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(vis_map_rgb)
        
        # Plot retrieval results
        colors = plt.cm.rainbow(np.linspace(0, 1, min(10, len(query_positions))))
        
        for i, pos in enumerate(query_positions[:10]):
            row, col, _ = pos
            
            # Best match gets special treatment
            if i == 0:
                marker = '*'
                markersize = 20
                color = 'red'
                label = 'Best Match'
            else:
                marker = 'o'
                markersize = 10
                color = colors[i]
                label = f'Match {i+1}'
            
            ax.plot(col, row, marker, markersize=markersize, color=color, 
                   markeredgecolor='white', markeredgewidth=2, label=label)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add title and labels
        ax.set_title(f'{retrieval_type.capitalize()} Retrieval Results on 2D Map', fontsize=16)
        ax.set_xlabel('X (grid cells)')
        ax.set_ylabel('Y (grid cells)')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"2D visualization saved to {save_path}")
        
        plt.show()
    
    def query(self, query: Union[str, Image.Image], 
              use_long_term: bool = True,
              use_voxel: bool = True,
              visualize: bool = True,
              top_k: int = 100) -> Dict:
        """
        Unified query interface for both voxel and long-term memory
        
        Args:
            query: Text or image query
            use_long_term: Whether to search long-term memory
            use_voxel: Whether to search voxel memory
            visualize: Whether to visualize results
            top_k: Number of top results to return
            
        Returns:
            Dictionary containing retrieval results
        """
        results = {
            'voxel': None,
            'long_term': None,
            'combined': None
        }
        
        # Voxel-based retrieval
        if use_voxel:
            best_pos, top_k_pos, top_k_sim = self.voxel_localize(query, K=top_k)
            if best_pos is not None:
                results['voxel'] = {
                    'best': best_pos,
                    'top_k': top_k_pos,
                    'similarities': top_k_sim
                }
                
                if visualize:
                    self.visualize_retrieval_results(top_k_pos, "voxel", show_top_k=5)
                    self.visualize_2d_retrieval(top_k_pos, "voxel")
        
        # Long-term memory retrieval (text only)
        if use_long_term and isinstance(query, str):
            locs = self.long_term_memory_retrieval(query)
            if locs is not None:
                results['long_term'] = {
                    'locations': locs,
                    'count': len(locs)
                }
                
                if visualize:
                    self.visualize_retrieval_results(locs, "long_term", show_top_k=5)
                    self.visualize_2d_retrieval(locs, "long_term")
        
        # Combine results if both are available
        if results['voxel'] and results['long_term']:
            # Simple combination strategy
            combined_locs = []
            
            # Add voxel results
            for pos in results['voxel']['top_k'][:5]:
                combined_locs.append(pos)
            
            # Add long-term results
            for pos in results['long_term']['locations'][:5]:
                combined_locs.append(pos)
            
            results['combined'] = np.array(combined_locs)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        stats = {
            'total_voxels': self.max_id,
            'total_objects': len(self.long_memory_dict),
            'memory_loaded': self.memory_loaded,
            'save_path': self.save_path,
            'features_preloaded': self.preloaded_feature_data is not None
        }
        
        # Add preloaded features statistics
        if self.preloaded_feature_data is not None:
            stats['preloaded_features_count'] = self.preloaded_feature_data.shape[0]
            stats['preloaded_features_dim'] = self.preloaded_feature_data.shape[1]
            stats['preloaded_gpu_memory_gb'] = (self.preloaded_feature_data.element_size() * 
                                               self.preloaded_feature_data.nelement()) / (1024**3)
        
        # Get token count from HDF5
        if os.path.exists(self.feat_path):
            total_tokens = 0
            with h5py.File(self.feat_path, 'r') as h5f:
                for group_name in h5f:
                    if group_name.startswith('grid_'):
                        group = h5f[group_name]
                        if 'features' in group:
                            total_tokens += group['features'].shape[0]
            stats['total_tokens'] = total_tokens
        
        # Object statistics
        if self.long_memory_dict:
            from collections import defaultdict
            object_counts = defaultdict(int)
            for obj in self.long_memory_dict:
                object_counts[obj['label']] += 1
            stats['object_counts'] = dict(object_counts)
        
        return stats


# Import the long_memory_localized function that uses GPT
def long_memory_localized(client, text_prompts, memory_dict):
    """
    This is a placeholder for the actual GPT-based retrieval function
    The user has already implemented this function
    """
    # This function should be imported from the user's implementation
    pass


def main():
    """Example usage demonstrating preloading feature"""
    from scipy.spatial.transform import Rotation as R
    memory_path = "memory/1"

    # Camera intrinsics
    camera_intrinsics = CameraIntrinsics(
        fx=607.96533203125,
        fy=607.874755859375,
        ppx=428.05804443359375,
        ppy=245.64642333984375,
        width=848,
        height=480
    )
    
    # Camera to base_link transform
    pitch_angle = 0
    camera_rotation = R.from_euler('y', pitch_angle, degrees=True)
    camera_quat = camera_rotation.as_quat()
    
    camera_transform = {
        'translation': [-0.1, 0.0, -1.35],
        'rotation': camera_quat.tolist()
    }
    
    optical_to_ros = R.from_euler('xyz', [-90, 0, -90], degrees=True)
    combined_rotation = camera_rotation * optical_to_ros
    camera_transform['rotation'] = combined_rotation.as_quat().tolist()
    
    # Example: Load memory with feature preloading
    print("\n=== Loading memory with feature preloading ===")
    
    # Initialize with preloading enabled
    memory_with_preload = Memory(
        camera_intrinsics=camera_intrinsics,
        camera_to_base_transform=camera_transform,
        save_path=memory_path,
        need_diffusion=True,
        gpt_client=None,
        preload_features=True  # Enable feature preloading
    )
    
    # Load existing memory (features will be preloaded automatically)
    memory_with_preload.load_memory()
    
    # Get statistics to see memory usage
    stats = memory_with_preload.get_statistics()
    print(f"Memory statistics with preloading: {stats}")
    
    # Perform multiple queries to see speed improvement
    queries = ["a yellow couch", "a white table", "a green plant"]
    
    print("\n=== Running queries with preloaded features ===")
    for query in queries:
        start_time = time.time()
        best_pos, top_k, similarities = memory_with_preload.voxel_localize(query, K=100, vis=False)
        query_time = time.time() - start_time
        if best_pos is not None:
            print(f"Query '{query}' completed in {query_time:.2f}s")
            print(f"Best match similarity: {similarities[0]:.3f}")
    
    # Clear preloaded features if needed
    print("\n=== Clearing preloaded features ===")
    memory_with_preload.clear_preloaded_features()
    
    # Compare with non-preloaded version
    print("\n=== Loading memory without feature preloading ===")
    memory_no_preload = Memory(
        camera_intrinsics=camera_intrinsics,
        camera_to_base_transform=camera_transform,
        save_path=memory_path,
        need_diffusion=True,
        gpt_client=None,
        preload_features=False  # Disable feature preloading
    )
    
    memory_no_preload.load_memory()
    
    print("\n=== Running queries without preloaded features ===")
    for query in queries:
        start_time = time.time()
        best_pos, top_k, similarities = memory_no_preload.voxel_localize(query, K=100, vis=False)
        query_time = time.time() - start_time
        if best_pos is not None:
            print(f"Query '{query}' completed in {query_time:.2f}s")
            print(f"Best match similarity: {similarities[0]:.3f}")


if __name__ == "__main__":
    main()