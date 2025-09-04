#!/usr/bin/env python3
"""
Navigation Demo
Supports category navigation, text navigation, and image navigation modes
"""

import os
import argparse
import numpy as np
import cv2
import torch
import gc
from pathlib import Path
from PIL import Image
from ultralytics import YOLOWorld
import habitat

# Import custom modules
from env import NavEnv, get_objnav_env
from memory_2 import VoxelTokenMemory
from BSCAgent import GESObjectNavRobot
from utils import keyboard_control_fast


def create_demo_args():
    """Create unified argument parser for navigation demo"""
    parser = argparse.ArgumentParser(description='Navigation Demo')
    
    # ========== Demo-specific arguments ==========
    # Navigation mode selection
    parser.add_argument('--nav_mode', type=str, default='category', 
                       choices=['category', 'text', 'image'],
                       help='Navigation mode: category (long-term memory only), text (working memory only), or image')
    
    # Navigation target specification
    parser.add_argument('--target', type=str, default='chair',
                       help='Navigation target: category name for category mode, text description for text mode')
    parser.add_argument('--target_image', type=str, default=None,
                       help='Path to target image for image navigation mode')
    
    # Memory configuration for demo
    parser.add_argument('--build_memory', action='store_true',
                       help='Build memory before navigation')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization during navigation')
    parser.add_argument('--save_trajectory', action='store_true',
                       help='Save navigation trajectory')
    parser.add_argument('--output_dir', type=str, default='./demo_output',
                       help='Output directory for results')
    
    # ========== Base arguments from original code ==========
    # Dataset and scene
    parser.add_argument('--dataset', type=str, default='hm3d', choices=['hm3d', 'mp3d'],
                       help='Dataset to use')
    parser.add_argument('--scene_name', type=str, default='00873-bxsVRursffK',
                       help='Scene name to load')
    parser.add_argument("--dataset_dir", type=str, default='/home/orbit/桌面/Nav-2025/data/scene_datasets/hm3d/hm3d/val') 
    parser.add_argument("--scene_dataset_config_file", type=str, default='/home/orbit/桌面/Nav-2025/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json')

    # Visualization control
    parser.add_argument('--no_vis', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--no_record', action='store_true',
                       help='Disable recording')
    
    # Sensors
    parser.add_argument('--color_sensor', action='store_false')
    parser.add_argument('--depth_sensor', action='store_false')
    parser.add_argument('--semantic_sensor', action='store_false')
    parser.add_argument('--sensor_height', type=float, default=1.5)
    parser.add_argument('--width', type=int, default=680)
    parser.add_argument('--height', type=int, default=680)
    parser.add_argument('--lidar_fov', type=int, default=360)
    parser.add_argument('--depth_img_for_lidar_n', type=int, default=20)
    
    # Actions
    parser.add_argument('--move_forward', type=float, default=0.25)
    parser.add_argument('--move_backward', type=float, default=-0.1)
    parser.add_argument('--turn_left', type=int, default=30)
    parser.add_argument('--turn_right', type=int, default=30)
    
    # Memory parameters
    parser.add_argument('--query_width', type=int, default=224)
    parser.add_argument('--query_height', type=int, default=224)
    parser.add_argument('--gen_width', type=int, default=512)
    parser.add_argument('--gen_height', type=int, default=512)
    parser.add_argument('--imagenary_num', type=int, default=3)
    parser.add_argument('--diffusion_id', type=str, default='stabilityai/stable-diffusion-3.5-medium')
    parser.add_argument('--dino_size', type=str, default='dinov2_vitl14_reg')
    parser.add_argument('--memory_path', type=str, default='/home/orbit/桌面/Nav-2025/memory')
    parser.add_argument('--load_memory_path', type=str, default='/home/orbit/桌面/Nav-2025/memory/demo/hm3d_00873-bxsVRursffK')
    parser.add_argument('--load_single_floor', action='store_true',
                       help='Use single floor mode for memory')
    parser.add_argument('--quite', type=bool, default=True)
    
    # Map parameters
    parser.add_argument('--floor_height', type=float, default=-10.0)
    parser.add_argument('--map_height', type=float, default=10.0)
    parser.add_argument('--cell_size', type=float, default=0.1)
    parser.add_argument('--grid_size', type=float, default=1000)
    
    # Coordinate system
    parser.add_argument('--base_forward_axis', type=list, default=[0, 0, -1])
    parser.add_argument('--base_left_axis', type=list, default=[-1, 0, 0])
    parser.add_argument('--base_up_axis', type=list, default=[0, 1, 0])
    parser.add_argument('--base2cam_rot', type=list, default=[1, 0, 0, 0, -1, 0, 0, 0, -1])
    parser.add_argument('--min_depth', type=float, default=0.1)
    parser.add_argument('--max_depth', type=float, default=10)
    parser.add_argument('--depth_sample_rate', type=int, default=1000)
    
    # Detection
    parser.add_argument('--detect_classes', type=list, 
                       default=['seating', 'chest of drawers', 'bed', 'bathtub', 'clothes', 
                               'toilet', 'stool', 'sofa', 'sink', 'tv monitor', 'picture', 
                               'cushion', 'towel', 'shower', 'counter', 'fireplace', 'chair', 
                               'table', 'gym equipment', 'cabinet', 'plant'])
    parser.add_argument('--detect_conf', type=float, default=0.55)
    
    # Exploration
    parser.add_argument('--explore_first', action='store_true')
    parser.add_argument('--random_move_num', type=float, default=30)
    parser.add_argument('--use_only_working_memory', action='store_true')
    
    # Navigation task
    parser.add_argument('--nav_task', type=str, default='objnav')
    parser.add_argument('--image_hfov', type=int, default=90)
    parser.add_argument('--benchmark_dataset', type=str, default='hm3d')
    parser.add_argument('--eval_episodes', type=int, default=1000)
    parser.add_argument('--max_episode_steps', type=int, default=5000)
    parser.add_argument('--success_distance', type=float, default=1.0)
    
    # Habitat paths
    HABITAT_ROOT_DIR = "/home/orbit/Desktop/Nav-2025/third-party/habitat-lab"
    parser.add_argument('--HM3D_CONFIG_PATH', type=str, 
                       default=f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml")
    parser.add_argument('--MP3D_CONFIG_PATH', type=str,
                       default=f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml")
    parser.add_argument('--HM3D_SCENE_PREFIX', type=str, 
                       default="/home/orbit/Desktop/Nav-2025/data/scene_datasets/hm3d/")
    parser.add_argument('--HM3D_EPISODE_PREFIX', type=str,
                       default="/home/orbit/Desktop/Nav-2025/baselines/Pixel-Navigator/checkpoints/objectnav_hm3d_v2/val/val.json.gz")
    parser.add_argument('--MP3D_SCENE_PREFIX', type=str,
                       default="/home/orbit/Desktop/Nav-2025/data/mp3d/mp3d_habitat/")
    parser.add_argument('--MP3D_EPISODE_PREFIX', type=str,
                       default="/home/orbit/Desktop/Nav-2025/baselines/Pixel-Navigator/checkpoints/objectnav_mp3d_v1/val/val.json.gz")
    
    return parser.parse_args()


class NavigationDemo:
    """Main navigation demo class"""
    
    def __init__(self, args):
        """
        Initialize navigation demo
        
        Args:
            args: Unified arguments
        """
        self.args = args
        
        # Apply demo-specific settings
        self.args.no_vis = not args.visualize
        
        # Set navigation mode specific flags
        if args.nav_mode == 'category':
            # Category mode uses only long-term memory
            self.args.use_only_working_memory = False
        elif args.nav_mode == 'text':
            # Text mode uses only working memory
            self.args.use_only_working_memory = True
        
        # Initialize device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize models
        self._initialize_models()
        
        # Initialize environment and memory
        self._initialize_environment()
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
    
    def _initialize_models(self):
        """Initialize perception models"""
        print("Initializing models...")
        
        # Load DINOv2 for visual features
        self.dinov2 = torch.hub.load(
            'facebookresearch/dinov2', 
            self.args.dino_size, 
            source='github'
        ).to(self.device)
        
        # Load YOLO World for object detection
        self.yolow = YOLOWorld("yolov8x-worldv2.pt").to(self.device)
        self.yolow.set_classes(self.args.detect_classes)
        
        print("Models initialized successfully")
    
    def _initialize_environment(self):
        """Initialize navigation environment and memory"""
        print("Initializing environment...")
        
        # Create Habitat environment if using object navigation
        self.habitat_env = NavEnv(self.args)
        
        # Initialize or load memory
        self._setup_memory()
        
        # Initialize navigation robot
        self.robot = GESObjectNavRobot(
            self.memory, 
            self.habitat_env,
            task='demo'
        )
        
        print("Environment initialized successfully")
    
    def _setup_memory(self):
        """Setup navigation memory"""
        memory_exists = False
        
        # Determine memory path
        if self.args.load_memory_path:
            memory_path = self.args.load_memory_path
            memory_exists = os.path.exists(memory_path)
        else:
            # Generate memory path based on scene
            memory_path = os.path.join(
                self.args.memory_path,
                'demo',
                f"{self.args.dataset}_{self.args.scene_name}"
            )
            memory_exists = os.path.exists(memory_path)
        
        # Initialize memory
        self.memory = VoxelTokenMemory(
            self.args,
            memory_path=memory_path if not memory_exists else None,
            preload_dino=self.dinov2,
            preload_yolo=self.yolow,
            need_diffusion=(self.args.nav_mode == 'text')
        )
        
        # Build or load memory
        if self.args.build_memory or not memory_exists:
            print("Building navigation memory...")
            self._build_memory()
        elif memory_exists:
            print(f"Loading existing memory from {memory_path}")
            self.args.load_memory_path = memory_path
            self.memory.load_memory()
    
    def _build_memory(self):
        """Build navigation memory through exploration"""
        print("Starting memory building process...")
        
        # Choose exploration method
        if self.args.visualize:
            # Interactive exploration with keyboard control
            print("Use keyboard to explore: W(forward), A(left), D(right), Q(quit)")
            self.memory.create_memory()
        else:
            # Automatic exploration
            print("Performing automatic exploration...")
            self.memory.exploring_create_memory()
        
        print("Memory building completed")
    
    def navigate_category(self, category):
        """
        Navigate to object category using long-term memory
        
        Args:
            category: Target object category
        """
        print(f"Starting category navigation to: {category}")
        
        # Reset robot state
        # obs = self.habitat_env.reset(self.args) if hasattr(self.habitat_env, 'reset') else None
        # self.robot.reset(obs)
        
        # Perform navigation using long-term memory
        print("Searching in long-term memory...")
        best_poses = self.robot.long_term_memory_retrival_v2(category)
        
        if best_poses is None:
            print("Target not found in long-term memory")
            return False
        
        # Navigate to found locations
        for i, pose in enumerate(best_poses):
            print(f"Navigating to location {i+1}/{len(best_poses)}")
            
            # Convert grid coordinates to world coordinates
            world_pose = self.robot._grid2loc(pose)
            
            try:
                # Plan and execute path
                path, goal = self.memory.Env.move2point(world_pose)
                if len(path) > 2000:
                    print("Path too long, trying next location")
                    continue
                
                self.robot.execute_path(path[:-1])
                
                # Check if target is visible
                self.robot.check_around(category, max_around=1)
                
                if self.robot.task_over:
                    print("Target found successfully!")
                    self._save_results()
                    return True
                    
            except Exception as e:
                print(f"Navigation failed: {e}")
                continue
        
        print("Target not found after checking all locations")
        return False
    
    def navigate_text(self, text_description):
        """
        Navigate using text description and working memory
        
        Args:
            text_description: Natural language description of target
        """
        print(f"Starting text navigation with description: {text_description}")
        
        # Reset robot state
        # obs = self.habitat_env.reset() if hasattr(self.habitat_env, 'reset') else None
        # self.robot.reset(obs)
        
        # Use working memory for text-based navigation
        print("Searching in working memory with text description...")
        best_poses = self.robot.working_memory_retrival(
            text_description, 
            vis_aug=False,  # Disable visual augmentation for demo
            text_aug=True   # Enable text augmentation
        )
        
        if best_poses is None or len(best_poses[0]) == 0:
            print("No matching locations found in working memory")
            return False
        
        # Navigate to top locations
        query_num = min(len(best_poses[0]), 3)
        for i in range(query_num):
            print(f"Navigating to location {i+1}/{query_num}")
            
            pose = best_poses[0][i]
            world_pose = self.robot._grid2loc(pose)
            
            try:
                path, goal = self.memory.Env.move2point(world_pose)
                if len(path) > 2000:
                    print("Path too long, trying next location")
                    continue
                
                self.robot.execute_path(path[:-1])
                
                # Visual verification
                self.robot.check_around(text_description, max_around=1)
                
                if self.robot.task_over:
                    print("Target found successfully!")
                    self._save_results()
                    return True
                    
            except Exception as e:
                print(f"Navigation failed: {e}")
                continue
        
        print("Target not found after checking all locations")
        return False
    
    def navigate_image(self, image_path):
        """
        Navigate to location matching target image
        
        Args:
            image_path: Path to target image
        """
        print(f"Starting image navigation with target: {image_path}")
        
        # Load target image
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return False
        
        target_image = Image.open(image_path).convert('RGB')
        
        # Reset robot state
        # obs = self.habitat_env.reset() if hasattr(self.habitat_env, 'reset') else None
        # self.robot.reset(obs)
        
        # Use image for navigation
        print("Searching for image match in working memory...")
        best_poses = self.robot.working_memory_retrival(target_image)
        
        if best_poses is None or len(best_poses[0]) == 0:
            print("No matching locations found")
            return False
        
        # Navigate to matched locations
        query_num = min(len(best_poses[0]), 3)
        for i in range(query_num):
            print(f"Navigating to location {i+1}/{query_num}")
            
            pose = best_poses[0][i]
            world_pose = self.robot._grid2loc(pose)
            
            try:
                path, goal = self.memory.Env.move2point(world_pose)
                if len(path) > 2000:
                    print("Path too long, trying next location")
                    continue
                
                self.robot.execute_path(path[:-1])
                
                # Visual verification with image
                self.robot.check_around(target_image, max_around=1)
                
                if self.robot.task_over:
                    print("Target found successfully!")
                    self._save_results()
                    return True
                    
            except Exception as e:
                print(f"Navigation failed: {e}")
                continue
        
        print("Target not found after checking all locations")
        return False
    
    def _save_results(self):
        """Save navigation results"""
        if self.args.save_trajectory:
            print("Saving navigation trajectory...")
            
            # Save trajectory data
            trajectory_path = os.path.join(
                self.args.output_dir, 
                'trajectory.json'
            )
            self.robot.save_log()
            
            # Save final image
            if hasattr(self.robot, 'curr_obs'):
                final_image_path = os.path.join(
                    self.args.output_dir,
                    'final_view.png'
                )
                cv2.imwrite(
                    final_image_path,
                    cv2.cvtColor(self.robot.curr_obs['rgb'], cv2.COLOR_RGB2BGR)
                )
            
            print(f"Results saved to {self.args.output_dir}")
    
    def run(self):
        """Run the navigation demo"""
        success = False
        
        try:
            if self.args.nav_mode == 'category':
                success = self.navigate_category(self.args.target)
            elif self.args.nav_mode == 'text':
                success = self.navigate_text(self.args.target)
            elif self.args.nav_mode == 'image':
                if self.args.target_image:
                    success = self.navigate_image(self.args.target_image)
                else:
                    print("Error: Image path required for image navigation mode")
            
            if success:
                print("\nNavigation completed successfully!")
            else:
                print("\nNavigation failed to find target")
                
        except KeyboardInterrupt:
            print("\nNavigation interrupted by user")
        except Exception as e:
            print(f"\nNavigation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            cv2.destroyAllWindows()
            gc.collect()
            torch.cuda.empty_cache()


def main():
    """Main entry point for navigation demo"""
    
    # Parse unified arguments
    args = create_demo_args()
    
    # Print configuration
    print("="*50)
    print("Navigation Demo Configuration")
    print("="*50)
    print(f"Mode: {args.nav_mode}")
    print(f"Target: {args.target}")
    print(f"Scene: {args.scene_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Visualization: {args.visualize}")
    print(f"Build Memory: {args.build_memory}")
    print(f"Single Floor: {args.load_single_floor}")
    print("="*50)
    
    # Create and run demo
    demo = NavigationDemo(args)
    demo.run()


if __name__ == "__main__":
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    
    main()