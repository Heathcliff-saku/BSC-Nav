import numpy as np
import cv2
import time
import logging
from typing import Tuple, Optional, List, Dict, Union
from PIL import Image
from scipy.spatial import distance
from collections import deque
import math
import threading
from datetime import datetime

# Import all the modules
from camera import OptimizedRealSenseCamera
from lowlevel import RobotNavigationController
from memory import Memory, CameraIntrinsics
from client import map_coordinate_to_png, png_coordinate_to_map
from utils import grid_id_3d2base_pos

class NavAgent:
    """
    高层导航代理，整合记忆检索、路径规划和机器人控制
    
    初始位置需要在实例化时指定，使用PNG地图坐标系
    """
    
    def __init__(self, 
                 ws_url: str,
                 http_url: str, 
                 map_name: str,
                 memory_path: str,
                 initial_pose: Optional[Tuple[float, float, float]] = None,
                 camera_device_index: int = 0,
                 camera_intrinsics: Optional[CameraIntrinsics] = None,
                 camera_to_base_transform: Optional[Dict] = None,
                 config: Optional[Dict] = None,
                 load_memory: bool = True,
                 load_diffusion: bool = True
                 ):
        """
        初始化导航代理
        
        Args:
            ws_url: WebSocket服务器地址
            http_url: HTTP API服务器地址
            map_name: 使用的地图名称
            memory_path: 记忆系统路径
            initial_pose: 初始位置 (x, y, theta) PNG坐标系，如果为None则不设置
            camera_device_index: 相机设备索引
            camera_intrinsics: 相机内参
            camera_to_base_transform: 相机到base_link的变换
            config: 配置参数
        """
        self.logger = logging.getLogger(__name__)
        
        # 保存初始位置
        self.initial_pose = initial_pose
        
        # 默认配置
        self.config = {
            'obstacle_threshold': 100,  # 障碍物判定阈值（灰度值）
            'navigable_color': (254, 254, 254),  # 可导航区域颜色
            'search_radius': 40,  # 搜索可导航点的初始半径（像素）
            'max_search_radius': 100,  # 最大搜索半径（像素）
            'safe_distance': 10,  # 安全距离（像素）
            'navigation_timeout': 300,  # 导航超时时间（秒）
            'position_threshold': 0.5,  # 到达目标的位置阈值（米）
            'stable_duration': 5.0,  # 稳定时间（秒）
            'video_fps': 10,  # 视频帧率
            'recording_interval': 0.1,  # 记录间隔（秒）
            'trajectory_color': (0, 255, 0),  # 轨迹颜色（绿色）
            'agent_color': (255, 0, 0),  # agent颜色（红色）
            'target_color': (0, 0, 255),  # 目标颜色（蓝色）
            'agent_radius': 12,  # agent圆点半径
            'trajectory_thickness': 3,  # 轨迹线条粗细
        }
        if config:
            self.config.update(config)
        
        # 初始化各个模块
        self.logger.info("初始化导航代理...")
        
        # 1. 初始化导航控制器
        self.nav_controller = RobotNavigationController(ws_url, http_url, map_name)
        self.map_name = map_name
        self._load_map_data()

        if not self.nav_controller.is_navigation_active:
            self.logger.info("启动导航系统...")
            if not self.nav_controller.start_navigation(
                map_name=self.map_name,
                initial_pose=self.initial_pose
            ):
                self.logger.error("导航系统启动失败")
                return None
        
        # 2. 初始化相机
        self.camera = OptimizedRealSenseCamera(
            device_index=camera_device_index,
            warmup_frames=30,
            enable_filters=True,
            depth_preset='high_accuracy',
            enable_motion_blur_detection=True
        )
        
        # 3. 初始化记忆系统
        if camera_intrinsics is None:
            # 使用默认相机内参
            camera_intrinsics = CameraIntrinsics(
                fx=607.96533203125,
                fy=607.874755859375,
                ppx=428.05804443359375,
                ppy=245.64642333984375,
                width=848,
                height=480
            )
        
        if camera_to_base_transform is None:
            # 使用默认变换
            from scipy.spatial.transform import Rotation as R
            pitch_angle = 0
            camera_rotation = R.from_euler('y', pitch_angle, degrees=True)
            optical_to_ros = R.from_euler('xyz', [-90, 0, -90], degrees=True)
            combined_rotation = camera_rotation * optical_to_ros
            camera_to_base_transform = {
                'translation': [-0.1, 0.0, -1.35],
                'rotation': combined_rotation.as_quat().tolist()
            }
        
        self.memory = Memory(
            camera_intrinsics=camera_intrinsics,
            camera_to_base_transform=camera_to_base_transform,
            save_path=memory_path,
            need_diffusion=load_diffusion,  # 启用文本查询
            gpt_client=None,  # 如果有GPT客户端可以传入
            preload_features=True
        )
        
        if load_memory:
            # 加载记忆
            self.memory.load_memory()
        
        
        # 5. 初始化记录系统
        self.trajectory_points = []  # 轨迹点列表
        self.observation_data = []   # 观测数据列表
        self.recording_thread = None
        self.is_recording = False
        self.recording_lock = threading.Lock()
    
                
        
        self.logger.info("导航代理初始化完成")
    
    def _load_map_data(self):
        """加载地图数据和信息"""
        # 获取地图信息
        self.map_info = self.nav_controller.http_client.get_map_info(self.map_name)
        if not self.map_info:
            raise ValueError(f"无法获取地图信息: {self.map_name}")
        
        # 下载地图PNG
        self.nav_controller.http_client.get_map_png(self.map_name)
        
        # 加载地图图像
        map_path = f'./map_source/{self.map_name}.png'
        self.map_image = cv2.imread(map_path)
        if self.map_image is None:
            raise ValueError(f"无法加载地图图像: {map_path}")
        
        # 转换为灰度图用于导航分析
        self.map_gray = cv2.cvtColor(self.map_image, cv2.COLOR_BGR2GRAY)
        
        # 保存原始地图用于可视化
        self.original_map = self.map_image.copy()
        
        self.logger.info(f"地图加载成功: {self.map_image.shape}")
    
    def _find_navigable_point(self, target_png: Tuple[int, int], 
                            visualize: bool = False) -> Optional[Tuple[int, int]]:
        """
        寻找最近的可导航点（简化版，不检查连通性）
        
        Args:
            target_png: 目标PNG坐标 (x, y)
            visualize: 是否可视化搜索过程
            
        Returns:
            可导航的PNG坐标，如果找不到返回None
        """
        x, y = int(target_png[0]), int(target_png[1])
        
        # 检查目标点是否已经是可导航的
        if self._is_navigable_point(x, y):
            if self._check_safe_radius(x, y):
                self._last_navigable_point = (x, y)  # 保存可导航点
                return (x, y)
        
        # 使用螺旋搜索最近的可导航点
        visited = set()
        search_radius = self.config['search_radius']
        
        while search_radius <= self.config['max_search_radius']:
            candidates = []
            
            # 在当前半径内搜索
            for angle in np.linspace(0, 2 * np.pi, max(8, int(search_radius))):
                for r in range(0, search_radius, 5):
                    nx = int(x + r * np.cos(angle))
                    ny = int(y + r * np.sin(angle))
                    
                    if (nx, ny) in visited:
                        continue
                    
                    # 边界检查
                    if nx < 0 or nx >= self.map_image.shape[1]:
                        continue
                    if ny < 0 or ny >= self.map_image.shape[0]:
                        continue
                    
                    visited.add((nx, ny))
                    
                    # 检查是否是可导航点
                    if self._is_navigable_point(nx, ny):
                        if self._check_safe_radius(nx, ny):
                            dist = math.sqrt((nx - x)**2 + (ny - y)**2)
                            candidates.append((nx, ny, dist))
            
            # 如果找到候选点，返回最近的
            if candidates:
                best_point = min(candidates, key=lambda p: p[2])
                
                if visualize:
                    self._visualize_search(target_png, best_point[:2], visited)
                
                self._last_navigable_point = best_point[:2]  # 保存可导航点
                return best_point[:2]
            
            # 增加搜索半径
            search_radius += 10
        
        self.logger.warning(f"在最大搜索半径内找不到可导航点: {target_png}")
        return None

    
    def _is_navigable_point(self, x: int, y: int) -> bool:
        """检查点是否可导航"""
        # 检查像素值是否表示可导航区域
        pixel_value = self.map_gray[y, x]
        return pixel_value >= 254  # 白色或接近白色
    
    def _check_safe_radius(self, x: int, y: int) -> bool:
        """检查点周围是否有足够的安全空间"""
        safe_dist = self.config['safe_distance']
        
        # 检查周围的点
        for dx in range(-safe_dist, safe_dist + 1):
            for dy in range(-safe_dist, safe_dist + 1):
                nx, ny = x + dx, y + dy
                
                # 边界检查
                if nx < 0 or nx >= self.map_image.shape[1]:
                    return False
                if ny < 0 or ny >= self.map_image.shape[0]:
                    return False
                
                # 检查是否是障碍物
                if self.map_gray[ny, nx] < self.config['obstacle_threshold']:
                    return False
        
        return True
    
    def _visualize_search(self, target: Tuple[int, int], 
                         found: Tuple[int, int], 
                         visited: set):
        """可视化搜索过程"""
        vis_map = self.map_image.copy()
        
        # 标记访问过的点
        for (x, y) in visited:
            cv2.circle(vis_map, (x, y), 1, (255, 255, 0), -1)
        
        # 标记原始目标
        cv2.circle(vis_map, target, 5, (0, 0, 255), -1)
        cv2.putText(vis_map, "Target", (target[0] + 10, target[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # 标记找到的可导航点
        cv2.circle(vis_map, found, 5, (0, 255, 0), -1)
        cv2.putText(vis_map, "Navigable", (found[0] + 10, found[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 画连线
        cv2.line(vis_map, target, found, (255, 0, 0), 2)
        
        # 显示
        cv2.imshow("Navigation Point Search", vis_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _start_recording(self, target_png: Tuple[int, int]):
        """开始记录导航过程"""
        self.trajectory_points = []
        self.observation_data = []
        self.target_png = target_png
        self.is_recording = True
        
        # 启动记录线程
        self.recording_thread = threading.Thread(target=self._recording_loop)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        
        self.logger.info("开始记录导航过程")
    
    def _stop_recording(self, save_path: Optional[str] = None):
        """停止记录并保存视频"""
        self.is_recording = False
        
        if self.recording_thread:
            self.recording_thread.join()
        
        # 保存视频
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"navigation_video_{timestamp}.mp4"
        
        self._save_navigation_video(save_path)
        self.logger.info(f"导航视频已保存: {save_path}")
    
    def _recording_loop(self):
        """记录循环"""
        while self.is_recording:
            try:
                # 获取当前观测
                color_img, depth_img = self.camera.get_obs(timeout=0.5)
                
                # 获取当前位置
                state = self.nav_controller.get_state()
                current_pos = state['pose']['position']
                current_png = map_coordinate_to_png(
                    [current_pos['x'], current_pos['y']], 
                    self.map_info
                )
                current_png = (int(current_png[0]), int(current_png[1]))
                
                # 记录数据
                with self.recording_lock:
                    self.trajectory_points.append(current_png)
                    self.observation_data.append({
                        'rgb': color_img.copy(),
                        'depth': depth_img.copy(),
                        'position': current_png,
                        'timestamp': time.time()
                    })
                
                # 控制记录频率
                time.sleep(self.config['recording_interval'])
                
            except Exception as e:
                self.logger.warning(f"记录错误: {e}")
                time.sleep(0.1)
    
    def _create_map_visualization(self, current_pos: Tuple[int, int]) -> np.ndarray:
        """创建地图可视化，显示轨迹和当前位置"""
        # 复制原始地图
        vis_map = self.original_map.copy()
        
        # 检查是否使用旋转后的轨迹点
        use_rotated = hasattr(self, 'trajectory_points_rotated') and self.trajectory_points_rotated
        trajectory_to_use = self.trajectory_points_rotated if use_rotated else self.trajectory_points
        
        # 如果使用旋转坐标，也需要转换当前位置和目标位置
        if use_rotated:
            # 转换当前位置
            current_pos = (current_pos[1], self.map_image.shape[1] - current_pos[0])
            
            # 转换目标位置
            if hasattr(self, 'target_png'):
                target_vis = (self.target_png[1], self.map_image.shape[1] - self.target_png[0])
            else:
                target_vis = None
        else:
            target_vis = self.target_png if hasattr(self, 'target_png') else None
        
        # 绘制目标位置
        if target_vis:
            cv2.circle(vis_map, target_vis, 10, self.config['target_color'], -1)
            cv2.putText(vis_map, "Target", 
                    (target_vis[0] + 15, target_vis[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config['target_color'], 2)
        
        # 绘制轨迹
        if len(trajectory_to_use) > 1:
            for i in range(1, len(trajectory_to_use)):
                cv2.line(vis_map, 
                        trajectory_to_use[i-1], 
                        trajectory_to_use[i], 
                        self.config['trajectory_color'], 
                        self.config['trajectory_thickness'])
        
        # 绘制当前位置（大圆点）
        cv2.circle(vis_map, current_pos, 
                self.config['agent_radius'], 
                self.config['agent_color'], -1)
        
        # 添加白色边框使agent更明显
        cv2.circle(vis_map, current_pos, 
                self.config['agent_radius'] + 2, 
                (255, 255, 255), 2)
        
        # 添加文字标注
        cv2.putText(vis_map, "Agent", 
                (current_pos[0] + 15, current_pos[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config['agent_color'], 2)
        
        return vis_map
        
    def _save_navigation_video(self, save_path: str):
        """保存导航视频（新布局版本）"""
        import os
        if not self.observation_data:
            self.logger.warning("没有记录数据，无法保存视频")
            return
        
        # 获取第一帧来确定尺寸
        first_obs = self.observation_data[0]
        rgb_h, rgb_w = first_obs['rgb'].shape[:2]
        map_h, map_w = self.map_image.shape[:2]
        
        # 设置高分辨率参数
        scale_factor = 2  # 提高分辨率的缩放因子
        rgb_h_scaled = rgb_h * scale_factor
        rgb_w_scaled = rgb_w * scale_factor
        
        # 尝试加载生成的图像
        generated_image = None
        generated_path = "./generated_0.png"
        if os.path.exists(generated_path):
            generated_image = cv2.imread(generated_path)
            if generated_image is not None:
                gen_h, gen_w = generated_image.shape[:2]
                self.logger.info(f"加载生成图像: {generated_path}, 尺寸: {gen_h}x{gen_w}")
        
        # 计算布局尺寸
        # 顶部区域（生成图像 + 任务文本）
        top_bar_height = 200  # 顶部横条高度
        gen_image_size = 160   # 生成图像显示大小（正方形）
        
        # 左右分栏
        left_width = rgb_w_scaled  # 左侧宽度（RGB和Depth）
        right_width = rgb_w_scaled  # 右侧宽度（地图）
        total_width = left_width + right_width + 20  # 总宽度（含间隔）
        
        # 左侧上下分布
        left_height = rgb_h_scaled * 2 + 10  # RGB + Depth + 间隔
        
        # 地图区域（右侧全部）
        map_area_width = right_width
        map_area_height = left_height
        
        # 计算地图缩放以适应区域（保持比例）
        map_scale_w = map_area_width / map_w
        map_scale_h = map_area_height / map_h
        map_scale = min(map_scale_w, map_scale_h) * 0.95  # 留出边距
        
        map_display_w = int(map_w * map_scale)
        map_display_h = int(map_h * map_scale)
        
        # 输出视频尺寸
        output_w = total_width
        output_h = top_bar_height + left_height + 20
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.config['video_fps']
        out = cv2.VideoWriter(save_path, fourcc, fps, (output_w, output_h))
        
        self.logger.info(f"生成视频: {len(self.observation_data)} 帧, 尺寸: {output_w}x{output_h}")
        
        # 获取导航目标文本
        navigation_text = "Navigation Task"
        if hasattr(self, 'current_navigation_text'):
            navigation_text = f"Navigate to: {self.current_navigation_text}"
        elif hasattr(self, 'current_navigation_mode'):
            if self.current_navigation_mode == 'text':
                navigation_text = "Text-based Navigation"
            else:
                navigation_text = "Image-based Navigation"
        
        # 计算目标位置（地图坐标）
        if hasattr(self, 'target_png'):
            target_map = png_coordinate_to_map(self.target_png, self.map_info)
        else:
            target_map = None
        
        # 处理每一帧
        for i, obs in enumerate(self.observation_data):
            # 创建输出帧（深色背景）
            frame = np.ones((output_h, output_w, 3), dtype=np.uint8) * 30  # 深灰色背景
            
            # 1. 顶部横条：生成图像 + 任务文本
            # 背景
            cv2.rectangle(frame, (0, 0), (output_w, top_bar_height), (50, 50, 50), -1)
            
            # 生成的图像（左侧）
            gen_x_start = 10
            gen_y_start = (top_bar_height - gen_image_size) // 2
            
            if generated_image is not None:
                # 缩放生成图像到指定大小（保持宽高比）
                gen_aspect = generated_image.shape[1] / generated_image.shape[0]
                if gen_aspect > 1:
                    # 宽图
                    new_w = gen_image_size
                    new_h = int(gen_image_size / gen_aspect)
                else:
                    # 高图
                    new_h = gen_image_size
                    new_w = int(gen_image_size * gen_aspect)
                
                gen_resized = cv2.resize(generated_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # 居中放置
                y_offset = (gen_image_size - new_h) // 2
                x_offset = (gen_image_size - new_w) // 2
                
                # 创建图像容器
                gen_container = np.ones((gen_image_size, gen_image_size, 3), dtype=np.uint8) * 30
                gen_container[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = gen_resized
                
                frame[gen_y_start:gen_y_start+gen_image_size, 
                    gen_x_start:gen_x_start+gen_image_size] = gen_container
                
                # 添加边框
                cv2.rectangle(frame, 
                            (gen_x_start, gen_y_start), 
                            (gen_x_start+gen_image_size, gen_y_start+gen_image_size), 
                            (200, 200, 200), 2)
            
            # 任务文本（右侧）
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_x_start = gen_x_start + gen_image_size + 20
            text_y_center = top_bar_height // 2 + 10
            
            cv2.putText(frame, navigation_text, 
                    (text_x_start, text_y_center), 
                    font, 1.3, (255, 255, 255), 2)
            
            # 2. 左侧上部：RGB图像
            rgb_x_start = 10
            rgb_y_start = top_bar_height + 10
            
            rgb_img = obs['rgb']
            rgb_img_scaled = cv2.resize(rgb_img, (rgb_w_scaled, rgb_h_scaled), interpolation=cv2.INTER_CUBIC)
            
            # 在RGB图像上叠加状态信息
            overlay = rgb_img_scaled.copy()
            
            # 获取当前状态信息
            try:
                state = self.nav_controller.get_state()
                current_pos_map = state['pose']['position']
                current_map = [current_pos_map['x'], current_pos_map['y']]
                
                # 计算距离
                if target_map:
                    distance = np.sqrt((current_map[0] - target_map[0])**2 + 
                                    (current_map[1] - target_map[1])**2)
                    distance_text = f"Distance: {distance:.2f}m"
                else:
                    distance_text = "Distance: N/A"
                
                # 速度信息
                linear_vel = state.get('velocity', {}).get('linear', {})
                speed = np.sqrt(linear_vel.get('x', 0)**2 + linear_vel.get('y', 0)**2)
                speed_text = f"Speed: {speed:.2f}m/s"
                
                # 导航状态
                nav_state = state.get('navigation_state', 'Unknown')
                state_text = f"State: {nav_state}"
                
            except:
                distance_text = "Distance: Error"
                speed_text = "Speed: Error"
                state_text = "State: Error"
            
            # 时间和进度信息
            timestamp = f"Frame: {i+1}/{len(self.observation_data)}"
            progress = (i + 1) / len(self.observation_data)
            progress_text = f"Progress: {progress*100:.1f}%"
            
            # 创建信息背景（半透明黑色）
            info_bg_height = 160
            info_bg = np.zeros((info_bg_height, rgb_w_scaled, 3), dtype=np.uint8)
            cv2.rectangle(info_bg, (0, 0), (rgb_w_scaled, info_bg_height), (0, 0, 0), -1)
            
            # 叠加半透明背景
            alpha = 0.7
            y_start = rgb_h_scaled - info_bg_height
            overlay[y_start:] = cv2.addWeighted(
                overlay[y_start:], 1-alpha, info_bg, alpha, 0
            )
            
            # 绘制状态信息（更大更清晰的字体）
            font_scale = 1.0  # 增大字体
            font_color = (255, 255, 255)
            thickness = 2
            line_height = 35
            
            # 文本位置
            text_x = 20
            y_offset = y_start + 30
            
            # 绘制各行信息
            cv2.putText(overlay, timestamp, (text_x, y_offset), 
                    font, font_scale, font_color, thickness)
            
            y_offset += line_height
            cv2.putText(overlay, distance_text, (text_x, y_offset), 
                    font, font_scale, font_color, thickness)
            
            y_offset += line_height
            cv2.putText(overlay, speed_text, (text_x, y_offset), 
                    font, font_scale, font_color, thickness)
            
            y_offset += line_height
            cv2.putText(overlay, state_text, (text_x, y_offset), 
                    font, font_scale, font_color, thickness)
            
            # 进度条
            progress_bar_y = y_offset + 15
            progress_bar_width = rgb_w_scaled - 40
            progress_bar_height = 12
            
            # 背景
            cv2.rectangle(overlay, (20, progress_bar_y), 
                        (20 + progress_bar_width, progress_bar_y + progress_bar_height), 
                        (80, 80, 80), -1)
            
            # 进度
            cv2.rectangle(overlay, (20, progress_bar_y), 
                        (20 + int(progress_bar_width * progress), progress_bar_y + progress_bar_height), 
                        (0, 255, 0), -1)
            
            # 进度文本
            cv2.putText(overlay, progress_text, 
                    (progress_bar_width // 2 - 40, progress_bar_y - 5), 
                    font, 0.8, font_color, thickness)
            
            # 将RGB图像放入帧中
            frame[rgb_y_start:rgb_y_start+rgb_h_scaled, 
                rgb_x_start:rgb_x_start+rgb_w_scaled] = overlay
            
            # 添加RGB标签
            cv2.putText(frame, "RGB Camera", 
                    (rgb_x_start + 10, rgb_y_start + 30), 
                    font, 0.8, (200, 200, 200), 1)
            
            # 3. 左侧下部：深度图像
            depth_y_start = rgb_y_start + rgb_h_scaled + 10
            
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(obs['depth'], alpha=0.03), 
                cv2.COLORMAP_JET
            )
            depth_scaled = cv2.resize(depth_colormap, (rgb_w_scaled, rgb_h_scaled), 
                                    interpolation=cv2.INTER_CUBIC)
            
            frame[depth_y_start:depth_y_start+rgb_h_scaled, 
                rgb_x_start:rgb_x_start+rgb_w_scaled] = depth_scaled
            
            # 添加深度图标签
            cv2.putText(frame, "Depth Map", 
                    (rgb_x_start + 10, depth_y_start + 30), 
                    font, 0.8, (200, 200, 200), 1)
            
            # 4. 右侧：PNG导航地图
            map_x_start = rgb_x_start + rgb_w_scaled + 10
            map_y_start = rgb_y_start
            
            # 创建地图可视化
            vis_map = self.original_map.copy()
            
            # 绘制轨迹
            if len(self.trajectory_points) > 1:
                for j in range(1, min(i+1, len(self.trajectory_points))):
                    cv2.line(vis_map, 
                            self.trajectory_points[j-1], 
                            self.trajectory_points[j], 
                            self.config['trajectory_color'], 
                            max(2, int(self.config['trajectory_thickness'] / map_scale)))
            
            # 绘制目标位置
            if hasattr(self, 'target_png'):
                cv2.circle(vis_map, self.target_png, 
                        max(10, int(20 / map_scale)), 
                        self.config['target_color'], -1)
                cv2.putText(vis_map, "Target", 
                        (self.target_png[0] + 25, self.target_png[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        max(0.5, 1.0 / map_scale), 
                        self.config['target_color'], 
                        max(1, int(2 / map_scale)))
            
            # 绘制当前位置
            current_pos = obs['position']
            cv2.circle(vis_map, current_pos, 
                    max(8, int(self.config['agent_radius'] / map_scale)), 
                    self.config['agent_color'], -1)
            cv2.circle(vis_map, current_pos, 
                    max(10, int(self.config['agent_radius'] / map_scale + 3)), 
                    (255, 255, 255), max(2, int(3 / map_scale)))
            
            # 缩放地图
            map_resized = cv2.resize(vis_map, (map_display_w, map_display_h), 
                                interpolation=cv2.INTER_AREA)
            
            # 居中放置地图
            map_y_offset = (map_area_height - map_display_h) // 2
            map_x_offset = (map_area_width - map_display_w) // 2
            
            # 创建地图背景
            map_bg = np.ones((map_area_height, map_area_width, 3), dtype=np.uint8) * 20
            map_bg[map_y_offset:map_y_offset+map_display_h, 
                map_x_offset:map_x_offset+map_display_w] = map_resized
            
            # 放置地图
            frame[map_y_start:map_y_start+map_area_height, 
                map_x_start:map_x_start+map_area_width] = map_bg
            
            # 添加地图边框
            cv2.rectangle(frame, 
                        (map_x_start + map_x_offset, map_y_start + map_y_offset), 
                        (map_x_start + map_x_offset + map_display_w, 
                        map_y_start + map_y_offset + map_display_h), 
                        (100, 100, 100), 2)
            
            # 添加地图标签
            cv2.putText(frame, "Navigation Map (Full View)", 
                    (map_x_start + 10, map_y_start + 30), 
                    font, 0.8, (200, 200, 200), 1)
            
            # 添加比例信息
            scale_text = f"Scale: {map_scale:.2f}x"
            cv2.putText(frame, scale_text, 
                    (map_x_start + map_area_width - 120, 
                        map_y_start + map_area_height - 10), 
                    font, 0.6, (200, 200, 200), 1)
            
            # 写入帧
            out.write(frame)
        
        # 释放视频写入器
        out.release()
        self.logger.info(f"视频保存完成: {save_path}")



    def save_navigation_summary(self, save_path: Optional[str] = None):
        """
        保存导航任务的完整可视化结果
        
        Args:
            save_path: 保存路径，如果为None则自动生成
        """
        if not self.trajectory_points or not self.observation_data:
            self.logger.warning("没有导航数据，无法生成可视化结果")
            return
        
        self.logger.info("生成导航路径可视化...")
        
        # 导入必要的库
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Polygon
        from matplotlib.lines import Line2D
        import matplotlib.patheffects as path_effects
        from matplotlib.font_manager import FontProperties
        
        # 设置科研绘图风格
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
        
        # 加载地图作为底图
        map_rgb = cv2.cvtColor(self.original_map, cv2.COLOR_BGR2RGB)
        ax.imshow(map_rgb, alpha=0.9)
        
        # 定义精美的配色方案（科研级别）
        colors = {
            'trajectory': '#2E86AB',      # 深蓝色 - 轨迹
            'agent': '#A23B72',          # 紫红色 - agent位置
            'target_original': '#F18F01', # 橙色 - 原始目标
            'target_navigable': '#C73E1D',# 红色 - 可导航目标
            'search_area': '#90A955',    # 绿色 - 搜索区域
            'text_bg': '#FFFFFF',        # 白色 - 文本背景
            'text_color': '#2C3E50',     # 深灰色 - 文本
            'annotation': '#34495E'      # 深灰色 - 注释
        }
        
        # 1. 绘制搜索区域（如果有）
        if hasattr(self, 'target_png') and hasattr(self, '_last_navigable_point'):
            # 原始目标点
            orig_x, orig_y = self.target_png
            nav_x, nav_y = self._last_navigable_point
            
            # 计算搜索半径
            search_radius = np.sqrt((nav_x - orig_x)**2 + (nav_y - orig_y)**2)
            
            # 绘制搜索圆形区域（半透明）
            search_circle = Circle((orig_x, orig_y), search_radius, 
                                color=colors['search_area'], 
                                alpha=0.15, 
                                linewidth=2,
                                linestyle='--',
                                fill=True,
                                label='Search Area')
            ax.add_patch(search_circle)
            
            # 绘制搜索圆形边界
            search_border = Circle((orig_x, orig_y), search_radius, 
                                color=colors['search_area'], 
                                alpha=0.5, 
                                linewidth=2.5,
                                linestyle='--',
                                fill=False)
            ax.add_patch(search_border)
        
        # 2. 绘制完整轨迹
        if len(self.trajectory_points) > 1:
            # 提取x和y坐标
            traj_x = [p[0] for p in self.trajectory_points]
            traj_y = [p[1] for p in self.trajectory_points]
            
            # 绘制主轨迹线
            ax.plot(traj_x, traj_y, 
                color=colors['trajectory'], 
                linewidth=3.5, 
                alpha=0.8,
                linestyle='-',
                label='Navigation Path',
                zorder=5)
            
            # 添加轨迹阴影效果
            ax.plot(traj_x, traj_y, 
                color='black', 
                linewidth=5, 
                alpha=0.2,
                zorder=4)
        
        # 3. 计算采样间隔（每隔一定距离绘制一个点）
        total_distance = 0
        distances = [0]
        for i in range(1, len(self.trajectory_points)):
            dx = self.trajectory_points[i][0] - self.trajectory_points[i-1][0]
            dy = self.trajectory_points[i][1] - self.trajectory_points[i-1][1]
            dist = np.sqrt(dx*dx + dy*dy)
            total_distance += dist
            distances.append(total_distance)
        
        # 每隔一定像素距离采样一个点
        sample_interval = 100  # 像素
        sampled_indices = [0]  # 始终包含起点
        
        for i in range(1, len(distances)-1):
            if distances[i] - distances[sampled_indices[-1]] >= sample_interval:
                sampled_indices.append(i)
        
        sampled_indices.append(len(self.trajectory_points)-1)  # 始终包含终点
        
        # 4. 绘制采样的agent位置点和时间标注
        for idx, i in enumerate(sampled_indices):
            x, y = self.trajectory_points[i]
            
            # 计算时间（假设帧率恒定）
            time_elapsed = i * self.config['recording_interval']
            time_str = f"{time_elapsed:.1f}s"
            
            # 绘制agent点
            if i == 0:  # 起点
                agent_color = colors['agent']
                marker_size = 180
                marker = 's'  # 方形
                edge_width = 3
            elif i == len(self.trajectory_points)-1:  # 终点
                agent_color = colors['target_navigable']
                marker_size = 200
                marker = 'D'  # 菱形
                edge_width = 3
            else:  # 中间点
                agent_color = colors['agent']
                marker_size = 120
                marker = 'o'
                edge_width = 2
            
            ax.scatter(x, y, 
                    c=agent_color, 
                    s=marker_size, 
                    marker=marker,
                    edgecolors='white',
                    linewidths=edge_width,
                    zorder=10,
                    alpha=0.9)
            
            # 添加时间标注框
            if i == 0:
                label_text = f"Start\n{time_str}"
            elif i == len(self.trajectory_points)-1:
                label_text = f"End\n{time_str}"
            else:
                label_text = time_str
            
            # 创建标注框
            bbox_props = dict(
                boxstyle="round,pad=0.3",
                facecolor=colors['text_bg'],
                edgecolor=colors['annotation'],
                alpha=0.9,
                linewidth=1.5
            )
            
            # 计算标注位置（避免重叠）
            offset_x = 20 if idx % 2 == 0 else -20
            offset_y = -25 if idx % 2 == 0 else 25
            
            annotation = ax.annotate(
                label_text,
                xy=(x, y),
                xytext=(x + offset_x, y + offset_y),
                fontsize=9,
                fontweight='bold',
                color=colors['text_color'],
                ha='center',
                va='center',
                bbox=bbox_props,
                arrowprops=dict(
                    arrowstyle='-',
                    connectionstyle='arc3,rad=0.2',
                    color=colors['annotation'],
                    linewidth=1.5,
                    alpha=0.7
                ),
                zorder=15
            )
        
        # 5. 绘制目标点
        if hasattr(self, 'target_png'):
            # 原始检索位置
            orig_x, orig_y = self.target_png
            ax.scatter(orig_x, orig_y, 
                    c=colors['target_original'], 
                    s=250, 
                    marker='*',
                    edgecolors='white',
                    linewidths=3,
                    zorder=12,
                    label='Retrieved Location')
            
            # 可导航位置
            if hasattr(self, '_last_navigable_point'):
                nav_x, nav_y = self._last_navigable_point
                ax.scatter(nav_x, nav_y, 
                        c=colors['target_navigable'], 
                        s=250, 
                        marker='X',
                        edgecolors='white',
                        linewidths=3,
                        zorder=12,
                        label='Navigable Location')
                
                # 连接线
                ax.plot([orig_x, nav_x], [orig_y, nav_y],
                    color=colors['target_original'],
                    linewidth=2,
                    linestyle=':',
                    alpha=0.7,
                    zorder=6)
        
        # 6. 添加统计信息
        stats_text = f"Total Distance: {total_distance:.1f} pixels\n"
        stats_text += f"Total Time: {len(self.trajectory_points) * self.config['recording_interval']:.1f}s\n"
        stats_text += f"Total Frames: {len(self.trajectory_points)}"
        
        # 创建统计信息框
        stats_box = FancyBboxPatch(
            (0.02, 0.02), 0.2, 0.12,
            boxstyle="round,pad=0.02",
            transform=ax.transAxes,
            facecolor=colors['text_bg'],
            edgecolor=colors['annotation'],
            alpha=0.9,
            linewidth=2
        )
        ax.add_patch(stats_box)
        
        ax.text(0.03, 0.11, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontweight='normal',
            color=colors['text_color'])
        
        # 7. 添加标题和任务信息
        if hasattr(self, 'current_navigation_text'):
            title_text = f"Navigation Task: {self.current_navigation_text}"
        else:
            title_text = "Navigation Path Visualization"
        
        ax.set_title(title_text, 
                    fontsize=18, 
                    fontweight='bold',
                    color=colors['text_color'],
                    pad=20)
        
        # 8. 创建图例
        legend_elements = [
            Line2D([0], [0], color=colors['trajectory'], linewidth=3, 
                label='Navigation Path'),
            Line2D([0], [0], marker='s', color='w', 
                markerfacecolor=colors['agent'], markersize=10, 
                label='Start Position'),
            Line2D([0], [0], marker='D', color='w', 
                markerfacecolor=colors['target_navigable'], markersize=10, 
                label='End Position'),
            Line2D([0], [0], marker='*', color='w', 
                markerfacecolor=colors['target_original'], markersize=12, 
                label='Retrieved Location'),
            Line2D([0], [0], marker='X', color='w', 
                markerfacecolor=colors['target_navigable'], markersize=12, 
                label='Navigable Location'),
            mpatches.Patch(color=colors['search_area'], alpha=0.3, 
                        label='Search Area')
        ]
        
        legend = ax.legend(handles=legend_elements, 
                        loc='upper right',
                        fontsize=10,
                        frameon=True,
                        fancybox=True,
                        shadow=True,
                        framealpha=0.9,
                        edgecolor=colors['annotation'])
        
        # 9. 设置坐标轴
        ax.set_xlabel('X (pixels)', fontsize=12, color=colors['text_color'])
        ax.set_ylabel('Y (pixels)', fontsize=12, color=colors['text_color'])
        
        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color(colors['annotation'])
        ax.spines['bottom'].set_color(colors['annotation'])
        
        # 设置网格
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # 10. 保存图像
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if hasattr(self, 'current_navigation_text'):
                task_name = self.current_navigation_text.replace(' ', '_')[:30]
                save_path = f"navigation_summary_{task_name}_{timestamp}.png"
            else:
                save_path = f"navigation_summary_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        self.logger.info(f"导航可视化结果已保存: {save_path}")
        
        # 同时保存一个简化版本（用于快速预览）
        preview_path = save_path.replace('.png', '_preview.jpg')
        fig_preview, ax_preview = plt.subplots(figsize=(8, 6), dpi=100)
        ax_preview.imshow(map_rgb, alpha=0.9)
        
        # 只绘制轨迹和关键点
        if len(self.trajectory_points) > 1:
            traj_x = [p[0] for p in self.trajectory_points]
            traj_y = [p[1] for p in self.trajectory_points]
            ax_preview.plot(traj_x, traj_y, color=colors['trajectory'], 
                        linewidth=2, alpha=0.8)
        
        # 起点和终点
        if self.trajectory_points:
            start = self.trajectory_points[0]
            end = self.trajectory_points[-1]
            ax_preview.scatter(*start, c=colors['agent'], s=100, marker='s')
            ax_preview.scatter(*end, c=colors['target_navigable'], s=100, marker='D')
        
        ax_preview.axis('off')
        plt.savefig(preview_path, dpi=100, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        self.logger.info(f"预览图已保存: {preview_path}")

    
    def calculate_path_length(self) -> float:
        """
        计算实际移动的路径长度（米）
        
        Returns:
            float: 路径总长度（米）
        """
        if len(self.trajectory_points) < 2:
            return 0.0
        
        total_distance = 0.0
        
        # 遍历轨迹点，计算相邻点之间的距离
        for i in range(1, len(self.trajectory_points)):
            # PNG坐标
            png_point1 = self.trajectory_points[i-1]
            png_point2 = self.trajectory_points[i]
            
            # 转换为地图坐标（米）
            map_point1 = png_coordinate_to_map(png_point1, self.map_info)
            map_point2 = png_coordinate_to_map(png_point2, self.map_info)
            
            # 计算欧几里得距离
            dx = map_point2[0] - map_point1[0]
            dy = map_point2[1] - map_point1[1]
            distance = math.sqrt(dx * dx + dy * dy)
            
            total_distance += distance
        
        return total_distance
    
    def get_navigation_statistics(self) -> Dict[str, float]:
        """
        获取导航统计信息
        
        Returns:
            dict: 包含路径长度、轨迹点数等统计信息
        """
        path_length = self.calculate_path_length()
        
        # 计算平均速度
        if self.observation_data:
            total_time = len(self.observation_data) * self.config['recording_interval']
            avg_speed = path_length / total_time if total_time > 0 else 0
        else:
            total_time = 0
            avg_speed = 0
        
        return {
            'path_length': path_length,  # 米
            'trajectory_points': len(self.trajectory_points),
            'total_time': total_time,  # 秒
            'average_speed': avg_speed,  # 米/秒
            'map_name': self.map_name,
            'target_position': self.target_png if hasattr(self, 'target_png') else None
        }


    def Nav2Text(self, text_prompt: str, 
                visualize: bool = True,
                wait_for_arrival: bool = True,
                record_video: bool = True,
                save_summary: bool = True) -> bool:
        """
        基于文本描述导航到目标位置
        
        Args:
            text_prompt: 文本描述
            visualize: 是否可视化
            wait_for_arrival: 是否等待到达
            record_video: 是否记录导航视频
            save_summary: 是否保存导航总结图
            
        Returns:
            是否成功导航
        """
        self.logger.info(f"开始文本导航: {text_prompt}")
        
        # 保存导航目标文本和模式
        self.current_navigation_text = text_prompt
        self.current_navigation_mode = 'text'
        
        try:
            # ... 原有的导航代码 ...
            
            # 2. 记忆检索
            self.logger.info("执行记忆检索...")
            results = self.memory.query(
                text_prompt, 
                use_long_term=False, 
                use_voxel=True, 
                visualize=visualize
            )
            
            # 保存生成的图像（如果有）
            if hasattr(self.memory, 'last_generated_image'):
                self.memory.last_generated_image = None  # 重置
            
            # 检查是否有生成的图像
            if results.get('generated_image') is not None:
                # 保存生成的图像供视频使用
                self.memory.last_generated_image = results['generated_image']
            elif results.get('voxel') and results['voxel'].get('visualization') is not None:
                # 从可视化结果中获取图像
                self.memory.last_generated_image = results['voxel']['visualization']
            
            # 获取最佳匹配位置
            best_world_pos = None
            if results['voxel'] and results['voxel']['best'] is not None:
                # 从voxel检索获取位置
                best_grid = results['voxel']['best'][0]
                best_world_pos = grid_id_3d2base_pos(
                    self.memory.gs, 
                    self.memory.cs, 
                    best_grid
                )
            elif results['long_term'] and results['long_term']['locations'] is not None:
                # 从长期记忆获取位置
                best_grid = results['long_term']['locations'][0]
                best_world_pos = grid_id_3d2base_pos(
                    self.memory.gs, 
                    self.memory.cs, 
                    best_grid
                )
            
            if best_world_pos is None:
                self.logger.error("记忆检索失败，未找到目标")
                return False
            
            self.logger.info(f"找到目标位置（世界坐标）: {best_world_pos}")
            
            # 3. 坐标转换：世界坐标 -> PNG坐标
            target_png = map_coordinate_to_png(best_world_pos[0][:2], self.map_info)
            target_png = (int(target_png[0]), int(target_png[1]))
            self.logger.info(f"目标PNG坐标: {target_png}")
            
            # 4. 寻找可导航点
            navigable_png = self._find_navigable_point(target_png, visualize=visualize)
            if navigable_png is None:
                self.logger.error("无法找到可导航点")
                return False
            
            self.logger.info(f"可导航PNG坐标: {navigable_png}")
            
            # 5. 转换回地图坐标
            navigable_map = png_coordinate_to_map(navigable_png, self.map_info)
            
            # 6. 开始记录（如果需要）
            if record_video:
                self._start_recording(navigable_png)
            
            # 7. 执行导航
            self.logger.info(f"导航到目标: {navigable_map}")
            success = self.nav_controller.nav_to_point(
                target=(navigable_map[0], navigable_map[1], 0),
                wait_for_result=wait_for_arrival,
                timeout=self.config['navigation_timeout'],
                position_threshold=self.config['position_threshold'],
                stable_duration=self.config['stable_duration']
            )
            
            # 8. 停止记录并保存视频
            if record_video:
                video_name = f"nav_{text_prompt.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                self._stop_recording(video_name)
            
            # 9. 保存导航总结图（如果需要）
            if save_summary and success:
                self.save_navigation_summary()
            
            if success:
                self.logger.info("成功到达目标位置")
                
                # 可选：到达后执行观察
                if visualize:
                    self._observe_target()
            else:
                self.logger.warning("导航失败或超时")
            
            return success
            
        except Exception as e:
            self.logger.error(f"文本导航失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保停止记录
            if self.is_recording:
                self._stop_recording()
            
            return False


    def Nav2Text_grasp(self, target_coords: Tuple[float, float, float],
                       text_prompt: str, 
                       visualize: bool = False,
                       wait_for_arrival: bool = True,
                       record_video: bool = False,
                       save_summary: bool = True) -> bool:
        """
        到达指定目标地点
        
        Args:
            text_prompt: 文本描述
            visualize: 是否可视化
            wait_for_arrival: 是否等待到达
            record_video: 是否记录导航视频
            save_summary: 是否保存导航总结图
            
        Returns:
            是否成功导航
        """
        self.logger.info(f"开始文本导航: {text_prompt}")
        
        # 保存导航目标文本和模式
        self.current_navigation_text = text_prompt
        self.current_navigation_mode = 'text'
        
        try:
            navigable_map = target_coords

            # 为了视频记录，计算PNG坐标
            if record_video:
                target_png = map_coordinate_to_png(target_coords[0:2], self.map_info)
                target_png = (int(target_png[0]), int(target_png[1]))
                self._start_recording(target_png)

            # 执行导航
            self.logger.info(f"导航到目标: {navigable_map}")
            success = self.nav_controller.nav_to_point(
                target=(navigable_map[0], navigable_map[1], navigable_map[2]),
                wait_for_result=wait_for_arrival,
                timeout=self.config['navigation_timeout'],
                position_threshold=self.config['position_threshold'],
                stable_duration=self.config['stable_duration']
                )

            # 停止记录并保存视频
            if record_video:
                video_name = f"nav_direct_{text_prompt.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                self._stop_recording(video_name)

            # 保存导航总结图
            if save_summary and success:
                self.save_navigation_summary()

            if success:
                self.logger.info("成功到达目标位置")
            if visualize:
                self._observe_target()
            else:
                self.logger.warning("导航失败或超时")

            return success
            
        except Exception as e:
            self.logger.error(f"文本导航失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保停止记录
            if self.is_recording:
                self._stop_recording()
            
            return False


    # 更新 Nav2Img 方法，在导航完成后生成可视化
    def Nav2Img(self, query_image: Union[str, Image.Image], 
                visualize: bool = True,
                wait_for_arrival: bool = True,
                record_video: bool = True,
                save_summary: bool = True) -> bool:
        """
        基于图像导航到目标位置
        
        Args:
            query_image: 查询图像（路径或PIL Image）
            visualize: 是否可视化
            wait_for_arrival: 是否等待到达
            record_video: 是否记录导航视频
            save_summary: 是否保存导航总结图
            
        Returns:
            是否成功导航
        """
        self.logger.info("开始图像导航")
        
        # 保存导航模式
        self.current_navigation_text = "target image"
        self.current_navigation_mode = 'image'
        
        try:
            # 加载图像
            if isinstance(query_image, str):
                query_image = Image.open(query_image)
            
            # 保存查询图像作为生成图像（转换为numpy array用于OpenCV）
            query_np = np.array(query_image)
            if len(query_np.shape) == 3:
                if query_np.shape[2] == 4:  # 如果有alpha通道
                    query_np = cv2.cvtColor(query_np, cv2.COLOR_RGBA2BGR)
                elif query_np.shape[2] == 3:  # RGB转BGR
                    query_np = cv2.cvtColor(query_np, cv2.COLOR_RGB2BGR)
            
            # 保存查询图像供视频使用
            if not hasattr(self.memory, 'last_generated_image'):
                self.memory.last_generated_image = None
            self.memory.last_generated_image = query_np
            
            # 1. 启动导航系统
            if not self.nav_controller.is_navigation_active:
                self.logger.info("启动导航系统...")
                if not self.nav_controller.start_navigation(
                    map_name=self.map_name,
                    initial_pose=self.initial_pose
                ):
                    self.logger.error("导航系统启动失败")
                    return False
            
            # 2. 图像检索（只使用voxel memory）
            self.logger.info("执行图像检索...")
            best_pos, top_k_pos, similarities = self.memory.voxel_localize(
                query_image, 
                K=100, 
                vis=visualize
            )
            
            if best_pos is None:
                self.logger.error("图像检索失败，未找到匹配")
                return False
            
            # 转换到世界坐标
            best_world_pos = grid_id_3d2base_pos(
                self.memory.gs, 
                self.memory.cs, 
                best_pos[0]
            )
            
            self.logger.info(f"找到目标位置（世界坐标）: {best_world_pos}")
            
            # 3. 坐标转换和导航（与文本导航相同）
            target_png = map_coordinate_to_png(best_world_pos[0][:2], self.map_info)
            target_png = (int(target_png[0]), int(target_png[1]))
            navigable_png = self._find_navigable_point(target_png, visualize=visualize)
            
            if navigable_png is None:
                self.logger.error("无法找到可导航点")
                return False
            
            navigable_map = png_coordinate_to_map(navigable_png, self.map_info)
            
            # 开始记录
            if record_video:
                self._start_recording(navigable_png)
            
            # 执行导航
            success = self.nav_controller.nav_to_point(
                target=(navigable_map[0], navigable_map[1], 0),
                wait_for_result=wait_for_arrival,
                timeout=self.config['navigation_timeout'],
                position_threshold=self.config['position_threshold'],
                stable_duration=self.config['stable_duration']
            )
            
            # 停止记录
            if record_video:
                video_name = f"nav_img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                self._stop_recording(video_name)
            
            # 保存导航总结图（如果需要）
            if save_summary and success:
                self.save_navigation_summary()
            
            if success:
                self.logger.info("成功到达目标位置")
                if visualize:
                    self._observe_target()
            
            return success
            
        except Exception as e:
            self.logger.error(f"图像导航失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保停止记录
            if self.is_recording:
                self._stop_recording()
            
            return False
        

    def Nav2Category(self, 
                 category: str, 
                 visualize: bool = True,
                 wait_for_arrival: bool = True,
                 record_video: bool = True,
                 save_summary: bool = True,
                 distance_weight: float = 0.7,
                 confidence_weight: float = 0.3,
                 max_distance: float = 20.0) -> bool:
        """
        基于物体类别导航到最佳目标位置
        
        Args:
            category: 目标物体类别 (如 'chair', 'table', 'sofa' 等)
            visualize: 是否可视化
            wait_for_arrival: 是否等待到达
            record_video: 是否记录导航视频
            save_summary: 是否保存导航总结图
            distance_weight: 距离权重 (0-1)
            confidence_weight: 置信度权重 (0-1)
            max_distance: 最大搜索距离（米）
            
        Returns:
            是否成功导航
        """
        self.logger.info(f"开始类别导航: 寻找 '{category}'")
        
        # 保存导航目标文本和模式
        self.current_navigation_text = f"nearest {category}"
        self.current_navigation_mode = 'category'
        
        try:
            # 1. 启动导航系统（如果需要）
            if not self.nav_controller.is_navigation_active:
                self.logger.info("启动导航系统...")
                if not self.nav_controller.start_navigation(
                    map_name=self.map_name,
                    initial_pose=self.initial_pose
                ):
                    self.logger.error("导航系统启动失败")
                    return False
            
            # 2. 获取当前机器人位置
            state = self.nav_controller.get_state()
            current_pos = state['pose']['position']
            current_world_pos = np.array([current_pos['x'], current_pos['y'], current_pos['z']])
            self.logger.info(f"当前位置（世界坐标）: {current_world_pos}")
            
            # 3. 从long_memory中检索指定类别的物体
            category_objects = []
            if hasattr(self.memory, 'long_memory_dict') and self.memory.long_memory_dict:
                for obj in self.memory.long_memory_dict:
                    if obj['label'].lower() == category.lower():
                        category_objects.append(obj)
            
            if not category_objects:
                self.logger.error(f"未找到类别 '{category}' 的物体")
                return False
            
            self.logger.info(f"找到 {len(category_objects)} 个 '{category}' 物体")
            
            # 4. 计算每个候选物体的得分
            candidates = []
            for obj in category_objects:
                # 将网格坐标转换为世界坐标
                grid_pos = np.array(obj['loc']) # [row, col, height_idx]
                world_pos = grid_id_3d2base_pos(
                    self.memory.gs, 
                    self.memory.cs, 
                    grid_pos
                )
                
                # 计算距离
                distance = np.linalg.norm(world_pos[0][:2] - current_world_pos[:2])
                
                # 跳过太远的物体
                if distance > max_distance:
                    continue
                
                # 计算归一化得分
                # 距离得分：越近越好（使用指数衰减）
                distance_score = np.exp(-distance / 5.0)  # 5米作为衰减常数
                
                # 置信度得分：直接使用检测置信度
                confidence_score = obj['confidence']
                
                # 综合得分
                total_score = (distance_weight * distance_score + 
                            confidence_weight * confidence_score)
                
                candidates.append({
                    'object': obj,
                    'world_pos': world_pos[0],
                    'distance': distance,
                    'distance_score': distance_score,
                    'confidence_score': confidence_score,
                    'total_score': total_score
                })
            
            if not candidates:
                self.logger.error(f"没有在 {max_distance}米范围内找到 '{category}'")
                return False
            
            # 5. 选择得分最高的候选物体
            best_candidate = max(candidates, key=lambda x: x['total_score'])
            
            self.logger.info(f"选择最佳目标:")
            self.logger.info(f"  - 位置: {best_candidate['world_pos']}")
            self.logger.info(f"  - 距离: {best_candidate['distance']:.2f}米")
            self.logger.info(f"  - 置信度: {best_candidate['confidence_score']:.2f}")
            self.logger.info(f"  - 总得分: {best_candidate['total_score']:.3f}")
            
            # 6. 可视化候选物体（如果需要）
            if visualize:
                self._visualize_category_candidates(candidates, best_candidate, category)
            
            # 7. 转换到PNG坐标并寻找可导航点
            target_world = best_candidate['world_pos'][:2]
            target_png = map_coordinate_to_png(target_world, self.map_info)
            target_png = (int(target_png[0]), int(target_png[1]))
            
            self.logger.info(f"目标PNG坐标: {target_png}")
            
            # 8. 寻找可导航点
            navigable_png = self._find_navigable_point(target_png, visualize=visualize)
            if navigable_png is None:
                self.logger.error("无法找到可导航点")
                return False
            
            self.logger.info(f"可导航PNG坐标: {navigable_png}")
            
            # 9. 转换回地图坐标
            navigable_map = png_coordinate_to_map(navigable_png, self.map_info)
            
            # 10. 开始记录（如果需要）
            if record_video:
                self._start_recording(navigable_png)
            
            # 11. 执行导航
            self.logger.info(f"导航到目标: {navigable_map}")
            success = self.nav_controller.nav_to_point(
                target=(navigable_map[0], navigable_map[1], 0),
                wait_for_result=wait_for_arrival,
                timeout=self.config['navigation_timeout'],
                position_threshold=self.config['position_threshold'],
                stable_duration=self.config['stable_duration']
            )
            
            # 12. 停止记录并保存视频
            if record_video:
                video_name = f"nav_category_{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                self._stop_recording(video_name)
            
            # 13. 保存导航总结图
            if save_summary and success:
                self.save_navigation_summary()
            
            if success:
                self.logger.info(f"成功到达 '{category}'")
                
                # 可选：到达后执行观察
                if visualize:
                    self._observe_target()
            else:
                self.logger.warning("导航失败或超时")
            
            return success
            
        except Exception as e:
            self.logger.error(f"类别导航失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 确保停止记录
            if self.is_recording:
                self._stop_recording()
            
            return False

    def _visualize_category_candidates(self, candidates: List[Dict], 
                                    best_candidate: Dict, 
                                    category: str):
        """
        可视化类别导航的候选物体
        
        Args:
            candidates: 候选物体列表
            best_candidate: 最佳候选物体
            category: 物体类别
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(f'Category Navigation: {category}', fontsize=16)
        
        # 左图：在地图上显示候选物体
        ax1.imshow(cv2.cvtColor(self.original_map, cv2.COLOR_BGR2RGB))
        ax1.set_title('Candidate Objects on Map')
        
        # 获取当前位置的PNG坐标
        state = self.nav_controller.get_state()
        current_pos = state['pose']['position']
        current_png = map_coordinate_to_png(
            [current_pos['x'], current_pos['y']], 
            self.map_info
        )
        
        # 绘制当前位置
        ax1.plot(current_png[0], current_png[1], 'go', markersize=12, 
                markeredgecolor='white', markeredgewidth=2, label='Current Position')
        
        # 绘制所有候选物体
        for i, cand in enumerate(candidates):
            # 转换到PNG坐标
            png_pos = map_coordinate_to_png(cand['world_pos'][:2], self.map_info)
            
            # 根据得分确定颜色强度
            score_normalized = cand['total_score'] / max(c['total_score'] for c in candidates)
            color_intensity = plt.cm.Reds(0.3 + 0.7 * score_normalized)
            
            # 绘制物体位置
            if cand == best_candidate:
                # 最佳候选用特殊标记
                ax1.plot(png_pos[0], png_pos[1], 'r*', markersize=20, 
                        markeredgecolor='yellow', markeredgewidth=2, label='Best Target')
                # 画圆圈强调
                circle = Circle(png_pos, radius=15, fill=False, edgecolor='red', 
                            linewidth=3, linestyle='--')
                ax1.add_patch(circle)
            else:
                ax1.plot(png_pos[0], png_pos[1], 'o', color=color_intensity, 
                        markersize=10, markeredgecolor='white', markeredgewidth=1)
            
            # 添加标签
            ax1.annotate(f'{i+1}', (png_pos[0]+5, png_pos[1]-5), 
                        fontsize=8, color='black', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        # 画连线从当前位置到最佳目标
        best_png = map_coordinate_to_png(best_candidate['world_pos'][:2], self.map_info)
        ax1.plot([current_png[0], best_png[0]], [current_png[1], best_png[1]], 
                'r--', linewidth=2, alpha=0.5)
        
        ax1.legend()
        ax1.axis('equal')
        
        # 右图：显示候选物体的得分详情
        ax2.axis('off')
        
        # 准备表格数据
        table_data = []
        headers = ['#', 'Distance(m)', 'Confidence', 'Total Score']
        
        for i, cand in enumerate(sorted(candidates, key=lambda x: x['total_score'], reverse=True)):
            is_best = cand == best_candidate
            row_data = [
                f"{'*' if is_best else ''}{i+1}",
                f"{cand['distance']:.2f}",
                f"{cand['confidence_score']:.3f}",
                f"{cand['total_score']:.3f}"
            ]
            table_data.append(row_data)
        
        # 创建表格
        table = ax2.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.1, 0.3, 0.3, 0.3])
        
        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 高亮最佳行
        for i, cand in enumerate(sorted(candidates, key=lambda x: x['total_score'], reverse=True)):
            if cand == best_candidate:
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#ffcccc')
        
        # 添加说明文字
        info_text = f"Total Candidates: {len(candidates)}\n"
        info_text += f"Distance Weight: {self.distance_weight:.1%}\n"
        info_text += f"Confidence Weight: {self.confidence_weight:.1%}\n"
        info_text += f"Max Search Distance: {self.max_distance}m"
        
        ax2.text(0.5, 0.2, info_text, transform=ax2.transAxes,
                fontsize=12, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # 等待用户确认
        input("\n按回车键继续导航...")
        plt.close(fig)
    
    def _get_current_position_png(self) -> Optional[Tuple[int, int]]:
        """获取当前位置的PNG坐标"""
        try:
            state = self.nav_controller.get_state()
            current_pos = state['pose']['position']
            current_png = map_coordinate_to_png(
                [current_pos['x'], current_pos['y']], 
                self.map_info
            )
            return (int(current_png[0]), int(current_png[1]))
        except:
            return None
    
    def _observe_target(self):
        """到达目标后进行观察"""
        self.logger.info("执行目标观察...")
        
        # 获取当前图像
        try:
            color_img, depth_img = self.camera.get_obs(timeout=2.0)
            
            # 显示观察到的图像
            cv2.imshow("Target Observation - RGB", color_img)
            
            # 显示深度图
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            cv2.imshow("Target Observation - Depth", depth_colormap)
            
            self.logger.info("按任意键继续...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.logger.error(f"观察失败: {e}")
    
    def shutdown(self):
        """关闭所有模块"""
        self.logger.info("关闭导航代理...")
        
        # 停止记录
        if self.is_recording:
            self._stop_recording()
        
        # 停止导航
        if hasattr(self, 'nav_controller'):
            self.nav_controller.close()
        
        # 停止相机
        if hasattr(self, 'camera'):
            self.camera.stop()
        
        self.logger.info("导航代理已关闭")
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动清理"""
        self.shutdown()


# 使用示例
if __name__ == "__main__":
    # 配置参数
    ws_url = "ws://192.168.1.102:9090"
    http_url = "http://192.168.1.102/apiUrl"
    map_name = "dongsheng04"
    memory_path = "memory/1"
    
    # 创建导航代理
    with NavAgent(
        ws_url=ws_url,
        http_url=http_url,
        map_name=map_name,
        memory_path=memory_path,
        initial_pose=(373, 697, 0),  # 在PNG地图上的初始位置 (x, y, theta)
        camera_device_index=1
    ) as agent:
        
        # 示例1：文本导航（带视频记录） little robot Wally in the movie Wall-E.
        print("=== 文本导航示例 ===")
        success = agent.Nav2Text(
            "A yellow armchair.",
            visualize=True,
            wait_for_arrival=True,
            record_video=True  # 记录导航视频
        )
        
        if success:
            print("文本导航成功!")
            time.sleep(5)
        
        # 示例2：图像导航（带视频记录）
        # print("\n=== 图像导航示例 ===")
        # 假设有一张查询图像
        # success = agent.Nav2Img(
        #     "query_image.jpg",
        #     visualize=True,
        #     wait_for_arrival=True,
        #     record_video=True
        # )
        
        # 示例3：连续导航
        # print("\n=== 连续导航示例 ===")
        # targets = [
        #     "the coffee table",
        #     "the desk with computer",
        #     "the entrance door"
        # ]
        
        # for target in targets:
        #     print(f"\n导航到: {target}")
        #     if agent.Nav2Text(target, visualize=False, wait_for_arrival=True, record_video=True):
        #         print(f"成功到达: {target}")
        #         time.sleep(3)
        #     else:
        #         print(f"无法到达: {target}")