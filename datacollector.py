import os
import json
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty, PriorityQueue
from typing import Tuple, Dict, Optional, List
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pickle
from collections import deque
import math
import shutil
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import psutil

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@dataclass
class SynchronizedData:
    """同步的数据条目"""
    timestamp: float
    frame_id: int
    robot_pose: Dict
    color_image: np.ndarray
    depth_image: np.ndarray
    time_diff: float
    blur_score: float = 0.0
    is_blurry: bool = False
    
    def __lt__(self, other):
        """用于优先队列排序"""
        return self.timestamp < other.timestamp

class AsyncDataWriter:
    """异步数据写入器 - 避免I/O阻塞主循环"""
    
    def __init__(self, session_dir: str, max_workers: int = 4):
        self.session_dir = session_dir
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.write_queue = Queue(maxsize=100)
        self.is_running = True
        self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.write_thread.start()
        self.pending_writes = 0
        self.write_lock = threading.Lock()
        self.logger = logging.getLogger(__name__ + ".AsyncWriter")
    
    def _write_loop(self):
        """后台写入循环"""
        while self.is_running or not self.write_queue.empty():
            try:
                data = self.write_queue.get(timeout=0.1)
                self.executor.submit(self._write_data, data)
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"写入循环错误: {e}")
    
    def _write_data(self, data: Dict):
        """实际写入数据"""
        try:
            # 写入图像
            cv2.imwrite(data['color_path'], data['color_image'])
            cv2.imwrite(data['depth_path'], data['depth_image'])
            
            # 保存位姿数据为单独的文件
            if 'pose_path' in data and 'pose_data' in data:
                # 转换numpy类型为Python原生类型
                pose_data_converted = self._convert_numpy_types(data['pose_data'])
                with open(data['pose_path'], 'w') as f:
                    json.dump(pose_data_converted, f, indent=2)
            
            with self.write_lock:
                self.pending_writes -= 1
                
        except Exception as e:
            self.logger.error(f"写入数据失败: {e}")
    
    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型"""
        import numpy as np
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    def write_async(self, sync_data: SynchronizedData, color_path: str, depth_path: str, pose_path: str = None):
        """异步写入数据"""
        with self.write_lock:
            self.pending_writes += 1
            
        write_data = {
            'color_image': sync_data.color_image,
            'depth_image': sync_data.depth_image,
            'color_path': color_path,
            'depth_path': depth_path
        }
        
        # 如果需要保存位姿数据
        if pose_path:
            # 计算yaw角
            orientation = sync_data.robot_pose['pose']['orientation']
            yaw = math.atan2(2 * (orientation['w'] * orientation['z'] + orientation['x'] * orientation['y']), 
                           1 - 2 * (orientation['y'] * orientation['y'] + orientation['z'] * orientation['z']))
            
            pose_data = {
                'timestamp': float(sync_data.timestamp),
                'frame_id': int(sync_data.frame_id),
                'position': sync_data.robot_pose['pose']['position'],
                'orientation': sync_data.robot_pose['pose']['orientation'],
                'yaw_radians': float(yaw),
                'yaw_degrees': float(math.degrees(yaw)),
                'time_diff': float(sync_data.time_diff)
            }
            write_data['pose_path'] = pose_path
            write_data['pose_data'] = pose_data
        
        # 非阻塞入队
        try:
            self.write_queue.put_nowait(write_data)
        except:
            # 队列满，等待
            self.write_queue.put(write_data, timeout=1.0)
    
    def wait_completion(self, timeout: float = 30.0):
        """等待所有写入完成"""
        start_time = time.time()
        while self.pending_writes > 0:
            if time.time() - start_time > timeout:
                self.logger.warning(f"等待写入超时，剩余: {self.pending_writes}")
                break
            time.sleep(0.1)
    
    def shutdown(self):
        """关闭写入器 - 兼容旧版本Python"""
        self.is_running = False
        self.write_thread.join(timeout=5.0)
        
        # Python版本兼容性处理
        try:
            # Python 3.9+ 支持timeout参数
            self.executor.shutdown(wait=True, timeout=10.0)
        except TypeError:
            # 旧版本Python不支持timeout参数
            self.executor.shutdown(wait=True)

class OptimizedRealTimeDataCollector:
    """
    优化的实时数据收集器 - 流畅采集，无阻塞
    """
    
    def __init__(self, camera, nav_controller, save_dir: str = "realtime_mapping_data",
                 collection_interval: float = 0.5, max_time_diff: float = 0.1,
                 blur_threshold: float = 100.0, enable_quality_filter: bool = True,
                 save_individual_poses: bool = True):
        """
        初始化优化的实时数据收集器
        
        Args:
            camera: RealSense相机实例
            nav_controller: 机器人导航控制器
            save_dir: 数据保存目录
            collection_interval: 采集间隔（秒）
            max_time_diff: 最大允许的时间差（秒）
            blur_threshold: 模糊阈值
            enable_quality_filter: 是否启用质量过滤
            save_individual_poses: 是否保存单独的位姿文件
        """
        self.logger = logging.getLogger(__name__)
        self.camera = camera
        self.nav_controller = nav_controller
        
        # 创建保存目录
        self.save_dir = save_dir
        self.session_dir = None
        self._create_session_dir()
        
        # 异步写入器
        self.data_writer = AsyncDataWriter(self.session_dir)
        
        # 数据收集控制
        self.is_collecting = False
        self.collection_thread = None
        self.frame_counter = 0
        self._stopping = False
        
        # 收集参数
        self.collection_interval = collection_interval
        self.max_time_diff = max_time_diff
        self.blur_threshold = blur_threshold
        self.enable_quality_filter = enable_quality_filter
        self.save_individual_poses = save_individual_poses
        
        # 设置相机模糊阈值
        if hasattr(camera, 'set_blur_threshold'):
            camera.set_blur_threshold(blur_threshold)
        
        # 数据缓冲区
        self.pose_buffer = deque(maxlen=200)  # 增大缓冲区
        self.pose_buffer_lock = threading.Lock()
        
        # 位姿更新线程
        self.pose_update_thread = None
        self.pose_update_interval = 0.01  # 100Hz更新频率
        
        # 数据索引
        self.data_index = []
        self.data_index_lock = threading.Lock()
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'successful_frames': 0,
            'failed_frames': 0,
            'blurry_frames': 0,
            'avg_time_diff': 0.0,
            'max_time_diff_recorded': 0.0,
            'avg_collection_time': 0.0,
            'max_collection_time': 0.0
        }
        
        # 信号处理
        self.original_sigint = signal.signal(signal.SIGINT, self._signal_handler)
        self.stop_event = threading.Event()
        self.signal_received = False
        self._in_main_thread = threading.current_thread() is threading.main_thread()
    
    def _signal_handler(self, signum, frame):
        """处理Ctrl+C信号"""
        if not self.signal_received:
            self.signal_received = True
            self.logger.info("\n收到停止信号 (Ctrl+C)，正在停止数据收集...")
            # 只设置停止事件，不修改is_collecting
            # 这样stop_collection()可以正常执行
            self.stop_event.set()
            # 不抛出异常，避免程序直接跳到except块
    
    def _create_session_dir(self):
        """创建本次会话的数据目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.save_dir, f"realtime_session_{timestamp}")
        
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "color"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, "poses"), exist_ok=True)  # 新增位姿目录
        
        self.logger.info(f"数据将保存到: {self.session_dir}")
    
    def start_collection(self):
        """开始数据收集"""
        if self.is_collecting:
            self.logger.warning("数据收集已经在进行中")
            return
        
        self.is_collecting = True
        self.stop_event.clear()
        self.frame_counter = 0
        self.data_index = []
        self._stopping = False
        
        # 启动位姿更新线程
        self.pose_update_thread = threading.Thread(target=self._optimized_pose_update_loop)
        self.pose_update_thread.daemon = True
        self.pose_update_thread.start()
        
        # 等待位姿缓冲区填充
        time.sleep(0.5)
        
        # 启动收集线程
        self.collection_thread = threading.Thread(target=self._optimized_collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        # 启动性能监控
        self.performance_monitor.start()
        
        self.logger.info("优化的实时数据收集已启动 (按 Ctrl+C 停止)")
    
    def _optimized_pose_update_loop(self):
        """优化的位姿更新循环 - 高频率，低延迟"""
        last_pose_time = 0
        
        while self.is_collecting and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # 避免过于频繁的查询
                if current_time - last_pose_time < self.pose_update_interval:
                    time.sleep(0.001)
                    continue
                
                # 获取当前机器人状态
                robot_state = self.nav_controller.get_state()
                if robot_state and robot_state.get('last_update'):
                    timestamp = time.time()
                    
                    with self.pose_buffer_lock:
                        self.pose_buffer.append({
                            'timestamp': timestamp,
                            'state': robot_state.copy()
                        })
                    
                    last_pose_time = current_time
                
            except Exception as e:
                self.logger.error(f"位姿更新错误: {e}")
                time.sleep(0.01)
    
    def _find_interpolated_pose(self, target_timestamp: float) -> Tuple[Optional[Dict], float]:
        """
        使用插值找到目标时间戳的位姿
        
        Returns:
            tuple: (插值后的位姿数据, 时间差)
        """
        with self.pose_buffer_lock:
            if len(self.pose_buffer) < 2:
                return None, float('inf')
            
            # 找到目标时间戳前后的位姿
            poses = list(self.pose_buffer)
            
            # 二分查找
            left, right = 0, len(poses) - 1
            while left < right - 1:
                mid = (left + right) // 2
                if poses[mid]['timestamp'] < target_timestamp:
                    left = mid
                else:
                    right = mid
            
            # 获取前后位姿
            pose1 = poses[left]
            pose2 = poses[right]
            
            # 如果目标时间戳在范围外，返回最近的
            if target_timestamp <= pose1['timestamp']:
                return pose1, abs(target_timestamp - pose1['timestamp'])
            elif target_timestamp >= pose2['timestamp']:
                return pose2, abs(target_timestamp - pose2['timestamp'])
            
            # 线性插值
            t1, t2 = pose1['timestamp'], pose2['timestamp']
            alpha = (target_timestamp - t1) / (t2 - t1)
            
            # 插值位置
            pos1 = pose1['state']['pose']['position']
            pos2 = pose2['state']['pose']['position']
            
            interpolated_pos = {
                'x': pos1['x'] + alpha * (pos2['x'] - pos1['x']),
                'y': pos1['y'] + alpha * (pos2['y'] - pos1['y']),
                'z': pos1['z'] + alpha * (pos2['z'] - pos1['z'])
            }
            
            # 四元数球面线性插值(SLERP)会更准确，但这里简化处理
            # 直接使用最近的方向
            interpolated_state = pose1['state'].copy()
            interpolated_state['pose']['position'] = interpolated_pos
            
            return {'timestamp': target_timestamp, 'state': interpolated_state}, 0.0
    
    def _optimized_collection_loop(self):
        """优化的数据收集循环 - 无阻塞，高性能"""
        last_collection_time = 0
        last_position = None
        last_yaw = None
        min_movement = 0.05
        min_rotation = math.radians(5)
        
        # 性能统计
        collection_times = deque(maxlen=100)
        
        while self.is_collecting and not self.stop_event.is_set():
            collection_start = time.time()
            
            try:
                current_time = time.time()
                
                # 控制采集频率
                time_since_last = current_time - last_collection_time
                if time_since_last < self.collection_interval:
                    time.sleep(0.005)
                    continue
                
                # 获取相机数据（非阻塞）
                camera_result = self.camera.get_latest_obs(skip_blurry=self.enable_quality_filter)
                
                if camera_result is None:
                    # 没有新数据，继续等待
                    time.sleep(0.01)
                    continue
                
                color_img, depth_img, metadata = camera_result
                camera_timestamp = metadata.get('timestamp', time.time())
                
                # 记录模糊帧统计
                if metadata.get('is_blurry', False):
                    self.stats['blurry_frames'] += 1
                    if self.enable_quality_filter:
                        self.logger.debug(f"跳过模糊帧 (blur_score: {metadata.get('blur_score', 0):.1f})")
                        continue
                
                # 获取插值后的位姿
                closest_pose, time_diff = self._find_interpolated_pose(camera_timestamp)
                
                if closest_pose is None:
                    self.logger.warning("没有可用的位姿数据")
                    self.stats['failed_frames'] += 1
                    continue
                
                # 检查时间差
                if time_diff > self.max_time_diff:
                    self.logger.warning(f"时间差过大: {time_diff:.3f}s > {self.max_time_diff}s")
                    self.stats['failed_frames'] += 1
                    continue
                
                # 获取当前位姿
                robot_state = closest_pose['state']
                current_pos = np.array([
                    robot_state['pose']['position']['x'],
                    robot_state['pose']['position']['y']
                ])
                
                # 计算当前yaw角
                current_yaw = self._quaternion_to_yaw(robot_state['pose']['orientation'])
                
                # 运动检测
                should_collect = False
                movement_info = ""
                
                if last_position is not None and last_yaw is not None:
                    position_change = np.linalg.norm(current_pos - last_position)
                    yaw_change = abs(self._normalize_angle(current_yaw - last_yaw))
                    
                    if position_change >= min_movement or yaw_change >= min_rotation:
                        should_collect = True
                        movement_info = f"位移: {position_change:.3f}m, 旋转: {math.degrees(yaw_change):.1f}°"
                else:
                    should_collect = True
                    movement_info = "初始帧"
                
                if not should_collect:
                    continue
                
                # 创建同步数据
                sync_data = SynchronizedData(
                    timestamp=camera_timestamp,
                    frame_id=self.frame_counter,
                    robot_pose=robot_state,
                    color_image=color_img,
                    depth_image=depth_img,
                    time_diff=time_diff,
                    blur_score=float(metadata.get('blur_score', 0)),
                    is_blurry=bool(metadata.get('is_blurry', False))
                )
                
                # 异步保存数据
                self._save_synchronized_data_async(sync_data)
                
                # 更新统计
                self.stats['successful_frames'] += 1
                self.stats['avg_time_diff'] = float(
                    (self.stats['avg_time_diff'] * (self.stats['successful_frames'] - 1) + time_diff) /
                    self.stats['successful_frames']
                )
                self.stats['max_time_diff_recorded'] = float(max(self.stats['max_time_diff_recorded'], time_diff))
                
                # 记录采集时间
                collection_time = time.time() - collection_start
                collection_times.append(collection_time)
                
                # 更新状态
                last_collection_time = current_time
                last_position = current_pos
                last_yaw = current_yaw
                
                # 性能日志
                self.performance_monitor.record_frame(collection_time)
                
                # 简化日志输出（降低频率）
                if self.frame_counter % 10 == 0:  # 每10帧输出一次
                    avg_collection_time = np.mean(collection_times) if collection_times else 0
                    self.logger.info(f"采集帧 {self.frame_counter} | "
                                   f"位置: [{current_pos[0]:.2f}, {current_pos[1]:.2f}] | "
                                   f"朝向: {math.degrees(current_yaw):.1f}° | "
                                   f"平均采集时间: {float(avg_collection_time)*1000:.1f}ms")
                
            except Exception as e:
                self.logger.error(f"数据收集循环错误: {e}")
                self.stats['failed_frames'] += 1
                time.sleep(0.1)
            
            finally:
                self.stats['total_frames'] += 1
    
    def _save_synchronized_data_async(self, sync_data: SynchronizedData):
        """异步保存同步的数据"""
        # 生成文件路径
        color_path = os.path.join(self.session_dir, "color", f"frame_{sync_data.frame_id:06d}.jpg")
        depth_path = os.path.join(self.session_dir, "depth", f"frame_{sync_data.frame_id:06d}.png")
        pose_path = None
        
        # 如果需要保存单独的位姿文件
        if self.save_individual_poses:
            pose_path = os.path.join(self.session_dir, "poses", f"pose_{sync_data.frame_id:06d}.json")
        
        # 异步写入图像和位姿
        self.data_writer.write_async(sync_data, color_path, depth_path, pose_path)
        
        # 获取相机内参
        intrinsics = self.camera.get_intrinsic()
        
        # 计算yaw角
        yaw = self._quaternion_to_yaw(sync_data.robot_pose['pose']['orientation'])
        
        # 创建数据条目
        data_entry = {
            'timestamp': float(sync_data.timestamp),
            'frame_id': int(sync_data.frame_id),
            'robot_pose': sync_data.robot_pose,
            'yaw_degrees': float(math.degrees(yaw)),
            'color_image_path': color_path,
            'depth_image_path': depth_path,
            'camera_intrinsics': intrinsics,
            'time_diff': float(sync_data.time_diff),
            'blur_score': float(sync_data.blur_score),
            'quality_metrics': {
                'is_blurry': bool(sync_data.is_blurry),
                'blur_score': float(sync_data.blur_score)
            }
        }
        
        # 线程安全地添加到索引
        with self.data_index_lock:
            self.data_index.append(data_entry)
            
        self.frame_counter += 1
    
    def _quaternion_to_yaw(self, orientation: Dict) -> float:
        """从四元数提取yaw角（弧度）"""
        x = orientation.get('x', 0)
        y = orientation.get('y', 0)
        z = orientation.get('z', 0)
        w = orientation.get('w', 1)
        
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw
    
    def _normalize_angle(self, angle: float) -> float:
        """将角度标准化到[-π, π]范围"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        else:
            return obj
    
    def stop_collection(self):
        """停止数据收集"""
        # 避免重复停止
        if self._stopping:
            return
        self._stopping = True
            
        self.is_collecting = False
        self.stop_event.set()
        
        # 停止性能监控
        self.performance_monitor.stop()
        
        # 等待线程结束
        if self.pose_update_thread and self.pose_update_thread.is_alive():
            self.pose_update_thread.join(timeout=2.0)
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)
        
        # 等待所有异步写入完成
        self.logger.info("等待所有数据写入完成...")
        self.data_writer.wait_completion()
        self.data_writer.shutdown()
        
        # 保存数据索引和统计信息
        if self.data_index:
            # 转换numpy类型
            data_index_converted = self._convert_numpy_types(self.data_index)
            
            # 保存完整的数据索引
            index_path = os.path.join(self.session_dir, "data_index.json")
            with open(index_path, 'w') as f:
                json.dump(data_index_converted, f, indent=2)
            
            # 保存简化的位姿列表（方便快速查看）
            poses_summary = []
            for entry in data_index_converted:
                pose = entry['robot_pose']['pose']
                poses_summary.append({
                    'frame_id': entry['frame_id'],
                    'timestamp': entry['timestamp'],
                    'x': pose['position']['x'],
                    'y': pose['position']['y'],
                    'z': pose['position']['z'],
                    'yaw_degrees': entry['yaw_degrees']
                })
            
            poses_summary_path = os.path.join(self.session_dir, "poses_summary.json")
            with open(poses_summary_path, 'w') as f:
                json.dump(poses_summary, f, indent=2)
            
            # 获取性能统计
            perf_stats = self.performance_monitor.get_stats()
            
            # 更新统计信息
            self.stats.update(perf_stats)
            
            # 转换并保存统计信息
            stats_converted = self._convert_numpy_types(self.stats)
            stats_path = os.path.join(self.session_dir, "collection_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats_converted, f, indent=2)
            
            # 过滤低质量数据
            if self.enable_quality_filter:
                self._post_process_quality_filter()
            
            self.logger.info(f"\n数据收集完成:")
            self.logger.info(f"  - 总帧数: {self.stats['total_frames']}")
            self.logger.info(f"  - 成功帧数: {self.stats['successful_frames']}")
            self.logger.info(f"  - 失败帧数: {self.stats['failed_frames']}")
            self.logger.info(f"  - 模糊帧数: {self.stats['blurry_frames']}")
            self.logger.info(f"  - 平均时间差: {self.stats['avg_time_diff']:.3f}s")
            self.logger.info(f"  - 最大时间差: {self.stats['max_time_diff_recorded']:.3f}s")
            self.logger.info(f"  - 平均采集时间: {perf_stats.get('avg_frame_time', 0)*1000:.1f}ms")
            self.logger.info(f"  - 数据保存到: {self.session_dir}")
        else:
            self.logger.warning("没有收集到任何数据")
        
        # 只在主线程中恢复原始信号处理
        if self._in_main_thread and threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.original_sigint)
        
        # 重置停止标志
        self._stopping = False
    
    def _post_process_quality_filter(self):
        """后处理质量过滤 - 移除低质量数据"""
        self.logger.info("执行后处理质量过滤...")
        
        filtered_index = []
        removed_count = 0
        
        for entry in self.data_index:
            # 基于多个指标过滤
            quality_metrics = entry.get('quality_metrics', {})
            
            # 过滤条件
            if (quality_metrics.get('blur_score', 0.0) < self.blur_threshold * 0.5 or  # 严重模糊
                entry.get('time_diff', 0.0) > self.max_time_diff * 2):  # 时间差过大
                
                # 删除对应的图像文件
                try:
                    os.remove(entry['color_image_path'])
                    os.remove(entry['depth_image_path'])
                    # 删除位姿文件
                    if self.save_individual_poses:
                        pose_path = os.path.join(self.session_dir, "poses", f"pose_{entry['frame_id']:06d}.json")
                        if os.path.exists(pose_path):
                            os.remove(pose_path)
                    removed_count += 1
                except:
                    pass
            else:
                filtered_index.append(entry)
        
        # 更新索引
        self.data_index = filtered_index
        
        # 保存过滤后的索引
        filtered_index_converted = self._convert_numpy_types(filtered_index)
        index_path = os.path.join(self.session_dir, "data_index_filtered.json")
        with open(index_path, 'w') as f:
            json.dump(filtered_index_converted, f, indent=2)
        
        self.logger.info(f"质量过滤完成，移除 {removed_count} 帧低质量数据")

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.frame_times = deque(maxlen=1000)
        self.start_time = None
        self.is_monitoring = False
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.monitor_thread = None
        try:
            self.process = psutil.Process()
        except:
            self.process = None
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self.is_monitoring = True
        if self.process:
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # CPU使用率
                cpu = self.process.cpu_percent(interval=0.1)
                self.cpu_usage.append(cpu)
                
                # 内存使用
                memory = self.process.memory_info().rss / 1024 / 1024  # MB
                self.memory_usage.append(memory)
                
                time.sleep(1.0)
            except:
                pass
    
    def record_frame(self, frame_time: float):
        """记录帧处理时间"""
        self.frame_times.append(frame_time)
    
    def stop(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.frame_times:
            return {}
        
        return {
            'avg_frame_time': float(np.mean(self.frame_times)),
            'max_frame_time': float(np.max(self.frame_times)),
            'min_frame_time': float(np.min(self.frame_times)),
            'total_runtime': float(time.time() - self.start_time) if self.start_time else 0.0,
            'avg_cpu_usage': float(np.mean(self.cpu_usage)) if self.cpu_usage else 0.0,
            'max_cpu_usage': float(np.max(self.cpu_usage)) if self.cpu_usage else 0.0,
            'avg_memory_mb': float(np.mean(self.memory_usage)) if self.memory_usage else 0.0,
            'max_memory_mb': float(np.max(self.memory_usage)) if self.memory_usage else 0.0
        }

# 向后兼容
RealTimeDataCollector = OptimizedRealTimeDataCollector

def main_realtime_mapping():
    """优化的实时建图主函数"""
    from camera import OptimizedRealSenseCamera
    from lowlevel import RobotNavigationController
    
    # 配置参数
    ws_url = "ws://192.168.1.102:9090"
    http_url = "http://192.168.1.102/apiUrl"
    map_name = "exp-4"
    
    # 初始化组件
    print("初始化相机...")
    camera = OptimizedRealSenseCamera(
        device_index=1, 
        warmup_frames=60,
        depth_preset='high_accuracy',  # 使用高精度预设
        enable_motion_blur_detection=True
    )
    
    print("连接机器人...")
    nav_controller = None
    collector = None
    
    try:
        nav_controller = RobotNavigationController(ws_url, http_url, map_name)
        
        # 启动导航
        print("启动导航系统...")
        if not nav_controller.start_navigation(initial_pose=[417, 994, 0]):
            print("导航系统启动失败")
            return
        
        time.sleep(5)
        
        # 创建优化的数据收集器
        collector = OptimizedRealTimeDataCollector(
            camera=camera,
            nav_controller=nav_controller,
            collection_interval=0.2,    # 5Hz采集频率
            max_time_diff=0.05,        # 最大50ms时间差
            blur_threshold=200.0,      # 模糊阈值
            enable_quality_filter=True, # 启用质量过滤
            save_individual_poses=False  # 保存单独的位姿文件
        )
        
        print("\n" + "="*60)
        print("优化的实时建图数据采集")
        print("="*60)
        print("请手动操控机器人移动")
        print("系统将自动采集高质量数据")
        print("按 Ctrl+C 停止采集")
        print("="*60 + "\n")
        
        # 开始数据收集
        collector.start_collection()
        
        # 等待用户停止
        while collector.is_collecting and not collector.stop_event.is_set():
            time.sleep(0.1)
        
        # 停止收集必须在主线程中执行
        collector.stop_collection()
            
    except KeyboardInterrupt:
        print("\n接收到中断信号")
        # 在这里也确保调用stop_collection
        if collector and collector.is_collecting:
            collector.stop_collection()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        # 恢复信号处理器（在主线程中）
        if collector and hasattr(collector, 'original_sigint'):
            try:
                signal.signal(signal.SIGINT, collector.original_sigint)
            except:
                pass  # 忽略信号处理错误
        
        if nav_controller:
            nav_controller.close()
        
        if camera:
            camera.stop()
        
        print("\n程序已安全退出")


if __name__ == "__main__":
    main_realtime_mapping()