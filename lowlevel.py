#!/usr/bin/env python
import asyncio
from websocket import create_connection
import websockets
import time
import requests
import json
import numpy as np
import math
from threading import Thread, Timer, Lock
import base64
import array
import cv2
from queue import Queue
from typing import Tuple, Dict, Optional, List
import logging

# 导入原有的类
from client import WSClient, HttpClient, quaternion_from_euler, png_coordinate_to_map, map_coordinate_to_png

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class RobotNavigationController:
    """
    高层机器人导航控制类
    封装WSClient和HttpClient，提供简洁的导航控制接口
    """
    
    def __init__(self, ws_url: str, http_url: str, map_name: str = None):
        """
        初始化导航控制器
        
        Args:
            ws_url: WebSocket服务器地址
            http_url: HTTP API服务器地址
            map_name: 默认使用的地图名称
        """
        self.logger = logging.getLogger(__name__)
        
        # 初始化客户端
        self.ws_client = None
        self.http_client = HttpClient(http_url)
        self.ws_url = ws_url
        
        # 导航状态
        self.is_navigation_active = False
        self.current_map_name = map_name
        self.map_info = None
        
        # 机器人状态
        self.robot_state = {
            'position': {'x': 0, 'y': 0, 'z': 0},
            'orientation': {'x': 0, 'y': 0, 'z': 0, 'w': 1},  # 四元数
            'velocity': {'linear': 0, 'angular': 0},
            'navigation_status': 'idle',  # idle, navigating, reached, failed
            'task_status': None,
            'slam_status': None,
            'battery_level': None,
            'last_update': None
        }
        
        # 线程安全
        self.state_lock = Lock()
        self.message_queue = Queue()
        self.receive_thread = None
        self.is_running = False
        
        # 初始化
        self._initialize()
    
    def _initialize(self):
        """初始化连接和登录"""
        try:
            # HTTP客户端登录
            self.logger.info("正在登录HTTP服务...")
            self.http_client.login_()
            
            # 创建WebSocket连接
            self.logger.info("正在连接WebSocket服务...")
            self.ws_client = WSClient(self.ws_url, f"nav_controller_{int(time.time())}")
            
            if not self.ws_client.isconnect:
                raise ConnectionError("WebSocket连接失败")
            
            # 启动心跳
            self.ws_client.start_heartbeat_timer(2)
            
            # 启动接收线程
            self._start_receive_thread()
            
            # 订阅必要的话题
            self._subscribe_topics()
            
            self.logger.info("导航控制器初始化成功")
            
        except Exception as e:
            self.logger.error(f"初始化失败: {e}")
            raise
    
    def _subscribe_topics(self):
        """订阅必要的ROS话题"""
        # 订阅任务状态
        self.ws_client.sub_task_status()
        # 订阅机器人状态
        self.ws_client.sub_robot_status()
        # 订阅SLAM状态
        self.ws_client.sub_slam_status()
        
        # 订阅里程计信息（获取位置）
        msg = {   
            "op": "subscribe",
            "topic": "/odom",
            "type": "nav_msgs/Odometry"
        }
        self.ws_client.send_msg(msg)
        
        self.logger.info("已订阅所有必要话题")
    
    def _start_receive_thread(self):
        """启动消息接收线程"""
        self.is_running = True
        self.receive_thread = Thread(target=self._receive_loop, daemon=True)
        self.receive_thread.start()
    
    def _receive_loop(self):
        """消息接收循环"""
        while self.is_running:
            try:
                if self.ws_client and self.ws_client.ws:
                    message = self.ws_client.ws.recv()
                    if message:
                        self._process_message(json.loads(message))
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"接收消息错误: {e}")
                time.sleep(0.1)
    
    def _process_message(self, message):
        """处理接收到的消息"""
        try:
            topic = message.get('topic', '')
            
            with self.state_lock:
                # 更新任务状态
                if topic == '/run_management/task_status':
                    status = message.get('msg', {})
                    self.robot_state['navigation_status'] = self._parse_nav_status(status)
                    self.robot_state['task_status'] = status
                
                # 更新机器人状态
                elif topic == '/dash_board/robot_status':
                    status = message.get('msg', {})
                    self._update_robot_status(status)
                
                # 更新SLAM状态
                elif topic == '/slam_status':
                    self.robot_state['slam_status'] = message.get('msg', {})
                
                # 更新位置信息（如果有odom话题）
                elif topic == '/odom':
                    odom = message.get('msg', {})
                    self._update_position_from_odom(odom)
                
                self.robot_state['last_update'] = time.time()
                
        except Exception as e:
            self.logger.error(f"处理消息错误: {e}")
    
    def _parse_nav_status(self, status):
        """解析导航状态"""
        # 根据实际的状态消息格式解析
        # 这里是一个示例实现
        if not status:
            return 'idle'
        
        task_state = status.get('state', '')
        if 'SUCCEEDED' in task_state:
            return 'reached'
        elif 'ACTIVE' in task_state or 'PENDING' in task_state:
            return 'navigating'
        elif 'ABORTED' in task_state or 'FAILED' in task_state:
            return 'failed'
        else:
            return 'idle'
    
    def _update_robot_status(self, status):
        """更新机器人状态信息"""
        # 根据实际的状态消息格式更新
        if 'battery' in status:
            self.robot_state['battery_level'] = status['battery']
        if 'velocity' in status:
            self.robot_state['velocity'] = status['velocity']
    
    def _update_position_from_odom(self, odom):
        """从里程计更新位置和速度信息"""
        # 更新位姿信息
        pose_data = odom.get('pose', {})
        if pose_data:
            pose = pose_data.get('pose', {})
            if pose:
                # 更新位置
                position = pose.get('position', {})
                self.robot_state['position']['x'] = position.get('x', 0)
                self.robot_state['position']['y'] = position.get('y', 0)
                self.robot_state['position']['z'] = position.get('z', 0)
                
                # 更新方向（四元数）
                orientation = pose.get('orientation', {})
                if orientation:
                    self.robot_state['orientation']['x'] = orientation.get('x', 0)
                    self.robot_state['orientation']['y'] = orientation.get('y', 0)
                    self.robot_state['orientation']['z'] = orientation.get('z', 0)
                    self.robot_state['orientation']['w'] = orientation.get('w', 1)
        
        # 更新速度信息
        twist_data = odom.get('twist', {})
        if twist_data:
            twist = twist_data.get('twist', {})
            if twist:
                linear = twist.get('linear', {})
                angular = twist.get('angular', {})
                
                # 计算线速度大小
                vx = linear.get('x', 0)
                vy = linear.get('y', 0)
                self.robot_state['velocity']['linear'] = math.sqrt(vx**2 + vy**2)
                
                # 角速度（通常是z轴）
                self.robot_state['velocity']['angular'] = angular.get('z', 0)
    
    def start_navigation(self, map_name: str = None, initial_pose: Tuple[float, float, float] = None) -> bool:
        """
        启动导航状态
        
        Args:
            map_name: 地图名称，如果为None则使用初始化时的地图
            initial_pose: 初始位置 (x, y, theta)，如果为None则使用当前位置
            
        Returns:
            bool: 是否成功启动
        """
        try:
            # 确定使用的地图
            if map_name:
                self.current_map_name = map_name
            
            if not self.current_map_name:
                raise ValueError("未指定地图名称")
            
            # 获取地图信息
            self.logger.info(f"获取地图信息: {self.current_map_name}")
            self.map_info = self.http_client.get_map_info(self.current_map_name)
            
            if not self.map_info:
                raise ValueError(f"无法获取地图信息: {self.current_map_name}")
            
            # 启动导航
            self.logger.info(f"启动导航: {self.current_map_name}")
            self.ws_client.follow_line(idtype="start", filename=self.current_map_name)
            
            # 等待导航启动
            time.sleep(2)
            
            # 设置初始位置
            if initial_pose:
                self.logger.info(f"设置初始位置: {initial_pose}")
                pos_x_i, pos_y_i = png_coordinate_to_map(initial_pose[:2], self.map_info)
                self.ws_client.initial_pos([pos_x_i, pos_y_i], initial_pose[2])
                time.sleep(1)
            else:
                # 如果没有指定初始位置，尝试使用当前位置初始化
                self.logger.info("使用当前位置作为初始位置")
                # 等待一下让odom数据到达
                time.sleep(2)
                current_pos = self.get_position_with_angle()
                if current_pos[0] != 0 or current_pos[1] != 0:
                    self.ws_client.initial_pos([current_pos[0], current_pos[1]], current_pos[3])
            
            self.is_navigation_active = True
            self.logger.info("导航启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"启动导航失败: {e}")
            return False
    
    def move_robot(self, linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0):
        """
        控制机器人移动（使用旧API，保留兼容性）
        
        Args:
            linear_x: 前进/后退速度 (+前进, -后退) 范围: +/-1.5 m/s
            linear_y: 横向移动速度 (仅适用于某些车型，通常为0)
            angular_z: 旋转速度 (+左转, -右转) 范围: +/-1.5 rad/s
        """
        if not self.is_navigation_active:
            self.logger.warning("导航未启动，将使用全局控制话题")
            use_nav_topic = False
        else:
            use_nav_topic = True
        
        topic = "/cmd_vel" if use_nav_topic else "/run_management/virtual_joy"
        
        msg = {
            "op": "publish",
            "topic": topic,
            "type": "geometry_msgs/Twist",
            "msg": {
                "linear": {
                    "x": linear_x,
                    "y": linear_y,
                    "z": 0
                },
                "angular": {
                    "x": 0,
                    "y": 0,
                    "z": angular_z
                }
            }
        }
        
        self.ws_client.publish_data(msg)
        self.logger.debug(f"发送移动命令: linear_x={linear_x:.2f}, angular_z={angular_z:.2f}")
    
    def chassis_control_move(self, distance: float, linear_x: float = 0.2) -> bool:
        """
        使用新的底盘控制API进行直线移动（同步调用）
        
        Args:
            distance: 移动距离（米）
            linear_x: 移动速度（米/秒）
            
        Returns:
            bool: 是否成功（运动完成后返回）
        """
        try:
            response = self.ws_client.chassis_control(
                distance=distance,
                rotation_angle=0.0,
                motion_type=0,  # 直线运动模式
                linear_x=linear_x,
                angular_z=0.0,
                stop=False
            )
            
            if response and response.get('result'):
                success = response.get('values', {}).get('success', False)
                if success:
                    self.logger.debug(f"直线移动完成: 距离={distance}m, 速度={linear_x}m/s")
                else:
                    self.logger.warning(f"直线移动失败: {response.get('values', {}).get('message', 'Unknown error')}")
                return success
            return False
        except Exception as e:
            self.logger.error(f"底盘控制移动失败: {e}")
            return False
    
    def chassis_control_rotate(self, rotation_angle: float, angular_z: float = 0.3) -> bool:
        """
        使用新的底盘控制API进行旋转（同步调用）
        
        Args:
            rotation_angle: 旋转角度（度）
            angular_z: 旋转速度（弧度/秒）
            
        Returns:
            bool: 是否成功（运动完成后返回）
        """
        try:
            response = self.ws_client.chassis_control(
                distance=0.0,
                rotation_angle=rotation_angle,
                motion_type=1,  # 旋转运动模式
                linear_x=0.0,
                angular_z=angular_z,
                stop=False
            )
            
            if response and response.get('result'):
                success = response.get('values', {}).get('success', False)
                if success:
                    self.logger.debug(f"旋转完成: 角度={rotation_angle}°, 速度={angular_z}rad/s")
                else:
                    self.logger.warning(f"旋转失败: {response.get('values', {}).get('message', 'Unknown error')}")
                return success
            return False
        except Exception as e:
            self.logger.error(f"底盘控制旋转失败: {e}")
            return False
    
    def chassis_control_stop(self) -> bool:
        """
        使用新的底盘控制API停止机器人
        
        Returns:
            bool: 是否成功
        """
        try:
            response = self.ws_client.chassis_control(
                distance=0.0,
                rotation_angle=0.0,
                motion_type=0,
                linear_x=0.0,
                angular_z=0.0,
                stop=True
            )
            
            if response and response.get('result'):
                success = response.get('values', {}).get('success', False)
                if success:
                    self.logger.debug("停止命令成功")
                else:
                    self.logger.warning(f"停止命令失败: {response.get('values', {}).get('message', 'Unknown error')}")
                return success
            return False
        except Exception as e:
            self.logger.error(f"底盘控制停止失败: {e}")
            return False
    
    def stop_robot(self):
        """停止机器人移动（优先使用新API）"""
        # 尝试使用新API停止
        if not self.chassis_control_stop():
            # 如果新API失败，使用旧API
            self.move_robot(0.0, 0.0, 0.0)
            self.logger.debug("使用旧API发送停止命令")
    
    def get_state(self) -> Dict:
        """
        获取机器人当前导航定位状态
        
        Returns:
            dict: 包含xyz坐标和旋转四元数的字典
        """
        with self.state_lock:
            # 返回导航定位状态的深拷贝
            return {
                'pose': {
                    'position': {
                        'x': self.robot_state['position']['x'],
                        'y': self.robot_state['position']['y'],
                        'z': self.robot_state['position'].get('z', 0)  # 默认z=0
                    },
                    'orientation': {
                        'x': self.robot_state['orientation'].get('x', 0),
                        'y': self.robot_state['orientation'].get('y', 0),
                        'z': self.robot_state['orientation'].get('z', 0),
                        'w': self.robot_state['orientation'].get('w', 1)
                    }
                },
                'velocity': {
                    'linear': self.robot_state['velocity']['linear'],  # 线速度 m/s
                    'angular': self.robot_state['velocity']['angular']  # 角速度 rad/s
                },
                'navigation_status': self.robot_state['navigation_status'],
                'is_navigation_active': self.is_navigation_active,
                'current_map': self.current_map_name,
                'last_update': self.robot_state['last_update']
            }
    
    def get_position_with_angle(self) -> Tuple[float, float, float, float]:
        """
        获取机器人位置和角度（方便导航使用）
        
        Returns:
            tuple: (x, y, z, theta) 其中theta为角度（度）
        """
        state = self.get_state()
        pose = state['pose']
        
        # 四元数转欧拉角
        orientation = pose['orientation']
        # 使用类似quaternion_to_euler的转换
        roll = math.atan2(2 * (orientation['w'] * orientation['x'] + orientation['y'] * orientation['z']), 
                          1 - 2 * (orientation['x'] * orientation['x'] + orientation['y'] * orientation['y']))
        pitch = math.asin(2 * (orientation['w'] * orientation['y'] - orientation['x'] * orientation['z']))
        yaw = math.atan2(2 * (orientation['w'] * orientation['z'] + orientation['x'] * orientation['y']), 
                         1 - 2 * (orientation['z'] * orientation['z'] + orientation['y'] * orientation['y']))
        
        # 转换为角度
        theta = math.degrees(yaw)
        
        return (pose['position']['x'], 
                pose['position']['y'], 
                pose['position']['z'], 
                theta)
    
    def nav_to_point(self, target: Tuple[float, float, float], wait_for_result: bool = False, 
                    timeout: float = 300, position_threshold: float = 0.5, 
                    stable_duration: float = 10.0) -> bool:
        """
        导航到指定点
        
        Args:
            target: 目标点 (x, y, theta)，其中theta为角度，xy为地图坐标
            wait_for_result: 是否等待导航结果
            timeout: 最大等待超时时间（秒），默认300秒
            position_threshold: 位置阈值（米），在此范围内认为到达目标，默认0.5米
            stable_duration: 稳定时间（秒），在目标附近保持此时间认为到达，默认10秒
            
        Returns:
            bool: 导航命令是否发送成功（如果wait_for_result=True，则返回是否到达目标）
        """
        if not self.is_navigation_active:
            self.logger.error("导航未启动，请先调用start_navigation()")
            return False
        
        try:
            # 直接使用传入的坐标，不进行转换（假设传入的是地图坐标）
            self.logger.info(f"导航到目标点: x={target[0]:.2f}, y={target[1]:.2f}, theta={target[2]:.2f}")
            # 发送导航命令，注意这里应该传递正确的坐标列表
            self.http_client.run_realtime_task([target[0], target[1], target[2]])
            
            if not wait_for_result:
                return True
            
            # 基于位置的导航结果判定
            start_time = time.time()
            stable_start_time = None
            last_position = None
            
            while time.time() - start_time < timeout:
                # 获取当前状态
                state = self.get_state()
                current_pos = state['pose']['position']
                
                # 计算与目标的距离
                distance = math.sqrt((current_pos['x'] - target[0])**2 + 
                                   (current_pos['y'] - target[1])**2)
                
                # 计算速度（用于判断是否停止）
                current_velocity = state['velocity']['linear']
                
                # 记录位置用于判断是否在移动
                is_moving = False
                if last_position is not None:
                    pos_change = math.sqrt((current_pos['x'] - last_position['x'])**2 + 
                                         (current_pos['y'] - last_position['y'])**2)
                    if pos_change > 0.05:  # 5cm的移动认为是在移动
                        is_moving = True
                last_position = current_pos.copy()
                
                # 判断是否在目标附近
                if distance <= position_threshold:
                    # 检查是否刚进入目标范围
                    if stable_start_time is None:
                        stable_start_time = time.time()
                        self.logger.info(f"进入目标范围 (距离: {distance:.2f}m)，开始稳定性检测...")
                    
                    # 检查是否保持稳定
                    stable_time = time.time() - stable_start_time
                    
                    # 如果速度很小且位置变化很小，认为是稳定的
                    if current_velocity < 0.1 and not is_moving:
                        if stable_time >= stable_duration:
                            self.logger.info(f"成功到达目标点！距离: {distance:.2f}m，稳定时间: {stable_time:.1f}s")
                            return True
                        else:
                            remaining_time = stable_duration - stable_time
                            if int(remaining_time) % 2 == 0:  # 每2秒打印一次
                                self.logger.info(f"在目标附近等待稳定... 剩余时间: {remaining_time:.1f}s")
                    else:
                        # 如果在移动，重置稳定计时
                        if is_moving:
                            stable_start_time = time.time()
                            self.logger.debug("检测到移动，重置稳定计时器")
                else:
                    # 离开目标范围，重置稳定计时
                    if stable_start_time is not None:
                        self.logger.info(f"离开目标范围 (距离: {distance:.2f}m)")
                        stable_start_time = None
                    
                    # 定期报告进度
                    elapsed = time.time() - start_time
                    if int(elapsed) % 10 == 0:  # 每10秒报告一次
                        self.logger.info(f"导航中... 距离目标: {distance:.2f}m, 速度: {current_velocity:.2f}m/s")
                
                # 检查导航状态（作为辅助判断）
                nav_status = state.get('navigation_status', 'unknown')
                if nav_status == 'failed':
                    self.logger.warning("导航状态显示失败，但继续基于位置判断...")
                
                time.sleep(0.5)  # 检查频率
            
            # 超时
            self.logger.error(f"导航超时 ({timeout}秒)")
            return False
            
        except Exception as e:
            self.logger.error(f"导航命令发送失败: {e}")
            return False
    
    def nav_to_png_coordinate(self, png_pos: Tuple[int, int], theta: float = 0, 
                             wait_for_result: bool = False, timeout: float = 300,
                             position_threshold: float = 0.5, stable_duration: float = 10.0) -> bool:
        """
        导航到PNG图片坐标对应的位置
        
        Args:
            png_pos: PNG图片上的坐标 (x, y)
            theta: 目标朝向角度
            wait_for_result: 是否等待导航结果
            timeout: 最大等待超时时间（秒）
            position_threshold: 位置阈值（米），在此范围内认为到达目标
            stable_duration: 稳定时间（秒），在目标附近保持此时间认为到达
            
        Returns:
            bool: 导航是否成功
        """
        if not self.map_info:
            self.logger.error("未加载地图信息")
            return False
        
        # 转换坐标
        map_x, map_y = png_coordinate_to_map(png_pos, self.map_info)
        return self.nav_to_point((map_x, map_y, theta), wait_for_result, timeout, 
                                position_threshold, stable_duration)
    
    def cancel_navigation(self):
        """取消当前导航任务"""
        self.logger.info("取消当前导航任务")
        self.ws_client.cancel_nav()
    
    def stop_navigation(self) -> bool:
        """
        停止导航并关闭导航状态
        
        Returns:
            bool: 是否成功停止
        """
        try:
            if self.is_navigation_active:
                # 取消当前任务
                self.cancel_navigation()
                time.sleep(1)
                
                # 关闭导航
                self.logger.info("关闭导航")
                self.ws_client.follow_line(idtype="stop", filename=self.current_map_name)
                
                self.is_navigation_active = False
                self.logger.info("导航已停止")
            
            return True
            
        except Exception as e:
            self.logger.error(f"停止导航失败: {e}")
            return False
    
    def close(self):
        """关闭所有连接和线程"""
        self.logger.info("关闭导航控制器...")
        
        # 停止导航
        self.stop_navigation()
        
        # 停止接收线程
        self.is_running = False
        if self.receive_thread:
            self.receive_thread.join(timeout=2)
        
        # 停止心跳
        if self.ws_client:
            self.ws_client.stop_heartbeat_timer()
            self.ws_client.on_close()
        
        self.logger.info("导航控制器已关闭")
    
    def __enter__(self):
        """支持with语句"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动清理"""
        self.close()


# 使用示例
if __name__ == '__main__':
    # 配置参数
    ws_url = "ws://192.168.1.102:9090"
    http_url = "http://192.168.1.102/apiUrl"
    map_name = "dongsheng03"
    
    # 使用with语句自动管理资源
    with RobotNavigationController(ws_url, http_url, map_name) as nav_controller:
        # 1. 启动导航，可选设置初始位置（PNG坐标）
        if nav_controller.start_navigation(initial_pose=(383, 120, 0)):
            
            # 2. 获取机器人状态（包含四元数）
            state = nav_controller.get_state()
            print(f"当前状态: ")
            print(f"  位置: x={state['pose']['position']['x']:.2f}, "
                  f"y={state['pose']['position']['y']:.2f}, "
                  f"z={state['pose']['position']['z']:.2f}")
            print(f"  方向(四元数): x={state['pose']['orientation']['x']:.3f}, "
                  f"y={state['pose']['orientation']['y']:.3f}, "
                  f"z={state['pose']['orientation']['z']:.3f}, "
                  f"w={state['pose']['orientation']['w']:.3f}")
            print(f"  速度: 线速度={state['velocity']['linear']:.2f} m/s, "
                  f"角速度={state['velocity']['angular']:.2f} rad/s")
            
            # 获取位置和角度（方便导航使用）
            x, y, z, theta = nav_controller.get_position_with_angle()
            print(f"  角度: {theta:.2f}°")
            
            # 3. 测试新的底盘控制API（同步调用）
            print("\n测试新的底盘控制API（同步模式）...")
            
            # 前进0.5米（运动完成后才返回）
            print("前进0.5米...")
            if nav_controller.chassis_control_move(distance=0.5, linear_x=0.2):
                print("前进完成")
            
            # 左转30度（运动完成后才返回）
            print("左转30度...")
            if nav_controller.chassis_control_rotate(rotation_angle=30, angular_z=0.3):
                print("左转完成")
            
            # 右转30度（运动完成后才返回）
            print("右转30度...")
            if nav_controller.chassis_control_rotate(rotation_angle=-30, angular_z=0.3):
                print("右转完成")
            
            # 停止
            print("停止机器人...")
            if nav_controller.chassis_control_stop():
                print("停止成功")
            
            # 获取最新状态
            final_state = nav_controller.get_state()
            print(f"\n最终位置: x={final_state['pose']['position']['x']:.2f}, "
                  f"y={final_state['pose']['position']['y']:.2f}")
            
            # 等待一段时间
            time.sleep(5)
        
        # 退出时自动停止导航并清理资源