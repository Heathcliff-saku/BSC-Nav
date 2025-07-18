import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import os
import time
import math
from typing import List, Dict, Optional, Tuple
from collections import deque
import argparse
from datetime import datetime

class TrajectoryVisualizer:
    """机器人轨迹可视化工具"""
    
    def __init__(self, window_size: int = 500, trail_length: int = 100):
        """
        初始化可视化器
        
        Args:
            window_size: 窗口显示的最大点数
            trail_length: 轨迹尾巴长度
        """
        self.window_size = window_size
        self.trail_length = trail_length
        
        # 数据存储
        self.positions = deque(maxlen=window_size)
        self.orientations = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size)
        
        # 图形设置
        self.fig = None
        self.ax_main = None
        self.ax_info = None
        
    def visualize_session(self, session_dir: str, speed: float = 1.0, 
                         show_orientation: bool = True, save_video: bool = False):
        """
        可视化已保存的会话数据
        
        Args:
            session_dir: 会话数据目录
            speed: 播放速度倍数
            show_orientation: 是否显示朝向
            save_video: 是否保存为视频
        """
        # 加载数据
        data_entries = self._load_session_data(session_dir)
        if not data_entries:
            print("没有找到有效数据")
            return
        
        # 提取轨迹数据
        trajectory_data = self._extract_trajectory(data_entries)
        
        # 设置图形
        self._setup_figure()
        
        # 动画播放
        if save_video:
            self._save_as_video(trajectory_data, show_orientation)
        else:
            self._animate_trajectory(trajectory_data, speed, show_orientation)
    
    def visualize_realtime(self, nav_controller, update_interval: float = 0.1):
        """
        实时可视化机器人轨迹
        
        Args:
            nav_controller: 机器人导航控制器
            update_interval: 更新间隔（秒）
        """
        # 设置图形
        self._setup_figure()
        
        # 实时更新
        self._realtime_update(nav_controller, update_interval)
    
    def visualize_static(self, session_dir: str, save_path: str = None):
        """
        生成静态轨迹图
        
        Args:
            session_dir: 会话数据目录
            save_path: 保存路径
        """
        # 加载数据
        data_entries = self._load_session_data(session_dir)
        if not data_entries:
            print("没有找到有效数据")
            return
        
        # 提取完整轨迹
        positions = []
        orientations = []
        timestamps = []
        
        for entry in data_entries:
            if self._has_valid_pose(entry):
                pose = entry['robot_pose']['pose']
                pos = pose['position']
                positions.append([pos['x'], pos['y']])
                
                # 计算朝向角
                orientation = pose['orientation']
                yaw = self._quaternion_to_yaw(orientation)
                orientations.append(yaw)
                
                timestamps.append(entry['timestamp'])
        
        if not positions:
            print("没有有效的位姿数据")
            return
        
        positions = np.array(positions)
        orientations = np.array(orientations)
        
        # 创建静态图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 主轨迹图
        ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='轨迹')
        ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, 
                   marker='o', label='起点', zorder=5)
        ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, 
                   marker='s', label='终点', zorder=5)
        
        # 添加朝向箭头（每隔一定数量显示）
        arrow_interval = max(1, len(positions) // 20)
        for i in range(0, len(positions), arrow_interval):
            x, y = positions[i]
            dx = 0.3 * np.cos(orientations[i])
            dy = 0.3 * np.sin(orientations[i])
            ax1.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, 
                     fc='gray', ec='gray', alpha=0.6)
        
        ax1.set_xlabel('X (m)', fontsize=12)
        ax1.set_ylabel('Y (m)', fontsize=12)
        ax1.set_title('机器人运动轨迹', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        ax1.legend()
        
        # 时间-距离图
        if len(timestamps) > 1:
            # 计算累积距离
            distances = [0]
            for i in range(1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[i-1])
                distances.append(distances[-1] + dist)
            
            # 时间轴
            start_time = timestamps[0]
            time_axis = [(t - start_time) for t in timestamps]
            
            ax2.plot(time_axis, distances, 'g-', linewidth=2)
            ax2.set_xlabel('时间 (秒)', fontsize=12)
            ax2.set_ylabel('累积距离 (m)', fontsize=12)
            ax2.set_title('运动距离-时间曲线', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            total_distance = distances[-1]
            total_time = time_axis[-1]
            avg_speed = total_distance / total_time if total_time > 0 else 0
            
            info_text = f'总距离: {total_distance:.2f}m\n'
            info_text += f'总时间: {total_time:.1f}s\n'
            info_text += f'平均速度: {avg_speed:.2f}m/s'
            
            ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"轨迹图已保存到: {save_path}")
        
        plt.show()
    
    def _load_session_data(self, session_dir: str) -> List[Dict]:
        """加载会话数据"""
        index_path = os.path.join(session_dir, "data_index.json")
        if not os.path.exists(index_path):
            # 尝试手动数据
            index_path = os.path.join(session_dir, "manual_poses.json")
        
        if not os.path.exists(index_path):
            return []
        
        with open(index_path, 'r') as f:
            return json.load(f)
    
    def _has_valid_pose(self, entry: Dict) -> bool:
        """检查是否有有效的位姿数据"""
        return (entry.get('robot_pose') and 
                entry['robot_pose'].get('pose') and
                entry['robot_pose']['pose'].get('position'))
    
    def _extract_trajectory(self, data_entries: List[Dict]) -> List[Dict]:
        """提取轨迹数据"""
        trajectory = []
        
        for entry in data_entries:
            if self._has_valid_pose(entry):
                pose = entry['robot_pose']['pose']
                pos = pose['position']
                orientation = pose['orientation']
                
                # 计算朝向角
                yaw = self._quaternion_to_yaw(orientation)
                
                # 计算速度（如果有）
                velocity = 0
                if 'velocity' in entry['robot_pose']:
                    velocity = entry['robot_pose']['velocity'].get('linear', 0)
                
                trajectory.append({
                    'timestamp': entry['timestamp'],
                    'x': pos['x'],
                    'y': pos['y'],
                    'yaw': yaw,
                    'velocity': velocity
                })
        
        return trajectory
    
    def _quaternion_to_yaw(self, orientation: Dict) -> float:
        """四元数转航向角"""
        x = orientation.get('x', 0)
        y = orientation.get('y', 0)
        z = orientation.get('z', 0)
        w = orientation.get('w', 1)
        
        # 计算yaw角
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return yaw
    
    def _setup_figure(self):
        """设置图形界面"""
        self.fig = plt.figure(figsize=(12, 8))
        
        # 主轨迹图
        self.ax_main = plt.subplot2grid((3, 3), (0, 0), rowspan=3, colspan=2)
        self.ax_main.set_xlabel('X (m)')
        self.ax_main.set_ylabel('Y (m)')
        self.ax_main.set_title('机器人实时轨迹')
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.axis('equal')
        
        # 信息面板
        self.ax_info = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
        self.ax_info.axis('off')
        
        plt.tight_layout()
    
    def _animate_trajectory(self, trajectory: List[Dict], speed: float, 
                          show_orientation: bool):
        """动画显示轨迹"""
        # 初始化绘图元素
        trail_line, = self.ax_main.plot([], [], 'b-', linewidth=2, alpha=0.6)
        current_pos, = self.ax_main.plot([], [], 'ro', markersize=10)
        orientation_arrow = None
        
        if show_orientation:
            orientation_arrow = self.ax_main.annotate('', xy=(0, 0), xytext=(0, 0),
                                                    arrowprops=dict(arrowstyle='->', 
                                                    color='red', lw=2))
        
        # 信息文本
        info_text = self.ax_info.text(0.1, 0.9, '', transform=self.ax_info.transAxes,
                                     verticalalignment='top', fontsize=10,
                                     family='monospace')
        
        # 数据准备
        xs = [point['x'] for point in trajectory]
        ys = [point['y'] for point in trajectory]
        
        # 设置坐标轴范围
        margin = 1.0
        self.ax_main.set_xlim(min(xs) - margin, max(xs) + margin)
        self.ax_main.set_ylim(min(ys) - margin, max(ys) + margin)
        
        def update(frame):
            # 当前点索引
            idx = int(frame * speed) % len(trajectory)
            
            # 轨迹尾巴
            trail_start = max(0, idx - self.trail_length)
            trail_xs = xs[trail_start:idx+1]
            trail_ys = ys[trail_start:idx+1]
            trail_line.set_data(trail_xs, trail_ys)
            
            # 当前位置
            current_x = trajectory[idx]['x']
            current_y = trajectory[idx]['y']
            current_pos.set_data([current_x], [current_y])
            
            # 朝向箭头
            if show_orientation and orientation_arrow:
                yaw = trajectory[idx]['yaw']
                arrow_len = 0.5
                dx = arrow_len * np.cos(yaw)
                dy = arrow_len * np.sin(yaw)
                orientation_arrow.set_position((current_x + dx, current_y + dy))
                orientation_arrow.xy = (current_x, current_y)
            
            # 更新信息
            info = f"帧: {idx}/{len(trajectory)}\n"
            info += f"位置: ({current_x:.2f}, {current_y:.2f})\n"
            info += f"朝向: {math.degrees(trajectory[idx]['yaw']):.1f}°\n"
            info += f"速度: {trajectory[idx]['velocity']:.2f} m/s\n"
            
            if idx > 0:
                # 计算移动距离
                dx = current_x - trajectory[idx-1]['x']
                dy = current_y - trajectory[idx-1]['y']
                dist = np.sqrt(dx**2 + dy**2)
                dt = trajectory[idx]['timestamp'] - trajectory[idx-1]['timestamp']
                if dt > 0:
                    calc_speed = dist / dt
                    info += f"计算速度: {calc_speed:.2f} m/s\n"
            
            info_text.set_text(info)
            
            return trail_line, current_pos, info_text
        
        # 创建动画
        interval = 50  # 毫秒
        frames = int(len(trajectory) / speed)
        
        anim = FuncAnimation(self.fig, update, frames=frames, 
                           interval=interval, blit=True, repeat=True)
        
        plt.show()
    
    def _realtime_update(self, nav_controller, update_interval: float):
        """实时更新轨迹"""
        # 初始化绘图元素
        trail_line, = self.ax_main.plot([], [], 'b-', linewidth=2, alpha=0.6)
        current_pos, = self.ax_main.plot([], [], 'ro', markersize=10)
        orientation_arrow = self.ax_main.annotate('', xy=(0, 0), xytext=(0, 0),
                                                arrowprops=dict(arrowstyle='->', 
                                                color='red', lw=2))
        
        # 信息文本
        info_text = self.ax_info.text(0.1, 0.9, '', transform=self.ax_info.transAxes,
                                     verticalalignment='top', fontsize=10,
                                     family='monospace')
        
        # 动态坐标轴范围
        x_range = [-5, 5]
        y_range = [-5, 5]
        
        def update(frame):
            try:
                # 获取当前状态
                state = nav_controller.get_state()
                if not state:
                    return trail_line, current_pos, info_text
                
                pos = state['pose']['position']
                orientation = state['pose']['orientation']
                velocity = state['velocity']
                
                # 添加到历史
                self.positions.append([pos['x'], pos['y']])
                self.orientations.append(self._quaternion_to_yaw(orientation))
                self.velocities.append(velocity['linear'])
                self.timestamps.append(time.time())
                
                if len(self.positions) > 0:
                    # 更新轨迹
                    positions = np.array(list(self.positions))
                    trail_line.set_data(positions[:, 0], positions[:, 1])
                    
                    # 当前位置
                    current_pos.set_data([pos['x']], [pos['y']])
                    
                    # 朝向箭头
                    yaw = self.orientations[-1]
                    arrow_len = 0.5
                    dx = arrow_len * np.cos(yaw)
                    dy = arrow_len * np.sin(yaw)
                    orientation_arrow.set_position((pos['x'] + dx, pos['y'] + dy))
                    orientation_arrow.xy = (pos['x'], pos['y'])
                    
                    # 动态调整坐标轴
                    margin = 2.0
                    x_min, x_max = positions[:, 0].min() - margin, positions[:, 0].max() + margin
                    y_min, y_max = positions[:, 1].min() - margin, positions[:, 1].max() + margin
                    
                    # 保持纵横比
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    half_range = max(x_max - x_min, y_max - y_min) / 2
                    
                    self.ax_main.set_xlim(x_center - half_range, x_center + half_range)
                    self.ax_main.set_ylim(y_center - half_range, y_center + half_range)
                    
                    # 更新信息
                    info = f"实时状态\n"
                    info += f"{'='*20}\n"
                    info += f"位置: ({pos['x']:.2f}, {pos['y']:.2f})\n"
                    info += f"朝向: {math.degrees(yaw):.1f}°\n"
                    info += f"线速度: {velocity['linear']:.2f} m/s\n"
                    info += f"角速度: {velocity['angular']:.2f} rad/s\n"
                    info += f"\n轨迹统计\n"
                    info += f"{'='*20}\n"
                    info += f"数据点数: {len(self.positions)}\n"
                    
                    if len(self.positions) > 1:
                        # 计算总距离
                        total_dist = 0
                        for i in range(1, len(positions)):
                            dist = np.linalg.norm(positions[i] - positions[i-1])
                            total_dist += dist
                        info += f"总距离: {total_dist:.2f} m\n"
                        
                        # 平均速度
                        avg_speed = np.mean(list(self.velocities))
                        info += f"平均速度: {avg_speed:.2f} m/s\n"
                    
                    info_text.set_text(info)
                
            except Exception as e:
                print(f"更新错误: {e}")
            
            return trail_line, current_pos, info_text
        
        # 创建动画
        anim = FuncAnimation(self.fig, update, interval=int(update_interval * 1000),
                           blit=True, cache_frame_data=False)
        
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='机器人轨迹可视化工具')
    parser.add_argument('mode', choices=['static', 'replay', 'realtime'],
                       help='可视化模式: static(静态图), replay(回放), realtime(实时)')
    parser.add_argument('--session', type=str, help='会话数据目录(static和replay模式需要)')
    parser.add_argument('--speed', type=float, default=1.0, help='回放速度倍数')
    parser.add_argument('--save', type=str, help='保存路径')
    parser.add_argument('--no-orientation', action='store_true', help='不显示朝向')
    parser.add_argument('--trail-length', type=int, default=100, help='轨迹尾巴长度')
    
    args = parser.parse_args()
    
    visualizer = TrajectoryVisualizer(trail_length=args.trail_length)
    
    if args.mode == 'static':
        if not args.session:
            print("静态模式需要指定 --session 参数")
            return
        
        save_path = args.save or f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        visualizer.visualize_static(args.session, save_path)
        
    elif args.mode == 'replay':
        if not args.session:
            print("回放模式需要指定 --session 参数")
            return
        
        visualizer.visualize_session(args.session, args.speed, 
                                   not args.no_orientation, bool(args.save))
        
    elif args.mode == 'realtime':
        # 实时模式需要导入导航控制器
        try:
            from lowlevel import RobotNavigationController
            
            # 配置参数
            ws_url = "ws://192.168.1.102:9090"
            http_url = "http://192.168.1.102/apiUrl"
            map_name = "dongsheng03"
            
            print("连接机器人...")
            with RobotNavigationController(ws_url, http_url, map_name) as nav_controller:
                print("启动导航系统...")
                if nav_controller.start_navigation():
                    print("开始实时轨迹可视化...")
                    visualizer.visualize_realtime(nav_controller, 0.1)
                else:
                    print("导航系统启动失败")
                    
        except ImportError:
            print("实时模式需要 lowlevel.py 模块")
        except Exception as e:
            print(f"实时模式错误: {e}")

if __name__ == "__main__":
    main()