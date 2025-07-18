import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
from queue import Queue, Empty
import logging
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class OptimizedRealSenseCamera:
    def __init__(self, device_index=0, warmup_frames=60, enable_filters=True, 
                 depth_preset='high_accuracy', enable_motion_blur_detection=True):
        """
        优化的RealSense相机类
        
        Args:
            device_index: 设备索引
            warmup_frames: 预热帧数
            enable_filters: 是否启用深度滤波器
            depth_preset: 深度预设 ('high_accuracy', 'high_density', 'medium_density', 'hand')
            enable_motion_blur_detection: 是否启用运动模糊检测
        """
        self.logger = logging.getLogger(__name__)
        
        # 配置管道
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 线程控制
        self.is_running = False
        self.capture_thread = None
        self.frame_queue = Queue(maxsize=5)  # 增大队列以减少丢帧
        
        # 运动模糊检测
        self.enable_motion_blur_detection = enable_motion_blur_detection
        self.blur_threshold = 100.0  # Laplacian方差阈值
        
        # 获取所有连接的设备
        context = rs.context()
        devices = context.query_devices()
        
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices detected.")
        
        # 选择设备
        if device_index < len(devices):
            self.device = devices[device_index]
        else:
            raise ValueError("Device index out of range.")
        
        self.logger.info(f"Using device: {self.device.get_info(rs.camera_info.serial_number)}")
        
        # 配置RGB和深度流
        self.config.enable_device(self.device.get_info(rs.camera_info.serial_number))
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        # 启动管道
        self.profile = self.pipeline.start(self.config)
        
        # 配置深度传感器预设
        self._configure_depth_sensor(depth_preset)
        
        # 配置彩色相机（减少运动模糊）
        # self._configure_color_sensor()
        
        # 获取深度和RGB流的内参
        self.depth_stream = self.profile.get_stream(rs.stream.depth)
        self.color_stream = self.profile.get_stream(rs.stream.color)
        
        # 获取相机内参
        self.depth_intrinsics = self.depth_stream.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()
        
        # 创建对齐对象
        self.align_to_color = rs.align(rs.stream.color)
        
        # 初始化深度滤波器（可选）
        self.enable_filters = enable_filters
        if self.enable_filters:
            self._init_basic_filters()
        
        # 预热相机
        self.logger.info(f"Warming up camera with {warmup_frames} frames...")
        for i in range(warmup_frames):
            try:
                self.pipeline.wait_for_frames()
                time.sleep(0.033)
            except Exception as e:
                self.logger.warning(f"Warmup frame {i} failed: {e}")
                continue
        
        self.logger.info("Camera warmup complete!")
        
        # 启动持续采集线程
        self.start_capture_thread()
    
    def _configure_depth_sensor(self, preset_name):
        """配置深度传感器预设"""
        depth_sensor = self.profile.get_device().first_depth_sensor()
        
        # 深度预设映射
        preset_map = {
            'high_accuracy': rs.rs400_visual_preset.high_accuracy,
            'high_density': rs.rs400_visual_preset.high_density,
            'medium_density': rs.rs400_visual_preset.medium_density,
            'hand': rs.rs400_visual_preset.hand,
            'default': rs.rs400_visual_preset.default
        }
        
        if preset_name in preset_map:
            preset = preset_map[preset_name]
            depth_sensor.set_option(rs.option.visual_preset, preset)
            self.logger.info(f"深度预设设置为: {preset_name}")
        
        # # 额外的深度传感器优化
        # if depth_sensor.supports(rs.option.enable_auto_exposure):
        #     depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        
        # # 设置深度单位为毫米
        # if depth_sensor.supports(rs.option.depth_units):
        #     depth_sensor.set_option(rs.option.depth_units, 0.001)
    
    def _configure_color_sensor(self):
        """配置彩色传感器以减少运动模糊"""
        color_sensor = self.profile.get_device().query_sensors()[1]  # 通常彩色传感器是第二个
        
        # 减少曝光时间以减少运动模糊
        if color_sensor.supports(rs.option.exposure):
            # 设置较短的曝光时间（微秒）
            color_sensor.set_option(rs.option.exposure, 5000)  # 5ms
            
        # 关闭自动曝光
        if color_sensor.supports(rs.option.enable_auto_exposure):
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            
        # 提高增益补偿曝光降低
        if color_sensor.supports(rs.option.gain):
            color_sensor.set_option(rs.option.gain, 64)
        
        self.logger.info("彩色相机配置完成（优化运动模糊）")
    
    def _init_optimized_filters(self):
        """初始化优化的深度滤波器"""
        # 1. 抽取滤波器（降低分辨率以提高性能）
        self.decimation_filter = rs.decimation_filter()
        self.decimation_filter.set_option(rs.option.filter_magnitude, 2)  # 2x下采样
        
        # 2. 空间滤波器（优化参数）
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 5)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        self.spatial_filter.set_option(rs.option.holes_fill, 3)
        
        # 3. 时间滤波器（减少参数以提高响应速度）
        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.3)  # 降低时间平滑
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20)
        
        # 4. 孔洞填充
        self.hole_filling_filter = rs.hole_filling_filter()
        
        # 5. 视差变换
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        
        # 6. 阈值滤波器
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, 0.2)
        self.threshold_filter.set_option(rs.option.max_distance, 10.0)
    
    def _init_basic_filters(self):
        """初始化基础深度滤波器（兼容原版本）"""
        # 2. 空间滤波器（边缘保持平滑）
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 5)  # 滤波器强度
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)  # 平滑度
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20)  # 边缘阈值
        self.spatial_filter.set_option(rs.option.holes_fill, 3)  # 孔洞填充（所有孔洞）
        
        # 3. 时间滤波器（减少时序噪声）
        self.temporal_filter = rs.temporal_filter()
        
        # 4. 孔洞填充滤波器
        self.hole_filling_filter = rs.hole_filling_filter()
        
        # 5. 视差变换（用于某些滤波器的预处理）
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        
        # 6. 阈值滤波器（移除超出范围的值）
        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, 0.2)  # 最小距离（米）
        self.threshold_filter.set_option(rs.option.max_distance, 10.0)  # 最大距离（米）
    
    def detect_motion_blur(self, image):
        """
        检测图像是否存在运动模糊
        
        Args:
            image: BGR图像
            
        Returns:
            tuple: (is_blurry, blur_score)
        """
        if not self.enable_motion_blur_detection:
            return False, 0.0
            
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算Laplacian方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = laplacian.var()
        
        # 判断是否模糊
        is_blurry = blur_score < self.blur_threshold
        
        return is_blurry, blur_score
    
    def process_depth_frame_optimized(self, depth_frame):
        """
        优化的深度帧处理（提高性能）
        """
        if not self.enable_filters:
            return depth_frame
        
        try:
            # 简化的滤波流程
            depth_frame = self.threshold_filter.process(depth_frame)
            depth_frame = self.decimation_filter.process(depth_frame)
            depth_frame = self.depth_to_disparity.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.disparity_to_depth.process(depth_frame)
            return depth_frame
        except Exception as e:
            self.logger.warning(f"Depth filtering failed: {e}")
            return depth_frame
    
    def process_depth_frame_simple(self, depth_frame):
        """
        简化的深度帧处理（兼容原版本）
        
        Args:
            depth_frame: rs.depth_frame对象
        
        Returns:
            处理后的深度帧
        """
        if not self.enable_filters:
            return depth_frame
        
        try:
            depth_frame = self.threshold_filter.process(depth_frame)
            depth_frame = self.depth_to_disparity.process(depth_frame)
            depth_frame = self.spatial_filter.process(depth_frame)
            depth_frame = self.temporal_filter.process(depth_frame)
            depth_frame = self.disparity_to_depth.process(depth_frame)
            depth_frame = self.hole_filling_filter.process(depth_frame)
            return depth_frame
        except Exception as e:
            self.logger.warning(f"Depth filtering failed: {e}")
            return depth_frame
    
    def start_capture_thread(self):
        """启动后台线程持续采集图像"""
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._optimized_capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        self.logger.info("Camera capture thread started")
    
    def _optimized_capture_loop(self):
        """优化的采集循环 - 减少阻塞"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        # 帧率统计
        fps_counter = deque(maxlen=30)
        last_fps_report = time.time()
        
        while self.is_running:
            loop_start = time.time()
            
            try:
                # 非阻塞获取帧
                frames = self.pipeline.poll_for_frames()
                if not frames:
                    time.sleep(0.001)  # 短暂休眠
                    continue
                
                # 对齐深度图到RGB图
                aligned_frames = self.align_to_color.process(frames)
                
                # 获取对齐后的深度帧和彩色帧
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                
                # 检测运动模糊
                is_blurry, blur_score = self.detect_motion_blur(color_image)
                
                # 优化的深度处理
                if self.enable_filters:
                    aligned_depth_frame = self.process_depth_frame_simple(aligned_depth_frame)
                
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                
                # 创建帧数据
                frame_data = {
                    'color': color_image,
                    'depth': depth_image,
                    'timestamp': frames.get_timestamp() / 1000.0,  # 转换为秒
                    'is_blurry': is_blurry,
                    'blur_score': blur_score
                }
                
                # 非阻塞入队
                try:
                    self.frame_queue.put_nowait(frame_data)
                    consecutive_errors = 0
                except:
                    # 队列满，丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except:
                        pass
                
                # FPS统计
                fps_counter.append(time.time())
                # if time.time() - last_fps_report > 5.0:  # 每5秒报告一次
                #     if len(fps_counter) > 1:
                #         fps = len(fps_counter) / (fps_counter[-1] - fps_counter[0])
                #         self.logger.info(f"相机FPS: {fps:.1f}")
                #     last_fps_report = time.time()
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:
                    self.logger.warning(f"Capture error: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error("Too many consecutive errors, stopping capture")
                    break
                
                time.sleep(0.1)
            
            # 动态休眠以维持稳定帧率
            loop_time = time.time() - loop_start
            if loop_time < 0.033:  # 目标30fps
                time.sleep(0.033 - loop_time)
    
    def _capture_loop(self):
        """简化的后台采集循环（兼容原版本）"""
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while self.is_running:
            try:
                # 设置较短的超时时间
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # 对齐深度图到RGB图
                aligned_frames = self.align_to_color.process(frames)
                
                # 获取对齐后的深度帧和彩色帧
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # 简化的深度处理
                if self.enable_filters:
                    aligned_depth_frame = self.process_depth_frame_simple(aligned_depth_frame)
                
                # 转换为numpy数组
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # 非阻塞地更新队列
                try:
                    # 如果队列满了，移除最老的帧
                    while self.frame_queue.qsize() >= self.frame_queue.maxsize:
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            break
                    
                    # 添加新帧
                    self.frame_queue.put_nowait((color_image, depth_image))
                    consecutive_errors = 0  # 重置错误计数
                    
                except:
                    # 队列操作失败，跳过这一帧
                    continue
                
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3:  # 只记录前几次错误
                    self.logger.warning(f"Error in capture loop: {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error("Too many consecutive errors, stopping capture thread")
                    break
                
                # 错误后短暂等待
                time.sleep(0.1)
    
    def get_obs(self, timeout=2.0, skip_blurry=True):
        """
        获取最新的RGB和深度图像
        
        Args:
            timeout: 超时时间（秒）
            skip_blurry: 是否跳过模糊帧
            
        Returns:
            tuple: (color_image, depth_image) 或 (color_image, depth_image, metadata)
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                # 检查数据格式（兼容旧版本）
                if isinstance(frame_data, dict):
                    # 新格式
                    if skip_blurry and frame_data['is_blurry']:
                        self.logger.debug(f"跳过模糊帧 (blur_score: {frame_data['blur_score']:.1f})")
                        continue
                    
                    return frame_data['color'], frame_data['depth']
                else:
                    # 旧格式 (color_image, depth_image)
                    return frame_data
                
            except Empty:
                continue
            except Exception as e:
                raise ValueError(f"Failed to get frames: {e}")
        
        raise ValueError(f"Failed to get frames within {timeout}s timeout")
    
    def get_obs_with_metadata(self, timeout=2.0, skip_blurry=True):
        """
        获取带元数据的RGB和深度图像
        
        Returns:
            tuple: (color_image, depth_image, metadata)
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                if isinstance(frame_data, dict):
                    if skip_blurry and frame_data['is_blurry']:
                        continue
                    
                    return frame_data['color'], frame_data['depth'], {
                        'timestamp': frame_data['timestamp'],
                        'blur_score': frame_data['blur_score'],
                        'is_blurry': frame_data['is_blurry']
                    }
                else:
                    # 旧格式，创建默认元数据
                    return frame_data[0], frame_data[1], {
                        'timestamp': time.time(),
                        'blur_score': 0.0,
                        'is_blurry': False
                    }
                
            except Empty:
                continue
        
        raise ValueError(f"Failed to get frames within {timeout}s timeout")
    
    def get_latest_obs(self, skip_blurry=True):
        """
        获取最新的观测数据（非阻塞）
        
        Returns:
            tuple: (color_image, depth_image, metadata) 或 None
        """
        latest_frame = None
        
        # 清空队列并获取最新帧
        while True:
            try:
                frame_data = self.frame_queue.get_nowait()
                if isinstance(frame_data, dict):
                    if not skip_blurry or not frame_data['is_blurry']:
                        latest_frame = frame_data
                else:
                    # 兼容旧格式
                    latest_frame = {'color': frame_data[0], 'depth': frame_data[1], 
                                  'timestamp': time.time(), 'blur_score': 0, 'is_blurry': False}
            except Empty:
                break
        
        if latest_frame:
            return latest_frame['color'], latest_frame['depth'], {
                'timestamp': latest_frame['timestamp'],
                'blur_score': latest_frame['blur_score'],
                'is_blurry': latest_frame['is_blurry']
            }
        
        return None
    
    def get_intrinsic(self):
        """获取深度和RGB相机的内参"""
        return {
            'depth': {
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'ppx': self.depth_intrinsics.ppx,
                'ppy': self.depth_intrinsics.ppy,
                'width': self.depth_intrinsics.width,
                'height': self.depth_intrinsics.height
            },
            'color': {
                'fx': self.color_intrinsics.fx,
                'fy': self.color_intrinsics.fy,
                'ppx': self.color_intrinsics.ppx,
                'ppy': self.color_intrinsics.ppy,
                'width': self.color_intrinsics.width,
                'height': self.color_intrinsics.height
            }
        }
    
    def set_blur_threshold(self, threshold):
        """动态设置模糊阈值"""
        self.blur_threshold = threshold
        self.logger.info(f"模糊阈值设置为: {threshold}")
    
    def show_obs(self):
        """显示单帧图像"""
        try:
            color_img, depth_img = self.get_obs()
            
            # 显示RGB图像
            cv2.imshow("RGB Image", color_img)
            
            # 显示深度图像
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_img, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            cv2.imshow("Depth Image", depth_colormap)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            self.logger.error(f"Failed to show observation: {e}")
    
    def show_realtime(self, update_interval=0.033):
        """
        实时显示图像流
        
        Args:
            update_interval: 更新间隔（秒）
        """
        self.logger.info("实时显示开始，按 'q' 退出")
        
        while True:
            try:
                # 获取最新帧（非阻塞）
                result = self.get_latest_obs()
                if result is None:
                    time.sleep(update_interval)
                    continue
                
                color_img, depth_img, metadata = result
                
                # 创建深度图的彩色表示
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_img, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # 并排显示
                combined_img = np.hstack((color_img, depth_colormap))
                
                # 添加深度范围信息
                valid_depth = depth_img[depth_img > 0]
                if len(valid_depth) > 0:
                    min_depth = np.min(valid_depth) / 1000.0  # 转换为米
                    max_depth = np.max(valid_depth) / 1000.0
                    cv2.putText(combined_img, f"Depth: {min_depth:.2f}m - {max_depth:.2f}m", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 添加队列状态
                queue_size = self.frame_queue.qsize()
                cv2.putText(combined_img, f"Queue: {queue_size}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 添加模糊检测信息
                if self.enable_motion_blur_detection:
                    blur_score = metadata.get('blur_score', 0)
                    is_blurry = metadata.get('is_blurry', False)
                    blur_text = f"Blur: {blur_score:.1f} {'(BLURRY)' if is_blurry else ''}"
                    color = (0, 0, 255) if is_blurry else (0, 255, 0)
                    cv2.putText(combined_img, blur_text, 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                cv2.imshow("Real-Time RGB and Depth", combined_img)
                
                # 按下'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
                time.sleep(update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in realtime display: {e}")
                time.sleep(0.5)
        
        cv2.destroyAllWindows()
        self.logger.info("实时显示结束")
    
    def is_healthy(self):
        """检查相机状态是否健康"""
        return (self.is_running and 
                self.capture_thread and 
                self.capture_thread.is_alive() and
                not self.frame_queue.empty())
    
    def get_status(self):
        """获取相机状态信息"""
        return {
            'is_running': self.is_running,
            'thread_alive': self.capture_thread.is_alive() if self.capture_thread else False,
            'queue_size': self.frame_queue.qsize(),
            'queue_maxsize': self.frame_queue.maxsize,
            'filters_enabled': self.enable_filters,
            'motion_blur_detection': self.enable_motion_blur_detection,
            'blur_threshold': self.blur_threshold
        }
    
    def restart_capture(self):
        """重启采集线程（用于错误恢复）"""
        self.logger.info("重启相机采集线程...")
        
        # 停止当前线程
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=3.0)
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        # 重启线程
        self.start_capture_thread()
        self.logger.info("相机采集线程重启完成")
    
    def stop(self):
        """停止相机"""
        self.logger.info("停止相机...")
        
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=3.0)
            if self.capture_thread.is_alive():
                self.logger.warning("采集线程未能正常结束")
        
        try:
            self.pipeline.stop()
        except Exception as e:
            self.logger.warning(f"停止管道时出错: {e}")
        
        cv2.destroyAllWindows()
        self.logger.info("相机已停止")

# 向后兼容的别名
RealSenseCamera = OptimizedRealSenseCamera

# 测试函数
def test_camera():
    """测试相机功能"""
    print("=== 相机测试开始 ===")
    
    try:
        # 创建相机实例（启用所有优化功能）
        camera = OptimizedRealSenseCamera(
            device_index=1, 
            enable_filters=True,
            depth_preset='high_accuracy',  # 测试深度预设
            enable_motion_blur_detection=True  # 测试模糊检测
        )
        
        print("相机初始化成功")
        
        # 显示相机状态
        status = camera.get_status()
        print(f"相机状态: {status}")
        
        # 测试单次获取
        print("测试单次图像获取...")
        color_img, depth_img = camera.get_obs()
        print(f"图像大小: RGB {color_img.shape}, Depth {depth_img.shape}")
        
        
        # 实时显示测试（可选）
        test_realtime = input("\n是否测试实时显示? (y/n): ").strip().lower()
        if test_realtime == 'y':
            print("将显示模糊检测信息...")
            camera.show_realtime()
        
    except Exception as e:
        print(f"相机测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            camera.stop()
        except:
            pass
        print("\n=== 相机测试结束 ===")

if __name__ == "__main__":
    test_camera()