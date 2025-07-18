import os
import sys
import time
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from PIL import Image

# Import the NavAgent class (assuming it's in Agent.py)
from Agent import NavAgent

from franky import *
import math
from scipy.spatial.transform import Rotation

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 预定义的导航任务
PREDEFINED_TASKS = {
    # 文本导航任务
    "text": [
        # "A yellow armchair",
        # "A model of little robot Wally in the movie Wall-E,
        # "The desk with computer monitor",
        # "The entrance door",
        # "The kitchen area",
        # "A bookshelf with books",
        # "The sofa near the window",
        # "The dining table",
        # "A plant in a pot",
        # "The TV stand"
    ],
    # 图像导航任务（需要准备对应的图像文件）
    "image": [
        # "query_images/chair.jpg",
        # "query_images/desk.jpg",
        # "query_images/door.jpg",
        # "query_images/kitchen.jpg",
        # "query_images/sofa.jpg"
    ],

    "category": [
        "chair",
        "table", 
        "sofa",
        "bed",
        "toilet",
        "tv",
        "laptop",
        "refrigerator",
        "microwave",
        "oven",
        "sink",
        "book",
        "vase",
        "cup",
        "bottle",
        "plant",
        "trash bin"
    ]
}

# 导航配置
NAVIGATION_CONFIG = {
    "ws_url": "ws://192.168.1.102:9090",
    "http_url": "http://192.168.1.102/apiUrl",
    "map_name": "exp-4",
    "memory_path": "memory/exp-4-2",
    # "initial_pose": (429, 1097, 0),  # PNG坐标系下的初始位置
    # "initial_pose": (634, 843, 0),  # PNG坐标系下的初始位置
    "initial_pose": (440, 1068, 0),  # PNG坐标系下的初始位置
    # "initial_pose": (485, 1021, 0),  # PNG坐标系下的初始位置
    "camera_device_index": 1
}

# 机器人配置
ROBOT_CONFIG = {
    "ws_url": "ws://192.168.1.102:9090",
    "http_url": "http://192.168.1.102/apiUrl"
}

class NavigationInterface:
    """导航系统交互界面"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agent = None
        self.navigation_history = []
        
    def print_banner(self):
        """打印欢迎界面"""
        print("\n" + "="*60)
        print("🤖 视觉导航系统 - Visual Navigation System")
        print("="*60)
        print(f"地图: {self.config['map_name']}")
        print(f"记忆路径: {self.config['memory_path']}")
        print(f"初始位置: {self.config['initial_pose']}")
        print("="*60 + "\n")
    
    def print_main_menu(self):
        """打印主菜单"""
        print("\n" + "-"*50)
        print("📋 主菜单 - Main Menu")
        print("-"*50)
        print("1. 🔤 文本导航 (Text Navigation)")
        print("2. 🖼️  图像导航 (Image Navigation)")
        print("3. 🏷️  类别导航 (Category Navigation)")  # 新增
        print("4. 🎯 预设任务 (Predefined Tasks)")
        print("5. 📊 查看历史 (View History)")
        print("6. ⚙️  设置选项 (Settings)")
        print("7. 🚪 退出系统 (Exit)")
        print("-"*50)   
    
    def print_text_tasks(self):
        """打印预定义的文本任务"""
        print("\n📝 预定义文本导航任务:")
        for i, task in enumerate(PREDEFINED_TASKS["text"], 1):
            print(f"  {i}. {task}")
        print(f"  0. 返回主菜单")
    
    def print_image_tasks(self):
        """打印预定义的图像任务"""
        print("\n🖼️  预定义图像导航任务:")
        for i, task in enumerate(PREDEFINED_TASKS["image"], 1):
            print(f"  {i}. {task}")
        print(f"  0. 返回主菜单")
    
    def get_navigation_settings(self) -> Dict:
        """获取导航设置"""
        print("\n⚙️  导航设置:")
        settings = {
            "visualize": True,
            "wait_for_arrival": True,
            "record_video": True,
            "save_summary": True
        }
        
        # 询问用户是否要修改默认设置
        use_default = input("使用默认设置? (y/n) [y]: ").lower().strip()
        if use_default == 'n':
            settings["visualize"] = input("显示可视化? (y/n) [y]: ").lower() != 'n'
            settings["wait_for_arrival"] = input("等待到达目标? (y/n) [y]: ").lower() != 'n'
            settings["record_video"] = input("录制导航视频? (y/n) [y]: ").lower() != 'n'
            settings["save_summary"] = input("保存导航总结? (y/n) [y]: ").lower() != 'n'
        
        return settings
    
    def handle_text_navigation(self):
        """处理文本导航"""
        while True:
            print("\n🔤 文本导航模式")
            print("1. 输入自定义文本描述")
            print("2. 选择预定义任务")
            print("3. 抓取任务导航")
            print("0. 返回主菜单")
            
            choice = input("\n请选择 (0-3): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                # 自定义文本输入
                text_prompt = input("\n请输入目标描述: ").strip()
                if text_prompt:
                    self.execute_text_navigation(text_prompt)
            elif choice == '3':
                # 自定义文本输入
                text_prompt = input("\n请输入目标描述: ").strip()
                robot = RobotGrasp()
                if text_prompt:
                    robot.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419])
                    # robot.joint_initialize(initial_pos=[-0.0106303, -0.610327, 0.021196, -1.80402, 0.0301998, 1.33754, 0.883195])
                    # demo1
                    # self.execute_text_grasp_navigation(text_prompt, [2.45, 0.81, -4.90])
                    # use_default = input("是否继续抓取? (y/n) [y]: ").lower().strip()
                    # if use_default == 'n':
                    #     break
                    # time.sleep(3)
                    # robot.run()
                    # use_default = input("是否继续导航? (y/n) [y]: ").lower().strip()
                    # if use_default == 'n':
                    #     break
                    # # self.execute_text_grasp_navigation(text_prompt, [12.04, -6.67, -99.66])
                    # self.execute_text_grasp_navigation(text_prompt, [12.11, -6.77, -146.77])
                    # time.sleep(3)
                    # robot.place()

                    # # demo2
                    # self.execute_text_grasp_navigation(text_prompt, [19.20, -11.65, 120.96])
                    # use_default = input("是否继续抓取? (y/n) [y]: ").lower().strip()
                    # if use_default == 'n':
                    #     break
                    # time.sleep(3)
                    # robot.clean()


                    # demo3
                    # 罐子
                    self.execute_text_grasp_navigation(text_prompt, [2.19, -0.22, 21.21])
                    use_default = input("是否继续抓取? (y/n) [y]: ").lower().strip()
                    if use_default == 'n':
                        break
                    time.sleep(3)
                    robot.demo3complex()

                    # 取盒子
                    self.execute_text_grasp_navigation(text_prompt, [12.11, -6.77, -146.77])
                    use_default = input("是否继续抓取? (y/n) [y]: ").lower().strip()
                    if use_default == 'n':
                        break
                    time.sleep(3)
                    robot.demo3move2()

                    # 倒盒子
                    self.execute_text_grasp_navigation(text_prompt, [2.19, -0.22, 21.21])
                    use_default = input("是否继续抓取? (y/n) [y]: ").lower().strip()
                    if use_default == 'n':
                        break
                    time.sleep(3)
                    robot.demo3pour2()

                    # 取牛奶
                    self.execute_text_grasp_navigation(text_prompt, [15.24, -13.33, -78.49])
                    use_default = input("是否继续抓取? (y/n) [y]: ").lower().strip()
                    if use_default == 'n':
                        break
                    time.sleep(3)
                    robot.demo3move3()

                    # 倒牛奶
                    self.execute_text_grasp_navigation(text_prompt, [2.19, -0.22, 21.21])
                    use_default = input("是否继续抓取? (y/n) [y]: ").lower().strip()
                    if use_default == 'n':
                        break
                    time.sleep(3)
                    robot.demo3pour3()



            elif choice == '2':
                # 预定义任务
                self.print_text_tasks()
                task_choice = input("\n请选择任务 (0-{}): ".format(len(PREDEFINED_TASKS["text"]))).strip()
                
                try:
                    task_idx = int(task_choice)
                    if task_idx == 0:
                        continue
                    elif 1 <= task_idx <= len(PREDEFINED_TASKS["text"]):
                        text_prompt = PREDEFINED_TASKS["text"][task_idx - 1]
                        self.execute_text_navigation(text_prompt)
                    else:
                        print("❌ 无效的选择")
                except ValueError:
                    print("❌ 请输入有效的数字")
    

    def handle_category_navigation(self):
        """处理类别导航"""
        while True:
            print("\n🏷️  类别导航模式")
            print("1. 输入自定义类别")
            print("2. 选择预定义类别")
            print("3. 设置搜索参数")
            print("0. 返回主菜单")
            
            choice = input("\n请选择 (0-3): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                # 自定义类别输入
                category = input("\n请输入目标类别 (如: chair, table, sofa等): ").strip()
                if category:
                    self.execute_category_navigation(category)
            elif choice == '2':
                # 预定义类别
                self.print_category_list()
                category_choice = input("\n请选择类别 (0-{}): ".format(len(PREDEFINED_TASKS["category"]))).strip()
                
                try:
                    cat_idx = int(category_choice)
                    if cat_idx == 0:
                        continue
                    elif 1 <= cat_idx <= len(PREDEFINED_TASKS["category"]):
                        category = PREDEFINED_TASKS["category"][cat_idx - 1]
                        self.execute_category_navigation(category)
                    else:
                        print("❌ 无效的选择")
                except ValueError:
                    print("❌ 请输入有效的数字")
            elif choice == '3':
                # 设置搜索参数
                self.set_category_search_params()

    def print_category_list(self):
        """打印预定义的类别列表"""
        print("\n🏷️  预定义物体类别:")
        categories = PREDEFINED_TASKS["category"]
        
        # 分列显示
        cols = 3
        for i in range(0, len(categories), cols):
            row = ""
            for j in range(cols):
                if i + j < len(categories):
                    row += f"{i+j+1:2d}. {categories[i+j]:15s} "
            print(row)
        print(f" 0. 返回上级菜单")

    def set_category_search_params(self):
        """设置类别搜索参数"""
        print("\n⚙️  类别导航参数设置:")
        
        # 获取当前默认值
        if not hasattr(self, 'category_nav_params'):
            self.category_nav_params = {
                'distance_weight': 0.7,
                'confidence_weight': 0.3,
                'max_distance': 20.0
            }
        
        print(f"当前参数:")
        print(f"  距离权重: {self.category_nav_params['distance_weight']}")
        print(f"  置信度权重: {self.category_nav_params['confidence_weight']}")
        print(f"  最大搜索距离: {self.category_nav_params['max_distance']}米")
        
        modify = input("\n是否修改? (y/n) [n]: ").lower().strip()
        if modify == 'y':
            try:
                dist_w = float(input(f"距离权重 (0-1) [{self.category_nav_params['distance_weight']}]: ") 
                            or self.category_nav_params['distance_weight'])
                conf_w = float(input(f"置信度权重 (0-1) [{self.category_nav_params['confidence_weight']}]: ") 
                            or self.category_nav_params['confidence_weight'])
                max_dist = float(input(f"最大搜索距离(米) [{self.category_nav_params['max_distance']}]: ") 
                            or self.category_nav_params['max_distance'])
                
                # 归一化权重
                total_weight = dist_w + conf_w
                if total_weight > 0:
                    self.category_nav_params['distance_weight'] = dist_w / total_weight
                    self.category_nav_params['confidence_weight'] = conf_w / total_weight
                    self.category_nav_params['max_distance'] = max_dist
                    print("✅ 参数已更新")
                else:
                    print("❌ 权重和必须大于0")
            except ValueError:
                print("❌ 无效的数值")

    def execute_category_navigation(self, category: str):
        """执行类别导航任务"""
        settings = self.get_navigation_settings()
        
        # 获取类别导航特定参数
        if not hasattr(self, 'category_nav_params'):
            self.category_nav_params = {
                'distance_weight': 0.7,
                'confidence_weight': 0.3,
                'max_distance': 20.0
            }
        
        print(f"\n🚀 开始类别导航: 寻找 '{category}'")
        print("导航设置:", settings)
        print("搜索参数:", self.category_nav_params)
        
        start_time = time.time()
        
        try:
            success = self.agent.Nav2Category(
                category,
                visualize=settings["visualize"],
                wait_for_arrival=settings["wait_for_arrival"],
                record_video=settings["record_video"],
                save_summary=settings["save_summary"],
                distance_weight=self.category_nav_params['distance_weight'],
                confidence_weight=self.category_nav_params['confidence_weight'],
                max_distance=self.category_nav_params['max_distance']
            )
            
            elapsed_time = time.time() - start_time
            
            # 获取导航统计信息
            nav_stats = self.agent.get_navigation_statistics()
            path_length = nav_stats['path_length']
            avg_speed = nav_stats['average_speed']
            
            # 记录导航历史
            self.navigation_history.append({
                "type": "category",
                "target": category,
                "success": success,
                "time": elapsed_time,
                "path_length": path_length,
                "average_speed": avg_speed,
                "trajectory_points": nav_stats['trajectory_points'],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "settings": settings,
                "search_params": self.category_nav_params.copy()
            })
            
            # 打印导航结果和统计信息
            if success:
                print(f"\n✅ 成功找到并到达 '{category}'!")
            else:
                print(f"\n❌ 未能到达 '{category}'")
            
            print(f"\n📊 导航统计:")
            print(f"  ⏱️  总用时: {elapsed_time:.1f} 秒")
            print(f"  📏 路径长度: {path_length:.2f} 米")
            print(f"  🚀 平均速度: {avg_speed:.2f} 米/秒")
            print(f"  📍 轨迹点数: {nav_stats['trajectory_points']}")
                
        except Exception as e:
            logger.error(f"类别导航执行错误: {e}")
            print(f"\n❌ 导航出错: {str(e)}")
        
        input("\n按回车键继续...")


    def handle_image_navigation(self):
        """处理图像导航"""
        while True:
            print("\n🖼️  图像导航模式")
            print("1. 输入图像路径")
            print("2. 选择预定义图像")
            print("0. 返回主菜单")
            
            choice = input("\n请选择 (0-2): ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                # 自定义图像路径
                image_path = input("\n请输入图像路径: ").strip()
                if image_path and os.path.exists(image_path):
                    self.execute_image_navigation(image_path)
                else:
                    print(f"❌ 图像文件不存在: {image_path}")
            elif choice == '2':
                # 预定义图像
                self.print_image_tasks()
                task_choice = input("\n请选择任务 (0-{}): ".format(len(PREDEFINED_TASKS["image"]))).strip()
                
                try:
                    task_idx = int(task_choice)
                    if task_idx == 0:
                        continue
                    elif 1 <= task_idx <= len(PREDEFINED_TASKS["image"]):
                        image_path = PREDEFINED_TASKS["image"][task_idx - 1]
                        if os.path.exists(image_path):
                            self.execute_image_navigation(image_path)
                        else:
                            print(f"❌ 图像文件不存在: {image_path}")
                    else:
                        print("❌ 无效的选择")
                except ValueError:
                    print("❌ 请输入有效的数字")
    
    def execute_text_grasp_navigation(self, text_prompt: str, target_croods):
        """执行抓取位置导航任务"""
        settings = self.get_navigation_settings()
        
        print(f"\n🚀 开始抓取任务导航: '{text_prompt}'")
        print("导航设置:", settings)
        
        start_time = time.time()
        
        try:
            success = self.agent.Nav2Text_grasp(
                target_croods,
                text_prompt,
                # visualize=settings["visualize"],
                visualize=False,
                wait_for_arrival=settings["wait_for_arrival"],
                # record_video=settings["record_video"],
                record_video=False,
                save_summary=settings["save_summary"]
            )
            
            elapsed_time = time.time() - start_time
            
            # 获取导航统计信息
            nav_stats = self.agent.get_navigation_statistics()
            path_length = nav_stats['path_length']
            avg_speed = nav_stats['average_speed']
            
            # 记录导航历史（增加路径长度信息）
            self.navigation_history.append({
                "type": "text",
                "target": text_prompt,
                "success": success,
                "time": elapsed_time,
                "path_length": path_length,  # 新增
                "average_speed": avg_speed,   # 新增
                "trajectory_points": nav_stats['trajectory_points'],  # 新增
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "settings": settings
            })
            
            # 打印导航结果和统计信息
            if success:
                print(f"\n✅ 导航成功!")
            else:
                print(f"\n❌ 导航失败!")
            
            print(f"\n📊 导航统计:")
            print(f"  ⏱️  总用时: {elapsed_time:.1f} 秒")
            print(f"  📏 路径长度: {path_length:.2f} 米")
            print(f"  🚀 平均速度: {avg_speed:.2f} 米/秒")
            print(f"  📍 轨迹点数: {nav_stats['trajectory_points']}")
                
        except Exception as e:
            logger.error(f"导航执行错误: {e}")
            print(f"\n❌ 导航出错: {str(e)}")
        
        input("\n按回车键继续...")
    
    def execute_text_navigation(self, text_prompt: str):
        """执行文本导航任务"""
        settings = self.get_navigation_settings()
        
        print(f"\n🚀 开始文本导航: '{text_prompt}'")
        print("导航设置:", settings)
        
        start_time = time.time()
        
        try:
            success = self.agent.Nav2Text(
                text_prompt,
                visualize=settings["visualize"],
                wait_for_arrival=settings["wait_for_arrival"],
                record_video=settings["record_video"],
                save_summary=settings["save_summary"]
            )
            
            elapsed_time = time.time() - start_time
            
            # 获取导航统计信息
            nav_stats = self.agent.get_navigation_statistics()
            path_length = nav_stats['path_length']
            avg_speed = nav_stats['average_speed']
            
            # 记录导航历史（增加路径长度信息）
            self.navigation_history.append({
                "type": "text",
                "target": text_prompt,
                "success": success,
                "time": elapsed_time,
                "path_length": path_length,  # 新增
                "average_speed": avg_speed,   # 新增
                "trajectory_points": nav_stats['trajectory_points'],  # 新增
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "settings": settings
            })
            
            # 打印导航结果和统计信息
            if success:
                print(f"\n✅ 导航成功!")
            else:
                print(f"\n❌ 导航失败!")
            
            print(f"\n📊 导航统计:")
            print(f"  ⏱️  总用时: {elapsed_time:.1f} 秒")
            print(f"  📏 路径长度: {path_length:.2f} 米")
            print(f"  🚀 平均速度: {avg_speed:.2f} 米/秒")
            print(f"  📍 轨迹点数: {nav_stats['trajectory_points']}")
                
        except Exception as e:
            logger.error(f"导航执行错误: {e}")
            print(f"\n❌ 导航出错: {str(e)}")
        
        input("\n按回车键继续...")
    
    def execute_image_navigation(self, image_path: str):
        """执行图像导航任务"""
        settings = self.get_navigation_settings()
        
        print(f"\n🚀 开始图像导航: '{image_path}'")
        print("导航设置:", settings)
        
        start_time = time.time()
        
        try:
            success = self.agent.Nav2Img(
                image_path,
                visualize=settings["visualize"],
                wait_for_arrival=settings["wait_for_arrival"],
                record_video=settings["record_video"],
                save_summary=settings["save_summary"]
            )
            
            elapsed_time = time.time() - start_time
            
            # 获取导航统计信息
            nav_stats = self.agent.get_navigation_statistics()
            path_length = nav_stats['path_length']
            avg_speed = nav_stats['average_speed']
            
            # 记录导航历史（增加路径长度信息）
            self.navigation_history.append({
                "type": "image",
                "target": image_path,
                "success": success,
                "time": elapsed_time,
                "path_length": path_length,  # 新增
                "average_speed": avg_speed,   # 新增
                "trajectory_points": nav_stats['trajectory_points'],  # 新增
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "settings": settings
            })
            
            # 打印导航结果和统计信息
            if success:
                print(f"\n✅ 导航成功!")
            else:
                print(f"\n❌ 导航失败!")
            
            print(f"\n📊 导航统计:")
            print(f"  ⏱️  总用时: {elapsed_time:.1f} 秒")
            print(f"  📏 路径长度: {path_length:.2f} 米")
            print(f"  🚀 平均速度: {avg_speed:.2f} 米/秒")
            print(f"  📍 轨迹点数: {nav_stats['trajectory_points']}")
                
        except Exception as e:
            logger.error(f"导航执行错误: {e}")
            print(f"\n❌ 导航出错: {str(e)}")
        
        input("\n按回车键继续...")
    
    def handle_predefined_tasks(self):
        """处理预定义任务批量执行"""
        print("\n🎯 预定义任务执行")
        print("1. 执行所有文本任务")
        print("2. 执行所有图像任务")
        print("3. 执行所有类别任务")  # 新增
        print("4. 自定义任务序列")
        print("0. 返回主菜单")
        
        choice = input("\n请选择 (0-4): ").strip()
        
        if choice == '1':
            # 执行所有文本任务
            self.execute_task_sequence(PREDEFINED_TASKS["text"], "text")
        elif choice == '2':
            # 执行所有图像任务
            self.execute_task_sequence(PREDEFINED_TASKS["image"], "image")
        elif choice == '3':
            # 执行所有类别任务
            self.execute_task_sequence(PREDEFINED_TASKS["category"], "category")
        elif choice == '4':
            # 自定义任务序列
            self.create_custom_sequence()
    
    def execute_task_sequence(self, tasks: List[str], task_type: str):
        """执行任务序列"""
        print(f"\n📋 将执行 {len(tasks)} 个{task_type}任务")
        confirm = input("确认执行? (y/n) [y]: ").lower().strip()
        
        if confirm == 'n':
            return
        
        settings = self.get_navigation_settings()
        
        # 任务序列统计
        sequence_start_time = time.time()
        total_path_length = 0
        successful_tasks = 0
        
        for i, task in enumerate(tasks, 1):
            print(f"\n[{i}/{len(tasks)}] 执行任务: {task}")
            
            if task_type == "text":
                self.execute_text_navigation(task)
            elif task_type == "image":
                if os.path.exists(task):
                    self.execute_image_navigation(task)
                else:
                    print(f"❌ 跳过不存在的图像: {task}")
            elif task_type == "category":  # 新增
                self.execute_category_navigation(task)
            
            # 累计统计（从最新的历史记录获取）
            if self.navigation_history:
                last_record = self.navigation_history[-1]
                if last_record["success"]:
                    successful_tasks += 1
                if 'path_length' in last_record:
                    total_path_length += last_record['path_length']
            
            if i < len(tasks):
                wait_time = 5
                print(f"\n⏳ {wait_time}秒后执行下一个任务...")
                time.sleep(wait_time)
        
        # 打印序列总结
        sequence_time = time.time() - sequence_start_time
        print(f"\n📊 任务序列完成!")
        print(f"  总任务数: {len(tasks)}")
        print(f"  成功任务: {successful_tasks}")
        print(f"  总用时: {sequence_time:.1f}秒")
        print(f"  总路径长度: {total_path_length:.2f}米")
        if sequence_time > 0:
            print(f"  平均速度: {total_path_length/sequence_time:.2f}米/秒")
    
    def create_custom_sequence(self):
        """创建自定义任务序列"""
        tasks = []
        print("\n创建自定义任务序列（输入'done'完成）:")
        
        while True:
            task_type = input("\n任务类型 (text/image/category/done): ").lower().strip()
            
            if task_type == 'done':
                break
            elif task_type == 'text':
                text = input("输入文本描述: ").strip()
                if text:
                    tasks.append(("text", text))
            elif task_type == 'image':
                path = input("输入图像路径: ").strip()
                if path:
                    tasks.append(("image", path))
            elif task_type == 'category':  # 新增
                category = input("输入物体类别: ").strip()
                if category:
                    tasks.append(("category", category))
            else:
                print("❌ 无效的任务类型")
        
        if tasks:
            print(f"\n将执行 {len(tasks)} 个任务")
            for task_type, target in tasks:
                if task_type == "text":
                    self.execute_text_navigation(target)
                elif task_type == "image":
                    self.execute_image_navigation(target)
                elif task_type == "category":
                    self.execute_category_navigation(target)
                time.sleep(3)
    
    def view_history(self):
        """查看导航历史"""
        if not self.navigation_history:
            print("\n📊 暂无导航历史")
            return
        
        print(f"\n📊 导航历史 (共 {len(self.navigation_history)} 条)")
        print("-" * 80)
        
        for i, record in enumerate(self.navigation_history, 1):
            status = "✅ 成功" if record["success"] else "❌ 失败"
            print(f"{i}. [{record['timestamp']}] {record['type'].upper()} - {status}")
            print(f"   目标: {record['target']}")
            print(f"   用时: {record['time']:.1f}秒")
            print(f"   路径长度: {record.get('path_length', 'N/A'):.2f}米")  # 新增
            print(f"   平均速度: {record.get('average_speed', 'N/A'):.2f}米/秒")  # 新增
            print(f"   轨迹点数: {record.get('trajectory_points', 'N/A')}")  # 新增
            print(f"   设置: 可视化={record['settings']['visualize']}, "
                f"录像={record['settings']['record_video']}")
            print("-" * 80)
        
        # 统计信息
        success_count = sum(1 for r in self.navigation_history if r["success"])
        total_time = sum(r["time"] for r in self.navigation_history)
        
        # 计算总路径长度（只计算有路径长度记录的）
        records_with_path = [r for r in self.navigation_history if 'path_length' in r]
        if records_with_path:
            total_path_length = sum(r["path_length"] for r in records_with_path)
            avg_path_length = total_path_length / len(records_with_path)
        else:
            total_path_length = 0
            avg_path_length = 0
        
        print(f"\n📈 统计:")
        print(f"   成功率: {success_count}/{len(self.navigation_history)} "
            f"({success_count/len(self.navigation_history)*100:.1f}%)")
        print(f"   总用时: {total_time:.1f}秒")
        print(f"   平均用时: {total_time/len(self.navigation_history):.1f}秒")
        
        if records_with_path:
            print(f"   总路径长度: {total_path_length:.2f}米")
            print(f"   平均路径长度: {avg_path_length:.2f}米")
            print(f"   平均行驶速度: {total_path_length/total_time:.2f}米/秒")
        
        input("\n按回车键继续...")

    
    def settings_menu(self):
        """设置菜单"""
        print("\n⚙️  系统设置")
        print("1. 修改初始位置")
        print("2. 修改相机设备索引")
        print("3. 查看当前配置")
        print("0. 返回主菜单")
        
        choice = input("\n请选择 (0-3): ").strip()
        
        if choice == '1':
            try:
                x = float(input("输入X坐标: "))
                y = float(input("输入Y坐标: "))
                theta = float(input("输入角度(弧度) [0]: ") or "0")
                
                # 重新初始化agent
                print("重新初始化导航系统...")
                self.agent.shutdown()
                self.config["initial_pose"] = (x, y, theta)
                self.initialize_agent()
                
            except ValueError:
                print("❌ 无效的坐标值")
                
        elif choice == '2':
            try:
                idx = int(input("输入相机设备索引: "))
                self.config["camera_device_index"] = idx
                print("⚠️  需要重启系统以应用更改")
            except ValueError:
                print("❌ 无效的设备索引")
                
        elif choice == '3':
            print("\n当前配置:")
            for key, value in self.config.items():
                print(f"  {key}: {value}")
        
        input("\n按回车键继续...")
    
    def initialize_agent(self):
        """初始化导航代理"""
        print("\n正在初始化导航系统...")
        try:
            self.agent = NavAgent(
                ws_url=self.config["ws_url"],
                http_url=self.config["http_url"],
                map_name=self.config["map_name"],
                memory_path=self.config["memory_path"],
                initial_pose=self.config["initial_pose"],
                camera_device_index=self.config["camera_device_index"],
                load_memory=False,
                load_diffusion=False
            )
            print("✅ 导航系统初始化成功")
            return True
        except Exception as e:
            logger.error(f"导航系统初始化失败: {e}")
            print(f"❌ 导航系统初始化失败: {str(e)}")
            return False
    
    def run(self):
        """运行主程序"""
        self.print_banner()
        
        # 初始化导航系统
        if not self.initialize_agent():
            print("\n系统初始化失败，程序退出")
            return
        
        try:
            while True:
                self.print_main_menu()
                choice = input("\n请选择功能 (1-7): ").strip()  # 改为1-7
                
                if choice == '1':
                    self.handle_text_navigation()
                elif choice == '2':
                    self.handle_image_navigation()
                elif choice == '3':  # 新增类别导航
                    self.handle_category_navigation()
                elif choice == '4':
                    self.handle_predefined_tasks()
                elif choice == '5':
                    self.view_history()
                elif choice == '6':
                    self.settings_menu()
                elif choice == '7':  # 退出改为7
                    confirm = input("\n确认退出? (y/n) [n]: ").lower().strip()
                    if confirm == 'y':
                        break
                else:
                    print("❌ 无效的选择，请重试")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  检测到中断信号")
        
        finally:
            # 清理资源
            print("\n正在关闭导航系统...")
            if self.agent:
                self.agent.shutdown()
            print("✅ 系统已安全关闭")
            print("\n再见! 👋")


class RobotGrasp:
    """机械臂交互"""
    
    def __init__(self):
        # 连接机器人和夹爪
        self.robot = Robot("192.168.1.10") # 替换为您的机器人IP
        self.gripper = Gripper("192.168.1.10")
        # 设置安全的动力学参数
        self.robot.relative_dynamics_factor = 0.1
        # 恢复错误状态
        self.robot.recover_from_errors()

        self.grasp_force = 15.0 # 抓取力 20N
        self.grasp_speed = 0.01 # 抓取速度 0.01 m/s

    def joint_initialize(self, initial_pos = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7], open = True):
        print("移动到初始位置...")
        initial_position = JointMotion(initial_pos)
        self.robot.move(initial_position)
        if open:
            print("打开夹爪...")
            self.gripper.open(speed=0.02) # 速度 0.02 m/s


    def relative_move(self, x=0,y=0, z=0): # 相对移动
        action = Affine([x, y, z])
        motion = CartesianMotion(action, ReferenceType.Relative)
        self.robot.move(motion)

    def relative_rotation(self, direction='y', x=0, y=0, z=0, angle = 30):
        rotate_angle = Rotation.from_euler(direction, math.radians(angle)).as_quat()
        # Move the robot 20cm along the relative X-axis of its end-effector[x,y,z]
        gripper_rotation = Affine([x, y, z], rotate_angle)
        motion = CartesianMotion(gripper_rotation, ReferenceType.Relative)
        self.robot.move(motion)

    def graspitem(self):
        # 抓取物体
        print("抓取物体...")
        success = self.gripper.grasp(
        width=0.00, # 抓取到接触为止
        speed=self.grasp_speed,
        force=self.grasp_force,
        epsilon_outer=0.1 # 外部误差容限
        )

        if success:
            print("成功抓取物体！")
            # 获取抓取的物体宽度
            grasped_width = self.gripper.width
            print(f"抓取的物体宽度: {grasped_width:.3f} m")
        else:
            print("抓取失败！")
        return success
    
    def open(self):
        self.gripper.open(speed=0.02)
    
    def run(self): # 实际运行
        self.joint_initialize(initial_pos=[-0.0106303, -0.610327, 0.021196, -1.80402, 0.0301998, 1.33754, 0.883195])
        self.relative_move(0.21, 0, 0.40)
        success = self.graspitem()
        if success:
            self.relative_move(-0.21, 0, -0.41)
            # self.relative_move(0, 0, 0.1)
            # self.open()

    def place(self):
        # 恢复错误状态
        self.robot.recover_from_errors()
        self.relative_move(0.15, 0, 0)
        # self.relative_move(0, 0, 0.01)
        self.open()
        self.joint_initialize(initial_pos=[-0.0106303, -0.610327, 0.021196, -1.80402, 0.0301998, 1.33754, 0.883195])
        print("任务结束！")
    
    def clean(self):
        self.robot.recover_from_errors()
        self.joint_initialize(initial_pos=[-0.0106303, -0.610327, 0.021196, -1.80402, 0.0301998, 1.33754, 0.883195])
        self.relative_move(0.25, 0, 0.485)
        success = self.graspitem()
        for i in range(5):
            self.relative_move(0, 0.25, 0.0)
            self.relative_move(0, -0.25, 0.0)
        self.open()
        self.joint_initialize(initial_pos=[-0.0106303, -0.610327, 0.021196, -1.80402, 0.0301998, 1.33754, 0.883195])
        print("任务结束！")

    def demo3move2(self, pos1=0.26, pos2=0.2, pos3=-0.1): # # [z, y, x]，z是负为下，y是正为前
        self.robot.recover_from_errors()
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419])
        self.relative_move(pos1, -0.05, pos3)
        self.relative_move(0, pos2, 0)
        self.relative_move(0, 0, -2*pos3)
        self.grasp_force = 40
        success = self.graspitem()
        self.relative_move(0, -pos2, pos3)
        self.relative_move(-pos1, 0, 0)
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419], open=False)
        
    
    def demo3pour2(self, pos1=0.08, pos2=0.33, pos3=0.07, angle=-75):
        self.robot.recover_from_errors()
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419],open=False)
        self.grasp_force = 40
        # self.graspitem()
        self.relative_move(pos1, pos2, pos3)
        self.relative_rotation('y', 0, 0, 0, angle)
        time.sleep(0.5)
        self.relative_rotation('y', 0, 0, 0, -angle)
        self.relative_move(-0.05, 0.05, -0.05)
        self.relative_move(-pos1, 0, -0.05)
        self.open()
        self.relative_move(0, 0, -0.03)
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419])
    
    def demo3move3(self, pos1=-0.33, pos2=0.22, pos3=-0.04): # # [z, y, x]，z是负为下，y是正为前
        self.robot.recover_from_errors()
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419])
        self.relative_move(pos1, 0, pos3)
        self.relative_move(0, pos2, 0)
        self.relative_move(0, 0, -2*pos3)
        self.grasp_force = 45
        success = self.graspitem()
        # self.relative_move(0, 0, pos3)
        # self.relative_move(-pos1, -pos2, 0)
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419], open=False)
        
    def demo3pour3(self, pos1=-0.08, pos2=0.33, pos3=0.15, angle=-70):
        self.robot.recover_from_errors()
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419],open=False)
        # self.grasp_force = 35
        # self.graspitem()
        
        self.relative_move(pos1, pos2, pos3)
        self.relative_rotation('y', 0.05, 0, 0, angle)
        time.sleep(0.5)
        self.relative_rotation('y', 0, 0, 0, -angle)
        # self.relative_move(0, 0, -0.12)
        # self.relative_move(-0.08, 0, 0)
        # self.relative_move(0, -0.1, 0)
        # self.open()
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419],open=False)
        
    def demo3complex(self):
        self.robot.recover_from_errors()
        self.open()
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419])
        self.relative_rotation('y', 0, 0, 0, -90)
        self.relative_move(0, 0.33, 0)
        self.relative_move(0, 0, 0.04)
        self.grasp_force = 40
        success = self.graspitem()
        self.relative_move(0.12, 0, -0.1)
        self.relative_rotation('y', 0, 0, 0, -115)
        self.relative_rotation('y', 0, 0, 0, 115)
        self.relative_move(-0.12, 0.05, 0.04)
        self.open()
        self.relative_move(0, 0, -0.1)
        self.joint_initialize(initial_pos=[-0.0787125, -0.173616, -0.651959, -2.5732, 0.966888, 1.18251, -1.70419],open=False)



def main():
    """主函数入口"""
    # 可以通过命令行参数或配置文件覆盖默认配置
    config = NAVIGATION_CONFIG.copy()
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        # 例如: python main.py --ws_url ws://192.168.1.100:9090
        import argparse
        parser = argparse.ArgumentParser(description='Visual Navigation System')
        parser.add_argument('--ws_url', type=str, help='WebSocket URL')
        parser.add_argument('--http_url', type=str, help='HTTP API URL')
        parser.add_argument('--map_name', type=str, help='Map name')
        parser.add_argument('--memory_path', type=str, help='Memory path')
        parser.add_argument('--camera_index', type=int, help='Camera device index')
        
        args = parser.parse_args()
        
        if args.ws_url:
            config['ws_url'] = args.ws_url
        if args.http_url:
            config['http_url'] = args.http_url
        if args.map_name:
            config['map_name'] = args.map_name
        if args.memory_path:
            config['memory_path'] = args.memory_path
        if args.camera_index is not None:
            config['camera_device_index'] = args.camera_index
    
    # # 机械臂创建并运行
    robot = RobotGrasp()
    robot.open()
    # robot.demo3move3()
    # robot.demo3pour2()  
    
    # 创建并运行界面
    interface = NavigationInterface(config)
    interface.run()


if __name__ == "__main__":
    main()