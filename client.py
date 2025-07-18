#!/usr/bin/env python
import asyncio
from websocket  import create_connection
import websockets
import time
import requests
import json
import numpy as np
import math
from threading import Thread, Timer
import base64
import array
import cv2

# ws_url = "ws://101.33.245.108:9090" 
# http_url = "http://101.33.245.108:9358/apiUrl"
ws_url = "ws://192.168.1.102:9090" #填写实际的机器人IP地址
http_url = "http://192.168.1.102/apiUrl"
token = None

def map_coordinate_to_png(pos, map_info):
    '''
        坐标转换函数:将导航时实际使用坐标转换成地图png图片坐标。
        输入参数:地图导航时的实际使用坐标, 地图png绑定的yaml信息数据,包含原点,宽度高度,分辨率。
        输出参数:转成png坐标(x,y)
    ''' 
    map_pos_x = (pos[0] - map_info['originX']) / map_info['resolution']
    map_pos_y = map_info['gridHeight'] -  (pos[1] - map_info['originY']) / map_info['resolution']
    return (map_pos_x, map_pos_y)

def png_coordinate_to_map(pos, map_info):
    """
        坐标转换函数:将地图png图片坐标转换成导航时实际使用坐标。
        输入参数:png坐标(x,y), 地图png绑定的yaml信息数据,包含原点,宽度高度,分辨率。
        输出参数:转成后的实际使用坐标
    """

    png_x = pos[0] * map_info['resolution'] + map_info['originX']

    png_y = (map_info['gridHeight'] - pos[1]) * map_info['resolution'] + map_info['originY'];
    return (png_x, png_y)

def quaternion_from_euler(roll, pitch, yaw):
    """
        角度先转成弧度，再转换成四元素,返回四元素
    """
    roll = roll *  (math.pi / 180)
    pitch = pitch *  (math.pi / 180)
    yaw = yaw *  (math.pi / 180)
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

    return [qx, qy, qz, qw]

def quaternion_to_euler(ori):
    """
    四元素转成弧度，再转换成角度,返回yaw 角度
    """
    roll = math.atan2(2 * (ori.w * ori.x + ori.y * ori.z), 1 - 2 * (ori.x * ori.x + ori.y * ori.y))
    pitch = math.asin(2 * (ori.w * ori.y - ori.x * ori.z))
    yaw = math.atan2(2 * (ori.w * ori.z + ori.x * ori.y), 1 - 2 * (ori.z * ori.z + ori.y * ori.y))
    math.degrees(yaw)
    return yaw

class WSClient:
    def __init__(self, address, id_):
        self.isconnect = False
        self.ws = create_connection(address)
        if self.ws.connected:
            self.isconnect = True

        self.input_data = {
            "op": "call_service",
            "service": "/input/op",
            "type":"tools_msgs/input",
            "args": {
                "file_name": "",
                "op_type":"",
                "id_type":""
            }
        }
        self.heart_msg = {
            "op": "ping",
            "timeStamp": 0,
            "id":id_
        }

    def send_msg(self, args):
        if self.ws is not None:
            msg = json.dumps(args, ensure_ascii=False).encode("utf-8")
            self.ws.send(msg)

    def receive(self):
        """
        The data returned by the websocket server is in this function 
        """
        try:
            while(True):
                print('receive_message = ',  self.ws.recv())
        except KeyboardInterrupt:
            print("KeyboardInterrupt")

    def get_bytes_data(self, args):
        if self.ws is not None:
            msg = json.dumps(args, ensure_ascii=False).encode("utf-8")
            self.ws.send(msg)
            # Used for a single call to an interface test to return bytes data
            return self.ws.recv()
        
    def call_service_and_get_response(self, args):
        """
        调用服务并等待响应
        
        Args:
            args: 服务调用参数字典
            
        Returns:
            dict: 服务响应数据
        """
        if self.ws is not None:
            msg = json.dumps(args, ensure_ascii=False).encode("utf-8")
            self.ws.send(msg)
            
            # 等待服务响应
            try:
                response = self.ws.recv()
                return json.loads(response)
            except Exception as e:
                print(f"Error getting service response: {e}")
                return None

    def publish_data(self, args):
        if self.ws is not None:
            msg = json.dumps(args, ensure_ascii=False).encode("utf-8")
            self.ws.send(msg)
        
    def heart_beat(self):
        if self.isconnect == True:
            self.heart_msg['timeStamp'] = str(time.time()*1000).split(".")[0]
            self.send_msg(self.heart_msg)
            
    def on_close(self):
        if self.ws is not None and self.isconnect:
            self.ws.close()
            self.isconnect = False

    def call_input(self, op_type, id_type, file_name = ''):
        self.input_data['args']['file_name'] = file_name
        self.input_data['args']['op_type'] = id_type
        self.input_data['args']['id_type'] = op_type

        return self.send_msg(self.input_data)
    
    def record_bag(self, idtype, filename = ''):
        self.call_input(op_type = 'record_data',id_type=idtype, file_name=filename)

    def mapping_3d(self, idtype, filename = ''):
        res = self.call_input(op_type = 'map_3d',id_type=idtype, file_name=filename)
    
    def mapping_2d(self, idtype, filename = ''):
        res = self.call_input(op_type = 'map_2d',id_type=idtype, file_name=filename)

    def follow_line(self, idtype, filename = ''):
        res = self.call_input(op_type = 'follow_line',id_type=idtype, file_name=filename)

    def initial_pos(self, pos:list, angle):
        qua = quaternion_from_euler(0, 0, angle * ( math.pi / 180)  * -1)

        msg = {
            "op": "publish",
            "topic": "/initialpose",
            "type": "geometry_msgs/PoseWithCovarianceStamped",
            "msg":{
                "header": {"frame_id":"map_2d"},
                "pose":{
                    "pose": {
                        "position": {
                            "x": pos[0],
                            "y": pos[1],
                            "z": 0
                        },
                        "orientation": {
                            "w": qua[3],
                            "z": qua[2],
                            "y": qua[1],
                            "x": qua[0]
                        }
                    },
                    "covariance": [
                        0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,0,0,0,0
                    ]
                }
            }
        }
        self.publish_data(msg)

    def sub_slam_status(self):
        msg = {   
            "op": "subscribe",
            "topic": "/slam_status"
            # "type":"nav_msgs/Odometry" 
        }
    
        self.send_msg(msg)

    def sub_robot_status(self):
        msg = {   
            "op": "subscribe",
            "topic": "/dash_board/robot_status"
        }
        self.send_msg(msg)

    def cancel_nav(self):
        msg = {
                "op": "publish",
                "topic": "/run_management/navi_task/cancel",
                "type": "actionlib_msgs/GoalID",
                "msg":{
                    "stamp": {
                        "secs": 0,
                        "nsecs": 0
                    },
                "id": ""
            }
        }
        self.publish_data(msg)

    def start_heartbeat_timer(self, interval):
        
        self.timer = threading.Timer(interval, self.start_heartbeat_timer, (interval,))
        self.timer.start()

        self.heart_beat()

    def stop_heartbeat_timer(self):
        self.timer.cancel()

    def sub_pointCloud2(self):
        msg = {   
            "op": "subscribe",
            "topic": "/points_raw"
            #"compression": "cbor" #如果带有这条字段,获取到的是bytes，
        }
        
        # compression 字段使用下面接口获取数据
        # res = self.get_bytes_data(msg)
        # data_array = array.array('B',res)
        # print(data_array)


        # 无compression 字段使用默认的发送数据接口
        res = self.send_msg(msg)
        data= res.get("msg").get("data")   
        decode_data = base64.b64decode(data, altchars=None)
        data_array = array.array('B',decode_data)
        # print(data_array)


    def sub_scan(self):
        msg = {   
            "op": "subscribe",
            "topic": "/scan",
            "compression": "cbor" #如果带有这条字段,获取到的是bytes，

        }
    
        # 有compression 字段使用下面接口获取数据
        res = self.get_bytes_data(msg)
        data_array = array.array('B',res)
        print(data_array)
        
        # 无compression 字段使用默认的发送数据接口
        # res = self.send_msg(msg)
        # data= res.get("msg").get("ranges")
        # print(data)    

    def sub_camera_pointCloud(self):
        msg = {   
            "op": "subscribe",
            "topic": "/camera/color/image_raw"
            # "compression": "cbor" 
        }
    
        # compression 字段使用下面接口获取数据
        message = self.send_msg(msg)
        # data_array = array.array('B',message)
        
        image_data = message["msg"]['data']
        
        decode_data = base64.b64decode(image_data, altchars=None)
        img_array = array.array('B',decode_data)
        # img_array = bytearray('B', image_data)
        img_array = np.frombuffer(img_array, np.uint8)
    
        # print(img_array)
        image_ = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        image_name = R"E:\workspace\userapi\test\te.jpg"
        cv2.imwrite(image_name, image_)

        # # Build the image format and dimensions
        # format = message['encoding']
        # width = message['width']
        # height = message['height']

        # # Build the PIL Image object
        # # image = Image.frombytes(format, (width, height), base64.b64decode(image_data))
        # image = Image.frombytes(format, (width, height), image_data)

        # # Save the image as a picture file
    
        # image.save(image_name)
        
    def sub_task_status(self):
        msg = {   
            "op": "subscribe",
            "topic": "/run_management/task_status"
        }
        self.send_msg(msg)
    
    def move_robot(self, linear_x=0.0, linear_y=0.0, angular_z=0.0, use_nav_topic=True):
        """
        控制机器人移动（旧API，保留兼容性）
        
        Args:
            linear_x: 前进/后退速度 (+前进, -后退) 范围: +/-1.5 m/s
            linear_y: 横向移动速度 (仅适用于某些车型)
            angular_z: 旋转速度 (+左转, -右转) 范围: +/-1.5 rad/s
            use_nav_topic: True使用/cmd_vel (仅导航时有效), False使用/run_management/virtual_joy (全局有效)
        """
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
        
        self.publish_data(msg)
    
    def chassis_control(self, distance=0.0, rotation_angle=0.0, motion_type=0, 
                       linear_x=0.0, angular_z=0.0, stop=False):
        """
        使用新的底盘控制服务API（同步调用，运动完成后才返回）
        
        Args:
            distance: 机器人竖直方向的距离，单位：m
            rotation_angle: 机器人旋转的角度，单位：度
            motion_type: 机器人控制模式(0-直线运动模式, 1-旋转运动模式)
            linear_x: 仅在竖直方向移动时有效
            angular_z: 仅在旋转时有效
            stop: 是否停止控制，默认为false
            
        Returns:
            dict: 服务响应数据（运动完成后返回）
        """
        msg = {
            "op": "call_service",
            "service": "/chassis_control_service",
            "type": "bt_task_msgs/ChassisCommand",
            "args": {
                "distance": distance,
                "rotation_angle": rotation_angle,
                "motion_type": motion_type,
                "linear_x": linear_x,
                "angular_z": angular_z,
                "stop": stop
            }
        }
        
        return self.call_service_and_get_response(msg)
    
    def stop_robot(self):
        """停止机器人移动"""
        self.move_robot(0.0, 0.0, 0.0)


# 添加线程模块导入
import threading

class HttpClient():
    def __init__(self, url):
        self.url = url
        self.token = None
        self.map_list = None

    def login_(self):
        url = self.url + '/user/passport/login'

        payload = json.dumps({
            "username": "agilex",
            "password": "NTdhMTE5NGNiMDczY2U4YjNiYjM2NWU0YjgwNWE5YWU="
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        token = json.loads(response.text)
        self.token = token.get('data')

    def get_maplist(self):
        url = self.url + "/map_list?page=1&limit=-1&sortType=&reverse="
        
        headers = {
            'Authorization': self.token
        }

        response = requests.request("GET", url, headers=headers)
        res_json = json.loads(response.text)
        self.map_list = res_json.get('data')
        print('aaa', self.map_list)

    def get_map_png(self, map_name):
        url = self.url + "/downloadpng?mapName=" + map_name
        
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.request("GET", url, headers=headers)
        
        filepath = (f'./map_source/{map_name}.png')
        with open(filepath, 'wb') as f:
            f.write(response.content)

    def get_map_info(self, map_name):
        url = self.url + "/map_info?mapName=" + map_name #mapName是单个地图名称，不用包含.png
        
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        response = requests.request("GET", url, headers=headers)
        
        res_json = json.loads(response.text)
        if res_json.get('data'):
            return res_json.get('data')['mapInfo']
        else:
            return None

    def run_realtime_task(self, pos:list):

        payload = json.dumps(
        {
            "loopTime": 1, 
            "points": [
            { 
                "position": {
                    "x": pos[0],
                    "y": pos[1],
                    "theta": pos[2]
                },
                "isNew": False,
                "cpx": 0,
                "cpy": 0
            }],
            "mode": "point"# mode = path 时需要设置多个点,进行路径导航
        })
        
        url = self.url + "/realtime_task"
        
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers,data=payload)
        res_json = json.loads(response.text)

        if res_json.get('code') == 0 or res_json.get('successed')== True:
            print('success~~~')

    def run_list_task(self, map_name, task_name, looptime: int):
        payload = {
            "mapName": map_name, 
            "taskName": task_name,
            "loopTime":looptime
        }
        
        url = self.url + "/run_list_task"
        
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    
        res_json = json.loads(response.text)
        if res_json.get('success') == True:
            print('success~~~')

    def set_list_task(self):
        payload = {
            "mode": "point",
            "speed": 1,
            "evadible": 1,
            "points": [
                {
                "isNew": True,
                "index": "point-1",
                "pointType": "navigation",
                "pointName": "",
                "actions": [],
                "position": {
                    "x": -3.35,
                    "y": 37.48,
                    "theta": 78.24   
                },
                "cpx": 0,
                "cpy": 0
                },
                {
                "isNew": True,
                "index": "point-2",
                "pointType": "navigation",
                "pointName": "",
                "actions": [],
                "position": {     
                    "x": -2.45,
                    "y": 34.53,
                    "theta": 169.32
                },
                "cpx": 0,
                "cpy": 0
                }
            ],
            "mapName": "office",
            "taskName": "p1",
            "editedName": "p1",
            "gridItemIdx": 0,
            "remark": "",
            "personName": ""
        }

        url = self.url + "/set_task"
        
        headers = {
            'Authorization': self.token,
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

        res_json = json.loads(response.text)
        if res_json.get('code') == 0 or res_json.get('successed')== True:
            print('set task success~~~')


# 添加 RobotNavigationController 类以支持数据采集器
class RobotNavigationController:
    """机器人导航控制器 - 用于数据采集"""
    
    def __init__(self, ws_url, http_url, map_name):
        self.ws_url = ws_url
        self.http_url = http_url
        self.map_name = map_name
        
        # 初始化客户端
        self.http_client = HttpClient(http_url)
        self.ws_client = None
        
        # 机器人状态
        self.robot_state = None
        self.is_navigating = False
        
    def __enter__(self):
        """进入上下文管理器"""
        # 登录HTTP客户端
        self.http_client.login_()
        
        # 创建WebSocket客户端
        self.ws_client = WSClient(self.ws_url, "nav_controller")
        
        # 启动心跳
        self.ws_client.start_heartbeat_timer(2)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        if self.ws_client:
            # 停止导航
            if self.is_navigating:
                self.stop_navigation()
            
            # 停止心跳
            self.ws_client.stop_heartbeat_timer()
            
            # 关闭连接
            self.ws_client.on_close()
    
    def start_navigation(self, initial_pose=None):
        """
        启动导航
        
        Args:
            initial_pose: [x, y, theta] 初始位置和角度
        """
        # 启动导航
        self.ws_client.follow_line(idtype="start", filename=self.map_name)
        time.sleep(2)
        
        # 设置初始位置
        if initial_pose:
            self.ws_client.initial_pos(initial_pose[:2], initial_pose[2] if len(initial_pose) > 2 else 0)
            time.sleep(2)
        
        self.is_navigating = True
        return True
    
    def stop_navigation(self):
        """停止导航"""
        self.ws_client.follow_line(idtype="stop", filename=self.map_name)
        self.is_navigating = False
    
    def move_robot(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        """
        移动机器人（使用旧API）
        
        Args:
            linear_x: 前进/后退速度
            linear_y: 横向速度
            angular_z: 旋转速度
        """
        if self.ws_client:
            self.ws_client.move_robot(linear_x, linear_y, angular_z, use_nav_topic=True)
    
    def chassis_control(self, distance=0.0, rotation_angle=0.0, motion_type=0, 
                       linear_x=0.0, angular_z=0.0, stop=False):
        """
        使用新的底盘控制API
        
        Args:
            distance: 机器人竖直方向的距离，单位：m
            rotation_angle: 机器人旋转的角度，单位：度
            motion_type: 机器人控制模式(0-直线运动模式, 1-旋转运动模式)
            linear_x: 仅在竖直方向移动时有效
            angular_z: 仅在旋转时有效
            stop: 是否停止控制
            
        Returns:
            dict: 服务响应
        """
        if self.ws_client:
            return self.ws_client.chassis_control(
                distance=distance,
                rotation_angle=rotation_angle,
                motion_type=motion_type,
                linear_x=linear_x,
                angular_z=angular_z,
                stop=stop
            )
    
    def stop_robot(self):
        """停止机器人"""
        if self.ws_client:
            self.ws_client.stop_robot()
    
    def get_state(self):
        """
        获取机器人当前状态（模拟）
        注意：实际实现需要订阅相应话题并解析状态
        """
        # 这里需要实际订阅和解析机器人状态
        # 为了演示，返回模拟数据
        return {
            'pose': {
                'position': {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0
                },
                'orientation': {
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0,
                    'w': 1.0
                }
            },
            'last_update': time.time()
        }


if __name__ == '__main__':
    ### Http 客户端
    http_client =  HttpClient(http_url)
    http_client.login_()

    # ### websocket 客户端
    ws_client =  WSClient(ws_url, "aa-sss-1")
    # ## 定时发送心跳包
    # ws_client.start_heartbeat_timer(2)

    ###  订阅设备当前导航状态
    # ws_client.sub_task_status()
    # 如果单开线程来 跑接收数据函数，可能会导致无法退出程序
    # heartbeat_thread = threading.Thread(target= ws_client.receive)
    # heartbeat_thread.start()
    # ws_client.sub_robot_status()
    # ws_client.sub_slam_status()
    # 注意这里会阻塞,可以放到最后
    


    ###  订阅激光雷达3D点云数据
    # ws_client.sub_pointCloud2()

    ###  订阅激光雷达2D点云数据,只有在开启导航的情况下才会订阅得到数据
    # ws_client.sub_scan()

    ##  订阅相机数据
    # data1 =  ws_client.sub_camera_pointCloud()

    # save_image(data1)
    
    map_name = 'dongsheng04'

    '''
    ### 录包->建图->启动导航->流程
    
    ### 启动录包,间隔10秒结束录包,可以根据实际需求设置实际
    time_1 = 10
    time_2 = 5
    ws_client.record_bag(idtype="start", filename=map_name)
    time.sleep(time_1)

    ws_client.record_bag(idtype="stop", filename=map_name)
    time.sleep(time_2)

    ### 启动3d建图,间隔10秒结束3d建图
    ws_client.mapping_3d(idtype="start", filename=map_name)
    time.sleep(time_1)

    ws_client.mapping_3d(idtype="stop", filename=map_name)
    time.sleep(time_2)

    ### 启动2d建图,间隔10秒结束2d建图
    ws_client.mapping_2d(idtype="start", filename=map_name)
    time.sleep(time_1)

    ws_client.mapping_2d(idtype="stop", filename=map_name)
    time.sleep(time_2)

    ### 启动导航
    
    '''
    
    ###获取地图列表
    http_client.get_maplist()

    ###获取某张地图
    http_client.get_map_png(map_name),#下载到本地,路径需修改,

    ### 获取地图对应的信息
    map_info = http_client.get_map_info(map_name)
    print(map_info)

    

    ### 使用windows 系统自带的画图软件打开下载好的地图png,使用鼠标点击图片白色区域任务点,左下角显示该点对应的坐标
    ### 使用,标转换成地图真实坐标，如下坐标[x,y]
    # png_coor = [174,330]
    # png_coor_init = [383,120]
    # if map_info:
    #     pos_x,pos_y =  png_coordinate_to_map(png_coor, map_info)
    #     pos_x_i,pos_y_i =  png_coordinate_to_map(png_coor_init, map_info)
    #     print("1 ",pos_x, pos_y)

    # ws_client.follow_line(idtype="start", filename=map_name)

    # ## 设置初始化点
    # ws_client.initial_pos([pos_x_i, pos_y_i], 0)

    # ### 设置导航目标点,执行实时任务
    # http_client.run_realtime_task([pos_x, pos_y, 80])

    ### 设置列表任务
    # http_client.set_list_task()

    ### 执行列表任务
    # http_client.run_list_task(map_name=map_name, task_name='p1', looptime=1)

    ### 结束任务
    # ws_client.cancel_nav()

    ### 关闭导航
    # ws_client.follow_line(idtype="stop", filename=map_name)

    # ws_client.receive() 

    # ws_client.on_close()