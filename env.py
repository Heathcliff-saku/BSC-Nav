import os
import habitat_sim
from typing import Dict, List, Tuple, Union
import numpy as np
import magnum as mn
import time
import random
import imageio
from tqdm import tqdm
from args import get_args
from utils import show_obs, keyboard_control_fast
import cv2
import habitat
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.config.default_structured_configs import LookUpActionConfig,LookDownActionConfig,NumStepsMeasurementConfig
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height

class NavEnv():
    def __init__(self, args, init_state=None, build_map=False):
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"
        
        self.args = args
        if args.dataset == 'hm3d':
            self.scene_dir = os.path.join(args.dataset_dir, args.scene_name, args.scene_name.split('-')[1] + ".basis.glb")
        else:
            self.scene_dir = os.path.join(args.dataset_dir, args.scene_name, args.scene_name + ".glb")
        print(f"Loding scene {args.scene_name}\n")
        self.cfg = self.make_cfg()
        
        self.sims = habitat_sim.Simulator(self.cfg)
        self.agent = self.sims.initialize_agent(0)
        agent_state = habitat_sim.AgentState()

        # 🐶 建图模式下 需要保证agent_state的rotation为[0，0，0，0]以确保建图时方向正确，因此，仅设定position
        if init_state:
            agent_state.position = init_state.position
            if not build_map:
                agent_state.rotation = init_state.rotation
        else:
            random_pt = self.sims.pathfinder.get_random_navigable_point()
            random_pt = self.sims.pathfinder.get_random_navigable_point()
            random_pt = self.sims.pathfinder.get_random_navigable_point()
            random_pt = self.sims.pathfinder.get_random_navigable_point()
            # random_pt = self.sims.pathfinder.get_random_navigable_point()
            agent_state.position = random_pt

        self.agent.set_state(agent_state)
        agent_state = self.agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation, "\n")
        
        self.original_state = agent_state
        
        self.plnner = habitat_sim.nav.GreedyGeodesicFollower(pathfinder=self.sims.pathfinder, agent=self.agent, goal_radius=0.3, stop_key="stop")
        # self.add_object()
        print("island:", self.plnner.pathfinder.get_island(agent_state.position))

    def reset(self, args, init_state=None, build_map=False):
        self.args = args

        if args.dataset == 'hm3d':
            self.scene_dir = os.path.join(self.args.dataset_dir, self.args.scene_name, self.args.scene_name.split('-')[1] + ".basis.glb")
        else:
            self.scene_dir = os.path.join(self.args.dataset_dir, self.args.scene_name, self.args.scene_name + ".glb")
        print(f"Loding scene {self.args.scene_name}\n")

        self.cfg = self.make_cfg()
        
        self.sims.reconfigure(self.cfg)
        self.agent = self.sims.initialize_agent(0)
        agent_state = habitat_sim.AgentState()

        if init_state:
            agent_state.position = init_state.position
            if not build_map:
                agent_state.rotation = init_state.rotation
        else:
            random_pt = self.sims.pathfinder.get_random_navigable_point()
            agent_state.position = random_pt

        self.agent.set_state(agent_state)

        agent_state = self.agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation, "\n")

        self.original_state = agent_state
        self.plnner = habitat_sim.nav.GreedyGeodesicFollower(pathfinder=self.sims.pathfinder, agent=self.agent, goal_radius=0.3, stop_key="stop")

    # def add_object(self):
    #     prim_templates_mgr = self.sim.get_asset_template_manager()
    #     obj_templates_mgr = self.sim.get_object_template_manager()
    #     rigid_obj_mgr = self.sim.get_rigid_object_manager()

    #     sphere_template_id = obj_templates_mgr.load_configs(str('/home/orbit-new/桌面/Nav-2025/3D-object-asset/computer01.object_config.json'))[0]
    #     sphere_obj = rigid_obj_mgr.add_object_by_template_id(sphere_template_id)
    #     # move sphere
    #     sphere_obj.translation = [0.01703353,  1.12711, -0.5]
        
    
    def get_navigable_point_near(self, circle_center, max_tries=500):
        
        # goal_x, goal_z, goal_y = circle_center[0], circle_center[1], circle_center[2]
        # goal = circle_center
        # radius=0.1
        
        # island_goal = self.plnner.pathfinder.get_island(goal)
        # island_begin = self.plnner.pathfinder.get_island(self.agent.get_state().position)

        # while (not self.plnner.pathfinder.is_navigable(goal)) or (island_goal != island_begin):
        #     for _ in range(max_tries):
        #         new_x = random.uniform(goal_x - radius, goal_x + radius)
        #         new_y = random.uniform(goal_y - radius, goal_y + radius)
        #         # distance = np.sqrt((new_x - goal_x) ** 2 + (new_y - goal_y) ** 2)
        #         goal = np.array([new_x, goal_z, new_y])
        #         island_goal = self.plnner.pathfinder.get_island(goal)

        #     radius += 0.1

        island_begin = self.plnner.pathfinder.get_island(self.agent.get_state().position)
        goal = self.plnner.pathfinder.snap_point(circle_center, island_index=island_begin)
        goal = np.array([goal[0], goal[1], goal[2]])
        return goal
     
    def move2point(self, goal):
        
        if not self.plnner.pathfinder.is_navigable(goal):
            goal = self.get_navigable_point_near(circle_center=goal, max_tries=1000)

        path = self.plnner.find_path(goal)
        print("ready move to goal_pos:", goal)
        
        return path, goal
                
        
    def make_cfg(self) -> habitat_sim.Configuration:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_id = self.scene_dir
        sim_cfg.enable_physics = False
        sim_cfg.scene_dataset_config_file = self.args.scene_dataset_config_file
        
        sensor_spec = []
        back_rgb_sensor_spec = self.make_sensor_spec(
            "back_rgb",
            habitat_sim.SensorType.COLOR,
            self.args.height,
            self.args.width,
            [0.0, self.args.sensor_height, 0.0],
            orientation=mn.Vector3(-np.pi / 8, 0, 0),
        )
        sensor_spec.append(back_rgb_sensor_spec)

        if self.args.color_sensor:
            rgb_sensor_spec = self.make_sensor_spec(
                "rgb",
                habitat_sim.SensorType.COLOR,
                self.args.height,
                self.args.width,
                [0.0, self.args.sensor_height, 0.0],
            )
            sensor_spec.append(rgb_sensor_spec)

        if self.args.depth_sensor:
            depth_sensor_spec = self.make_sensor_spec(
                "depth",
                habitat_sim.SensorType.DEPTH,
                self.args.height,
                self.args.width,
                [0.0, self.args.sensor_height, 0.0],
            )
            sensor_spec.append(depth_sensor_spec)

        if self.args.semantic_sensor:
            semantic_sensor_spec = self.make_sensor_spec(
                "semantic",
                habitat_sim.SensorType.SEMANTIC,
                self.args.height,
                self.args.width,
                [0.0, self.args.sensor_height, 0.0],
            )
            sensor_spec.append(semantic_sensor_spec)

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_spec
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward",
                habitat_sim.agent.ActuationSpec(amount=self.args.move_forward),
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=self.args.turn_right)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=self.args.turn_right)
            ),
        }

        return habitat_sim.Configuration(sim_cfg, [agent_cfg])


    def make_sensor_spec(
        self,
        uuid: str,
        sensor_type: str,
        h: int,
        w: int,
        position: Union[List, np.ndarray],
        orientation: Union[List, np.ndarray] = None,
    ) -> Dict:
        sensor_spec = habitat_sim.CameraSensorSpec()
        sensor_spec.uuid = uuid
        sensor_spec.sensor_type = sensor_type
        sensor_spec.resolution = [h, w]
        sensor_spec.position = position
        if orientation:
            sensor_spec.orientation = orientation

        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        return sensor_spec
    
    
    def move2object():
        pass  
        
    
    def keyboard_explore(self):
        obs = self.sims.get_sensor_observations(0)
        last_action = None
        release_count = 0
        
        while True:
            show_obs(obs)
            k, action = keyboard_control_fast()
            if k != -1:
                if action == "stop":
                    break
                if action == "record":
                    init_agent_state = self.sims.get_agent(0).get_state()
                    actions_list = []
                    continue
                last_action = action
                release_count = 0
            else:
                if last_action is None:
                    time.sleep(0.01)
                    continue
                else:
                    release_count += 1
                    if release_count > 1:
                        # print("stop after release")
                        last_action = None
                        release_count = 0
                        continue
                    action = last_action

            obs = self.sims.step(action)
            agent_state = self.agent.get_state()
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
            print('island:', self.plnner.pathfinder.get_island(agent_state.position))


def get_objnav_env(args):
    if args.benchmark_dataset == 'hm3d':
        config =  hm3d_data_config(args)
    else:
        config = mp3d_data_config(args)

    sim = habitat.Env(config)
    return sim

def hm3d_data_config(args, stage:str='val',
                    episodes=100):

    habitat_config = habitat.get_config(args.HM3D_CONFIG_PATH)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = args.HM3D_SCENE_PREFIX
        habitat_config.habitat.dataset.data_path = args.HM3D_EPISODE_PREFIX
        habitat_config.habitat.simulator.scene_dataset = args.HM3D_SCENE_PREFIX + "/hm3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        # habitat_config.habitat.environment.iterator_options.shuffle = False
        habitat_config.habitat.environment.max_episode_steps = args.max_episode_steps
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=False,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = args.image_hfov
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = args.image_hfov

        habitat_config.habitat.simulator.agents.main_agent.height=1.5
        habitat_config.habitat.simulator.agents.main_agent.radius=0.1

        habitat_config.habitat.simulator.forward_step_size = args.move_forward
        habitat_config.habitat.simulator.turn_angle = args.turn_left

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = args.height 
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = args.width
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = args.height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = args.width

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0,args.sensor_height,0]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0,args.sensor_height,0]

        habitat_config.habitat.simulator.habitat_sim_v0.allow_sliding = True
        habitat_config.habitat.task.measurements.success.success_distance = args.success_distance

    return habitat_config


def mp3d_data_config(args,stage:str='val',
                    episodes=200):

    habitat_config = habitat.get_config(args.HM3D_CONFIG_PATH)

    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = args.MP3D_SCENE_PREFIX 
        habitat_config.habitat.dataset.data_path = args.MP3D_EPISODE_PREFIX
        habitat_config.habitat.simulator.scene_dataset = args.MP3D_SCENE_PREFIX + "/mp3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        # habitat_config.habitat.environment.iterator_options.shuffle = False
        habitat_config.habitat.environment.max_episode_steps = args.max_episode_steps
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=False,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = args.image_hfov
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = args.image_hfov

        habitat_config.habitat.simulator.agents.main_agent.height=1.5
        habitat_config.habitat.simulator.agents.main_agent.radius=0.1

        habitat_config.habitat.simulator.forward_step_size = args.move_forward
        habitat_config.habitat.simulator.turn_angle = args.turn_left

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = args.height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = args.width   
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = args.height
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = args.width 

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0,args.sensor_height,0]
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0,args.sensor_height,0]

        habitat_config.habitat.simulator.habitat_sim_v0.allow_sliding = True
        habitat_config.habitat.task.measurements.success.success_distance = args.success_distance
    return habitat_config



def adjust_topdown(metrics):
    return cv2.cvtColor(colorize_draw_agent_and_fit_to_height(metrics['top_down_map'],1024),cv2.COLOR_BGR2RGB)

def keyboard_control():
    k = cv2.waitKey(0)
    if k == ord("w"):
        action = "move_forward"
    elif k == ord("a"):
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("4"):
        action = "look_up"
    elif k == ord("5"):
        action = "look_down"
    elif k == ord("q"):
        action = 'stop'
    elif k == ord(" "):
        return k, "record"
    elif k == -1:
        return k, None
    else:
        return -1, None
    return k, action

def show_obs(obs):
    # 获取 RGB 图像并转换为 BGR 格式
    bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
    cv2.imshow("RGB", bgr)

if __name__ == "__main__":
    
    args = get_args()

    # env = NavBenchmarkEnv(args)
    # habitat_env = env.sims
    # evaluation_metrics = []
    
    # for i in tqdm(range(args.eval_episodes)):
    #     obs = habitat_env.reset()
    #     dir = "./tmp/trajectory_%d"%i
    #     os.makedirs(dir, exist_ok=True)
    #     fps_writer = imageio.get_writer("%s/fps.mp4"%dir, fps=4)
    #     topdown_writer = imageio.get_writer("%s/metric.mp4"%dir,fps=4)
    #     episode_images = [obs['rgb']]
    #     episode_topdowns = [adjust_topdown(habitat_env.get_metrics())]
        
    #     print(habitat_env.current_episode.object_category)


    #     while not habitat_env.episode_over:
    #         show_obs(obs)
    #         k, action = keyboard_control()
    #         print(action)
    #         obs = habitat_env.step(action)
    #         agent_state = env.agent.get_state()
    #         print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

    #         episode_images.append(obs['rgb'])
    #         episode_topdowns.append(adjust_topdown(habitat_env.get_metrics()))
    #         # print(habitat_env.get_metrics())
        
    #     for image,topdown in zip(episode_images,episode_topdowns):
    #         fps_writer.append_data(image)
    #         topdown_writer.append_data(topdown)
    #     fps_writer.close()
    #     topdown_writer.close()
    #     evaluation_metrics.append({'success':habitat_env.get_metrics()['success'],
    #                            'spl':habitat_env.get_metrics()['spl'],
    #                            'distance_to_goal':habitat_env.get_metrics()['distance_to_goal'],
    #                            'object_goal':habitat_env.current_episode.object_category})
        
    # print(evaluation_metrics)
        

    env = NavEnv(args)
    
    env.keyboard_explore()
    # env.move2point(goal=np.array([1.16672724,  3.2034254, -0.4141059]))