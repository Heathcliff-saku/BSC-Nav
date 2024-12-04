import os
import habitat_sim
from typing import Dict, List, Tuple, Union
import numpy as np
import magnum as mn
import time
import cv2
import random


from args import get_args
from utils import show_obs, keyboard_control_fast

class NavEnv():
    def __init__(self, args):
        os.environ["MAGNUM_LOG"] = "quiet"
        os.environ["HABITAT_SIM_LOG"] = "quiet"
        
        self.args = args
        self.scene_dir = os.path.join(args.dataset_dir, args.scene_name, args.scene_name + ".glb")
        print(f"Loding scene {args.scene_name}\n")
        self.cfg = self.make_cfg()
        
        self.sim = habitat_sim.Simulator(self.cfg)
        self.agent = self.sim.initialize_agent(0)
        agent_state = habitat_sim.AgentState()
        random_pt = self.sim.pathfinder.get_random_navigable_point()
        random_pt = self.sim.pathfinder.get_random_navigable_point()
        random_pt = self.sim.pathfinder.get_random_navigable_point()
        agent_state.position = random_pt
        self.agent.set_state(agent_state)
        
        agent_state = self.agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation, "\n")
        self.original_state = agent_state
        
        self.plnner = habitat_sim.nav.GreedyGeodesicFollower(pathfinder=self.sim.pathfinder, agent=self.agent, goal_radius=0.3, stop_key="stop")
    
    
    def get_random_navigable_point_near(self, circle_center, radius=0.3, max_tries=1000):
        goal_x, goal_z, goal_y = circle_center[0], circle_center[1], circle_center[2]
        
        for _ in range(max_tries):
            new_x = random.uniform(goal_x - radius, goal_x + radius)
            new_y = random.uniform(goal_y - radius, goal_y + radius)
            distance = np.sqrt((new_x - goal_x) ** 2 + (new_y - goal_y) ** 2)
            
            if distance < radius:
                goal = np.array([new_x, goal_z, new_y])
                if self.plnner.pathfinder.is_navigable(goal):
                    return goal
     
    def move2point(self, goal):
        
        if not self.plnner.pathfinder.is_navigable(goal):
            goal = self.get_random_navigable_point_near(circle_center=goal, radius=0.3, max_tries=1000)
            
        obs = self.sim.get_sensor_observations(0)
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
            "back_color_sensor",
            habitat_sim.SensorType.COLOR,
            self.args.height,
            self.args.width,
            [0.0, self.args.sensor_height, 1.3],
            orientation=mn.Vector3(-np.pi / 8, 0, 0),
        )
        sensor_spec.append(back_rgb_sensor_spec)

        if self.args.color_sensor:
            rgb_sensor_spec = self.make_sensor_spec(
                "color_sensor",
                habitat_sim.SensorType.COLOR,
                self.args.height,
                self.args.width,
                [0.0, self.args.sensor_height, 0.0],
            )
            sensor_spec.append(rgb_sensor_spec)

        if self.args.depth_sensor:
            depth_sensor_spec = self.make_sensor_spec(
                "depth_sensor",
                habitat_sim.SensorType.DEPTH,
                self.args.height,
                self.args.width,
                [0.0, self.args.sensor_height, 0.0],
            )
            sensor_spec.append(depth_sensor_spec)

        if self.args.semantic_sensor:
            semantic_sensor_spec = self.make_sensor_spec(
                "semantic_sensor",
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
        obs = self.sim.get_sensor_observations(0)
        last_action = None
        release_count = 0
        
        while True:
            show_obs(obs)
            k, action = keyboard_control_fast()
            if k != -1:
                if action == "stop":
                    break
                if action == "record":
                    init_agent_state = self.sim.get_agent(0).get_state()
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

            obs = self.sim.step(action)
            agent_state = self.agent.get_state()
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
     
        
            

if __name__ == "__main__":
    
    args = get_args()
    env = NavEnv(args)
    
    # env.keyboard_explore()
    env.move2point(goal=np.array([1.16672724,  3.2034254, -0.4141059]))