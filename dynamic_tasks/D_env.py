
import sys
sys.path.append("/home/orbit/桌面/Nav-2025/")
import habitat_sim
from env import NavEnv, show_obs
from args import get_args
from utils import keyboard_control_fast
import magnum as mn
import json
import time
import random
import os
import numpy as np

class DynamicNavEnv(NavEnv):
    def __init__(self, args, init_state=None, build_map=False, dynamic_env_config=None, mode='build_map'):
        super().__init__(args, init_state, build_map)
        
        if dynamic_env_config is not None:
            with open(dynamic_env_config, 'r') as f:
                self.dynamic_env_config = json.load(f)

        self.base_config_path = '/home/orbit/桌面/Nav-2025/3D-object-asset'

        self.begin_time = time.time()
        self.time_step = 5

        self.prim_templates_mgr = self.sims.get_asset_template_manager()
        self.obj_templates_mgr = self.sims.get_object_template_manager()
        self.rigid_obj_mgr = self.sims.get_rigid_object_manager()
        self.dynamic_obj = {}
        self.mode = mode

        if self.mode == 'test':
            goal_nums = len(self.dynamic_env_config['objects'])
            state_nums = [len(obj['states']) for obj in self.dynamic_env_config['objects']]
            self.task_index = []
            for i in range(goal_nums):
                for j in range(state_nums[i]):
                    self.task_index.append((i, j))

            self.curr_task_id = -1

            # metric
            self.step_num = [0] * len(self.task_index)
            self.success = [0] * len(self.task_index)
            self.dist_to_goal = [0] * len(self.task_index)

            self.reset()

    def reset(self):

        self.curr_task_id += 1
        if self.curr_task_id < len(self.task_index):
            
            agent_state = habitat_sim.AgentState()
            random_pt = self.sims.pathfinder.get_random_navigable_point()
            agent_state.position = random_pt
            self.agent.set_state(agent_state)

            self.text_goal = self.dynamic_env_config['objects'][self.task_index[self.curr_task_id][0]]['description']
            new_state = self.dynamic_env_config['objects'][self.task_index[self.curr_task_id][0]]['states'][self.task_index[self.curr_task_id][1]]
            obj_name = self.dynamic_env_config['objects'][self.task_index[self.curr_task_id][0]]['name']
            config_path = os.path.join(self.base_config_path, f'{obj_name}.object_config.json')

            self.goal_loc = new_state['position']
            self.goal_rot = new_state['rotation']
            self.place_object(obj_name, config_path, self.goal_rot, self.goal_loc)
            print(f"update {obj_name} to new state")
            
        else:
            print("tasks done")


    def get_metric(self):
        agent_loc = self.agent.get_state().position
        self.dist_to_goal[self.curr_task_id] = np.linalg.norm(agent_loc - self.goal_loc)
        if self.dist_to_goal[self.curr_task_id] < self.args.success_distance:
            self.success[self.curr_task_id] = 1
        
        print(f"step_num: {self.step_num[self.curr_task_id]}, success: {bool(self.success[self.curr_task_id])}, dist_to_goal: {self.dist_to_goal[self.curr_task_id]}")


    def get_time(self):
        return time.time() - self.begin_time


    def place_object(self, obj_name, config_path, rotation, position):

        if obj_name in self.dynamic_obj:
            sphere_obj = self.dynamic_obj[obj_name]
        else:
            sphere_template_id = self.obj_templates_mgr.load_configs(str(config_path))[0]
            sphere_obj = self.rigid_obj_mgr.add_object_by_template_id(sphere_template_id)
            self.dynamic_obj[obj_name] = sphere_obj

        sphere_obj.translation = position
        
        rotation_x = mn.Quaternion.rotation(mn.Deg(rotation[0]), [-1.0, 0.0, 0.0])
        rotation_y = mn.Quaternion.rotation(mn.Deg(rotation[1]), [0.0, -1.0, 0.0])
        rotation_z = mn.Quaternion.rotation(mn.Deg(rotation[2]), [0.0, 0.0, -1.0])
        sphere_obj.rotation = rotation_x*rotation_y*rotation_z

    def update_object_state(self):
        
        for dynamic_obj in self.dynamic_env_config['objects']:
            new_state = random.choice(dynamic_obj['states'])

            # Find the object in the environment
            obj_name = dynamic_obj['name']
            config_path = os.path.join(self.base_config_path, f'{obj_name}.object_config.json')
            self.place_object(obj_name, config_path, new_state['rotation'], new_state['position'])
            print(f"update {obj_name} to new state")


    def step_dynamic_env(self, action):
        
        obs = self.sims.step(action)
        self.step_num[self.curr_task_id] += 1
        # if current time is greater than time_step, then update the object state chosen from dynamic_env_config
        if self.mode == 'build_map':
            if self.get_time() > self.time_step:
                self.update_object_state()
                self.begin_time = time.time()

        return obs

    
    def keyboard_explore(self):
        obs = self.sims.get_sensor_observations(0)
        last_action = None
        release_count = 0
        
        while True:
            show_obs(obs)
            k, action = keyboard_control_fast()

            if action is None:
                continue
            if action == "stop":
                self.get_metric()
                self.reset()
                obs = self.sims.get_sensor_observations(0)
                if self.curr_task_id >= len(self.task_index):
                    break

            else:
                obs = self.step_dynamic_env(action)
            agent_state = self.agent.get_state()
            print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
            print('island:', self.plnner.pathfinder.get_island(agent_state.position))



if __name__ == "__main__":
    args = get_args()
    env = DynamicNavEnv(args, dynamic_env_config='/home/orbit/桌面/Nav-2025/3D-object-asset/objects_scripts/test.json', mode='test')
    env.keyboard_explore()