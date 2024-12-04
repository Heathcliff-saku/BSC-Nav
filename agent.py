from memory_2 import VoxelTokenMemory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from args import get_args
import numpy as np
import time
from utils import *

class GESNavRobot:
    def __init__(self):
        
        self.args = get_args()
        self.memory = VoxelTokenMemory(self.args)
        
        if self.args.explore_first:
            self.memory.create_memory()
        else:
            self.memory.load_memory()    
    
    def vis_3d(self, best_pos):
           
        rgb = self.memory.grid_rgb[:self.memory.max_id] / 255.0
        pc = self.memory.grid_rgb_pos[:self.memory.max_id]
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=rgb, s=1)
        ax.scatter(best_pos[0], best_pos[1], best_pos[2], c='r', s=50)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Point Cloud Visualization')
        ax.set_facecolor((0, 0, 0))
        plt.show()
    
    
    def _grid2loc(self, grid_id):
        
        row, col, height = grid_id[0,0], grid_id[0,1], grid_id[0,2]
        initial_position = self.memory.Env.original_state.position  # [x, z, y]
        initial_x, initial_z, initial_y = initial_position

        actual_y = initial_y + (row - self.memory.gs // 2) * self.memory.cs
        actual_x = initial_x + (col - self.memory.gs // 2) * self.memory.cs
        actual_z = initial_z
        
        # 返回实际坐标
        return np.array([actual_x, actual_z, actual_y])
        

    def Move2TextPrompt(self, text_prompts):
        
        best_pos = self.memory.localized(text_prompts)
        # self.vis_3d(best_pos)
        best_pos = self._grid2loc(best_pos)
        path, goal = self.memory.Env.move2point(best_pos)
        
        return path, goal
    
    def main(self):
        
        obs = self.memory.Env.sim.get_sensor_observations(0)
        last_action = None
        release_count = 0
        
        while True:
            show_obs(obs)
            k, action = keyboard_control_nav()
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
            
            if action == "nav":
                text_prompt = input("Please describe where you want to go:")
                path, goal = self.Move2TextPrompt(text_prompt)
                for idx, action in enumerate(path):
                    if action != "stop":
                        show_obs(obs)
                        print("action:", action)
                        obs = self.memory.Env.sim.step(action)
                        agent_state = self.memory.Env.agent.get_state()
                        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
                        cv2.waitKey(1)
                    else:
                        print("----------------------")
                        print("Getting there!")
                        print("current pos:", agent_state.position)
                        print("goal pos:", goal)
                        print("distance:", np.sqrt(np.sum(np.square(agent_state.position - goal))))
                last_action = None
                continue
            
            else:
                obs = self.memory.Env.sim.step(action)
                agent_state = self.memory.Env.agent.get_state()
                print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

if __name__ == "__main__":
    
    Agent = GESNavRobot()
    # Agent.Move2TextPrompt(text_prompts='A marble island in a kitchen.')
    Agent.main()