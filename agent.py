from memory_2 import VoxelTokenMemory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from args import get_args
import numpy as np
import time
import torch
import re
from openai import OpenAI
import open_clip
from sklearn.cluster import DBSCAN

from LLMAgent import long_memory_localized, imagenary_helper, succeed_determine
from utils import *
from objnav_benchmark import GESObjectNavRobot
import habitat_sim.utils.datasets_download

# 🙀1. 完善成功判定，循环 
# 🙀-- 机器人上下摆动摄像头以确定目标
# 2. 完善 long-term memory写入和检索
# 3. 完善 working memory 网格内更新规则 替换为 faiss
# 🙀3. 加入物体
# 🙀4.联想增强，生成多个图像+舍弃外圈保留内圈token

# 🙀3. 随机探索建图

class GESNavRobot:
    def __init__(self):
        
        self.args = get_args()
        self.memory = VoxelTokenMemory(self.args)
        if self.args.explore_first:
            self.memory.create_memory()
        else:
            self.memory.load_memory()    
        self.client = OpenAI(
                base_url='https://xiaoai.plus/v1',
                # sk-xxx替换为自己的key
                api_key='sk-olO6Qk4p04qzy54JB8D453D3DeF0426eAaD29f3bAaAd0906'
            )

        self.pattern_loc = re.compile(r'Nav Loc:\s*\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]')
        self.pattern_unable = re.compile(r'Nav Loc:\s*Unable to find', re.IGNORECASE)
        self.pattern_unsuccess = re.compile(r'success:\s*(yes|no)', re.IGNORECASE)
        self.pattern_id = re.compile(r'best_img_id:\s*(\d+)', re.IGNORECASE)
        
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-H/14', pretrained='metaclip_fullcc')
        self.clip_model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
        self.clip_model.to(self.memory.device)
    
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
        
        row, col, height = grid_id[0], grid_id[1], grid_id[2]
        initial_position = self.memory.Env.original_state.position  # [x, z, y]
        initial_x, initial_z, initial_y = initial_position

        actual_y = initial_y + (row - self.memory.gs // 2) * self.memory.cs
        actual_x = initial_x + (col - self.memory.gs // 2) * self.memory.cs
        actual_z = initial_z
        
        # 返回实际坐标
        return np.array([actual_x, actual_z, actual_y])
        
    def weighted_cluster_centers(self, top_k_positions, top_k_similarity, eps=10, min_samples=5):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(top_k_positions)
        unique_labels = [lbl for lbl in set(labels) if lbl != -1]

        cluster_info = []
        for lbl in unique_labels:
            cluster_mask = (labels == lbl)
            cluster_points = top_k_positions[cluster_mask]
            cluster_weights = top_k_similarity[cluster_mask]
            weighted_center = np.average(cluster_points, axis=0, weights=cluster_weights)
            avg_similarity = np.mean(cluster_weights)  # 计算相似度均值
            cluster_info.append((avg_similarity, weighted_center, np.sum(cluster_mask)))

        # 根据相似度均值进行降序排序
        cluster_info.sort(key=lambda x: x[0], reverse=True)

        cluster_centers = np.array([info[1] for info in cluster_info])
        cluster_sizes = [info[2] for info in cluster_info]
        
        return cluster_centers, labels, cluster_sizes
    
    
    def long_term_memory_retrival(self, text_prompts):
        # 首先使用LLM进行long_memory检索
        print("search long memory...")
        while True:
            answer = long_memory_localized(self.client, text_prompts, self.memory.long_memory_dict)
            print(answer)
            if self.pattern_unable.search(answer):
                return None
            elif self.pattern_loc.search(answer):
                x, y, z = self.pattern_loc.search(answer).groups()  
                best_pos = np.array([int(x), int(y), int(z)])
                print("Extracted Loc Array using long memory:", best_pos)  
                return best_pos
            else:
                continue
            
    def working_memory_retrival(self, text_prompts):
        print("search voxel memory...")
        text_prompt_extend = imagenary_helper(self.client, text_prompts)
        best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(text_prompt_extend)
        
        cluster_centers, _, _ = self.weighted_cluster_centers(top_k_positions, top_k_similarity)
        
        print("Extracted Loc Array using voxel memory:", cluster_centers)  
        np.save(self.memory.memory_save_path + f"/best_pos_topK_{text_prompts}.npy", np.array(top_k_positions))
        np.save(self.memory.memory_save_path + f"/best_pos_centers_{text_prompts}.npy", np.array(cluster_centers))
        
        best_pos = np.array([cluster_centers])
        return best_pos
        
    
    # def check_around(self, text, max_around=5):
    #     matched_images = []
    #     complete_path = []
    #     matched_indices = []
        
    #     for j in range(max_around):
    #         action_around = ['turn_left'] * int(360 / self.args.turn_left)
    #         obss = self.execute_path(action_around, save_img_list=True)
    #         obss_tensor = [self.preprocess(obs) for obs in obss]
    #         batch_obs = torch.stack(obss_tensor).to(self.memory.device)
    #         text_inputs = open_clip.tokenize([text]).to(self.memory.device)
            
    #         with torch.no_grad(), torch.cuda.amp.autocast():
    #             image_features = self.clip_model.encode_image(batch_obs)
    #             text_features = self.clip_model.encode_text(text_inputs)
                
    #             image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    #             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
    #             similarities = (image_features @ text_features.T).squeeze(1)  # [num_turns]
    #             similarities = similarities.softmax(dim=0)
            
    #         max_val, max_idx = similarities.max(dim=0)
    #         target_angle = max_idx.item() * self.args.turn_left
        
    #         matched_images.append(obss[max_idx])
            
    #         if target_angle <= 180:                                         
    #             actions = ['turn_left' for _ in range(max_idx)]
    #         else:
    #             actions = ['turn_right' for _ in range(int(360/self.args.turn_left)-max_idx)]
            
    #         complete_path.extend(actions)
    #         _ = self.execute_path(actions)
    #         matched_indices.append(len(complete_path))
            
    #         rotate_actions = ['turn_right' for _ in range(int(180/self.args.turn_left))] + ['move_forward'] * 2
    #         _ = self.execute_path(rotate_actions)
            
    #         complete_path.extend(rotate_actions)
            
    #         del batch_obs
    #         torch.cuda.empty_cache()
            
    #     return matched_images, complete_path, matched_indices
    
    
    def check_around(self, text, max_around=2):
        # 环绕-采集图像-gpt判定方位-不存在则后退循环
        steps_per_30 = int(30 / self.args.turn_left)
        
        for j in range(max_around):
            action_around = ['turn_left'] * int(360 / self.args.turn_left)
            obss, obss_back = self.execute_path(action_around, save_img_list=True)
            combined_obss = obss + obss_back
            
            obss_tensor = [self.preprocess(obs) for obs in combined_obss]
            batch_obs = torch.stack(obss_tensor).to(self.memory.device)
            text_inputs = open_clip.tokenize([text]).to(self.memory.device)
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                image_features = self.clip_model.encode_image(batch_obs)
                text_features = self.clip_model.encode_text(text_inputs)
                
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ text_features.T).squeeze(1)  # [num_turns]
                similarities = similarities.softmax(dim=0)
            
            max_val, max_idx = similarities.max(dim=0)
            num_turns = int(360 / self.args.turn_left)
            if max_idx >= num_turns:
                max_idx -= num_turns
            
            match_obs = [combined_obss[max_idx], combined_obss[max_idx+num_turns]]
            
            target_angle = max_idx.item() * self.args.turn_left
            if target_angle <= 180:
                actions = ['turn_left'] * max_idx.item()
            else:
                total_steps = int(360 / self.args.turn_left)
                actions = ['turn_right'] * (total_steps - max_idx.item())
            _, _ = self.execute_path(actions)
                    
            success = self.handle_succeed_check(text, match_obs)
            if success:
                
                self.task_over = True
                print("----------------------")
                print('detect goal, task success')
                break
            else:
                rotate_steps = int(180 / self.args.turn_left)
                rotate_actions = ['turn_left'] * rotate_steps + ['move_forward'] * 2
                _, _ = self.execute_path(rotate_actions)
            
            
            
            # extracted_obss = [obss[i] for i in range(0, len(obss), steps_per_30)]
            
            # id_ = self.handle_succeed_check(text, extracted_obss)
            # if id_ is not None:
            #     target_turn_steps = id_ * steps_per_30
            #     target_angle = target_turn_steps * self.args.turn_left
                
            #     if target_angle <= 180:
            #         actions = ['turn_left'] * target_turn_steps
            #     else:
            #         total_steps = int(360 / self.args.turn_left)
            #         actions = ['turn_right'] * (total_steps - target_turn_steps)
                    
            #     _, _ = self.execute_path(actions)
            #     self.task_over = True
            #     print("----------------------")
            #     print('detect goal, task success')
            #     break
            # else:
            #     rotate_steps = int(180 / self.args.turn_left)
            #     rotate_actions = ['turn_left'] * rotate_steps + ['move_forward'] * 2
            #     _, _ = self.execute_path(rotate_actions)
        
        
        
    def handle_succeed_check(self, text_prompt, obss):
        while True:
            succeed_answer = succeed_determine(self.client, text_prompt, obss) 
            print(succeed_answer)
            match = self.pattern_unsuccess.search(succeed_answer)
            
            if match:
                status = match.group(1).lower()
                if status == 'no':
                    print("No match found.")
                    return False
                elif status == 'yes':
                    print(f"Match found.")
                    return True
            else:
                print("No valid 'Success' status found in the response. Continuing the loop...")
                # Depending on the desired behavior, you can choose to continue or break
                # For this example, we'll break and return (None, obs)
                continue
                
    def execute_path(self, path, save_img_list=False):
        obss = []
        obss_back = []
        for action in path:
            if action != "stop":
                show_obs(self.curr_obs)
                if not self.args.quite:
                    print("action:", action)
                self.curr_obs = self.memory.Env.sims.step(action)
                if save_img_list:
                    img = Image.fromarray(self.curr_obs["rgb"][:, :, :3])
                    img_back = Image.fromarray(self.curr_obs["back_rgb"][:, :, :3])
                    obss.append(img)
                    obss_back.append(img_back)
                    
                agent_state = self.memory.Env.agent.get_state()
                cv2.waitKey(1)
                
                if not self.args.quite:
                    print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)
                    
            else:
                if not self.args.quite:
                    print("----------------------")
                    print("Getting there!")
                    print("current pos:", agent_state.position)
                    print("goal pos:", self.goal)
                    print("distance:", np.sqrt(np.sum(np.square(agent_state.position - self.goal))))
            
        return obss, obss_back
            
    def move2textprompt(self):

        text_prompt = input("Please describe where you want to go:")
        self.task_over = False

        # Step 1: Attempt long-term memory retrieval
        best_pos = self.long_term_memory_retrival(text_prompt)
        if best_pos is not None:
            best_pos = self._grid2loc(best_pos)
            path, self.goal = self.memory.Env.move2point(best_pos)
            _, _ = self.execute_path(path)
            self.check_around(text_prompt) 
            if self.task_over:
                return

        # Step 2: Attempt working memory retrieval if long-term retrieval fails
        best_poses = self.working_memory_retrival(text_prompt)
        for best_pos in best_poses[0]:
            best_pos = self._grid2loc(best_pos)
            path, self.goal = self.memory.Env.move2point(best_pos)
            _, _ = self.execute_path(path)
            self.check_around(text_prompt)
            if self.task_over:
                break
            else:
                continue



    def main_text_input(self):
        
        self.curr_obs = self.memory.Env.sims.get_sensor_observations(0)
        last_action = None
        release_count = 0
        
        while True:
            show_obs(self.curr_obs)
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
                self.move2textprompt()
                last_action = None
                continue
            
            else:
                self.curr_obs = self.memory.Env.sims.step(action)
                agent_state = self.memory.Env.agent.get_state()
                print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)




if __name__ == "__main__":
    
    Agent = GESNavRobot()
    # Agent.Move2TextPrompt(text_prompts='A marble island in a kitchen.')
    Agent.main_text_input()