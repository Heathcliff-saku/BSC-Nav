from objnav_benchmark import *
from LLMAgent import *
from memory_2 import VoxelTokenMemory
import numpy as np
from PIL import Image
from args import get_args
import torch
from ultralytics import YOLOWorld
import os
import glob
import pickle
import random

class EQAAgent(GESObjectNavRobot):
    def __init__(self, memory, habitat_benchmark_env=None):
        super().__init__(memory, habitat_benchmark_env)
        self.save_dir = "localize_results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.obs_for_qa = []

    def _grid2loc(self, grid_id):
        
        row, col, height = grid_id[0], grid_id[1], grid_id[2]
        
        initial_position = self.memory.Env.original_state.position  # [x, z, y]
        initial_x, initial_z, initial_y = initial_position

        actual_y = initial_y + (row - self.memory.gs // 2) * self.memory.cs
        actual_x = initial_x + (col - self.memory.gs // 2) * self.memory.cs
        # actual_x = initial_z + 0.2
        actual_z = self.memory.Env.sims.agents[0].get_state().position[1] + 0.2
        # actual_z = (height + self.memory.minh) * self.memory.cs
        
        return np.array([actual_x, actual_z, actual_y])
    
    def working_memory_retrival(self, prompts, vis_aug=False, text_aug=True, region_radius=np.inf, curr_grid=None):
        
        if curr_grid is None and region_radius != np.inf:
            curr_state = self.memory.Env.sims.agents[0].get_state().position
            curr_grid = self._loc2grid(curr_state)

        if vis_aug:
            path = ['turn_left'] * int(360 / self.memory.args.turn_left)
            self.execute_path(path, save_img_list=True)
            vis = self.obss[::9]
        else:
            vis = None
        if isinstance(prompts, str):
            print("search voxel memory...")
            if text_aug:
                if vis:
                    while True:
                        try:
                            answer = imagenary_helper_visaug(self.client, prompts, vis)
                            print("aug_prompt:", answer)
                        except:
                            print("error, try again .....................................................")
                            time.sleep(50)
                            continue
                        pattern = r"\*\*Enhancement Description\*\*:\s*(.*?)(?=\n|\Z)"
                        match = re.search(pattern, answer, re.DOTALL)
                        if match:
                            text_prompt_extend = match.group(1).strip()
                            break
                        else:
                            continue
                else: 
                    while True:
                        try:
                            text_prompt_extend = imagenary_helper(self.client, prompts)
                            break
                        except:
                            print("error, try again .....................................................")
                            time.sleep(50)
                            continue
            else:
                text_prompt_extend = prompts
            
            best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(text_prompt_extend, region_radius=region_radius, curr_grid=curr_grid)

        elif isinstance(prompts, list):
            print("search voxel memory...")
            while True:
                try:
                    text_prompt_extend = imagenary_helper_long_text(self.client, prompts)
                    break
                except:
                    print("error, try again .....................................................")
                    time.sleep(50)
                    continue
                
            best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(text_prompt_extend, region_radius=region_radius, curr_grid=curr_grid)

        else:
            print("search voxel memory using image observ...")
            best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(prompts, region_radius=region_radius, curr_grid=curr_grid)
            
        cluster_centers, _, _ = self.weighted_cluster_centers(top_k_positions, top_k_similarity)
        print("Extracted Loc Array using voxel memory:", cluster_centers)  

        # if isinstance(prompts, str):
        #     if len(prompts) > 64:
        #         prompts = prompts[:64]
        #     np.save(self.memory.memory_save_path + f"/best_pos_topK_{prompts}.npy", np.array(top_k_positions))
        #     np.save(self.memory.memory_save_path + f"/best_pos_centers_{prompts}.npy", np.array(cluster_centers))
        
        best_pos = np.array([cluster_centers])
        return best_pos
    
    def execute_path(self, path, save_img_list=False):
        if len(self.obss) != 0:
            self.obss = []

        for action in path:

            if not self.memory.args.no_vis:
                # map_2d = self.trajectory_drawer.get_map(self.benchmark_env.sim.agents[0].get_state())
                show_obs(self.curr_obs)
                cv2.waitKey(1)
            if not self.memory.args.quite:
                print("action:", action)
            
            self.state_hist.append(self.memory.Env.sims.agents[0].get_state())
            self.curr_obs = self.memory.Env.sims.step(action)

            if not self.memory.args.no_record:
                self.episode_images.append(self.curr_obs["rgb"].copy())

            if save_img_list:
                img = Image.fromarray(self.curr_obs["rgb"][:, :, :3])
                self.obss.append(img)
        

    def move2anhorobject(self, text_prompt):
        self.obs_for_qa = []
        self.curr_obs = self.memory.Env.sims.get_sensor_observations(0)
        self.task_over = False
        
        best_poses = self.working_memory_retrival(text_prompt, vis_aug=False)
        query_num = min(len(best_poses[0]), 3)
        if best_poses is not None:
            self.loc_hist['working_memory'].extend(best_poses[0][:query_num])
            for best_pos in best_poses[0][:query_num]:
                self.nav_log['working_memory_query'] += 1
                self.nav_log['search_point'] += 1
                best_pos = self._grid2loc(best_pos)
                try:
                    path, self.goal = self.memory.Env.move2point(best_pos)
                    if len(path) > 2000:
                        print("path too long, skip")
                        continue
                    else:
                        self.execute_path(path[:-1], save_img_list=True)
                        # self.obs_for_qa += self.obss
                        if len(self.obss) > 10:
                            self.obss = random.sample(self.obss, 10)
                        self.obs_for_qa += self.obss
                except Exception as e:
                    print(f"move2point failed: {e}")
                    continue
                
                self.check_around(text_prompt)

                if self.task_over:
                    # self.execute_path(['stop'])
                    path = ["look_up"]*2 + ["turn_left"] * int(360 / self.memory.args.turn_left) + ["look_down"]*2 + ["turn_left"] * int(360 / self.memory.args.turn_left) + ["look_down"]*2 + ["turn_left"] * int(360 / self.memory.args.turn_left)
                    self.execute_path(path, save_img_list=True)
                    self.obs_for_qa += self.obss[::3]
                    self.save_log()
                    return
                else:
                    continue

        # self.touching_goal(text_prompt, [Image.fromarray(self.curr_obs["rgb"][:, :, :3])], max_steps=10)
        
        # self.execute_path(['stop'])
        self.save_log()
        
        return
    
    def random_move(self):
        self.obs_for_qa = []
        current_island = self.memory.Env.sims.pathfinder.get_island(state.position)
        area_shape = self.memory.Env.sims.pathfinder.island_area(current_island)
        random_move_num = int(area_shape / 2) + 1

        for _ in tqdm(range(random_move_num)):
            subgoal = self.memory.Env.plnner.pathfinder.get_random_navigable_point()
            island_goal = self.memory.Env.plnner.pathfinder.get_island(subgoal)
            island_begin = self.memory.Env.plnner.pathfinder.get_island(self.memory.Env.agent.get_state().position)

            while (not self.memory.Env.plnner.pathfinder.is_navigable(subgoal)) or (island_goal != island_begin):
                subgoal = self.memory.Env.plnner.pathfinder.get_random_navigable_point()
                island_goal = self.memory.Env.plnner.pathfinder.get_island(subgoal)

            try:
                path, goal = self.memory.Env.move2point(subgoal)
                self.execute_path(path[:-1], save_img_list=True)
                if len(self.obss) > 5:
                    self.obss = random.sample(self.obss, 5) 
                self.obs_for_qa += self.obss

                action_around = ['turn_left'] * int(360 / self.memory.args.turn_left)
                self.execute_path(action_around, save_img_list=True)
                self.obs_for_qa += self.obss[::3]

            except Exception as e:
                print(f"移动失败: {e}")
                continue

        # 如果self.obs_for_qa的长度大于50，则随机采样50个
        if len(self.obs_for_qa) > 50:
            self.obs_for_qa = random.sample(self.obs_for_qa, 50)



    def main(self, question):
        del self.obs_for_qa
        gc.collect()

        self.obs_for_qa = []

        while True:
            self.api_id += 1 
            self.client.api_key = self.api_key_pool[self.api_id%6]
            try:
                anchor_object_text = EQA_generate_anchor_object(self.client, question)
                break
            except:
                print("error, try again .....................................................")
                time.sleep(50)
                continue

        print(anchor_object_text)
        self.agent_response_log.append(anchor_object_text)
        if "{" in anchor_object_text:
            anchor_object_text = anchor_object_text.split('{')[1].split('}')[0]
            self.move2anhorobject(anchor_object_text)

            if len(self.obs_for_qa) == 0 or not self.task_over:
                self.random_move()

        else:
            self.random_move()

        # answer1 = EQA_Answer_o3(self.client, question, self.obs_for_qa)

        while True:
            self.api_id += 1 
            self.client.api_key = self.api_key_pool[self.api_id%6]
            try:
                answer2 = EQA_Answer_4o(self.client, question, self.obs_for_qa)
                break
            except:
                print("error, try again .....................................................")
                time.sleep(50)
                continue
        
        return self.episode_images, self.episode_topdowns, self.episode_vedio, answer2





if __name__ == "__main__":
    args = get_args()
    dinov2 = torch.hub.load('facebookresearch/dinov2', args.dino_size, source='github').to('cuda')
    yolow = YOLOWorld("yolov8x-worldv2.pt").to('cuda')
    yolow.set_classes(args.detect_classes)

    with open("/home/orbit/桌面/Nav-2025/data_episode/eqa/openeqa_hm3d_subset.json") as file:
        dataset = json.load(file)

    results = []

    all_scense = [name for name in os.listdir("/home/orbit/桌面/Nav-2025/memory/eqa")]

    memory = VoxelTokenMemory(args, build_map=False, preload_dino=dinov2, preload_yolo=yolow)
    robot = EQAAgent(memory)
    

    result_file = "/home/orbit/桌面/Nav-2025/data_episode/eqa/results_subset.json"
    # 首先尝试读取已有的 eqa_results.json，确定已经完成的条目数量，实现断点续跑
    current_results = []
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            current_results = json.load(f)

    # 构建已完成的question_id集合，加速查找
    finished_ids = set()
    for item in current_results:
        if "question_id" in item:
            finished_ids.add(item["question_id"])

    for i in tqdm(range(len(dataset))):
        torch.cuda.empty_cache()

        question_id = dataset[i]["question_id"]
        # 如果当前question_id已经存在于current_results中，则跳过本次循环
        if question_id in finished_ids:
            continue

        dir = "./tmp/trajectory_%d"%i
        os.makedirs(dir, exist_ok=True)
        fps_writer = imageio.get_writer("%s/fps.mp4"%dir, fps=4)

        question = dataset[i]['question']
        scense_raw = dataset[i]['episode_history'].split('-')[-1]
        scense = [name for name in all_scense if name.endswith(scense_raw)][0]

        memory_path = glob.glob('{}/eqa/{}'.format(args.memory_path, "*" + scense))[0]

        args.scene_name = scense
        args.load_memory_path = memory_path
        args.memory_save_path = memory_path

        with open(glob.glob("/home/orbit/桌面/Nav-2025/data_episode/eqa/data/frames/hm3d-v0/{}/00000.pkl".format("*"+scense_raw))[0], 'rb') as file:
            state = pickle.load(file)['agent_state']

        robot.memory.load_memory(init_state=state, build_map=False)

        robot.reset(log_dir=dir)
        episode_images, episode_topdowns, episode_vedio, answer_4o = robot.main(question)
        # dataset[i]['answer_o3'] = answer_o3
        dataset[i]['answer_4o'] = answer_4o
        current_results.append(dataset[i])

        # 每次迭代都保存一次结果，防止中途出错丢失进度
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(current_results, f, ensure_ascii=False, indent=2)

        for image in episode_images:
            fps_writer.append_data(image)
        fps_writer.close()

        del fps_writer
        gc.collect()
        torch.cuda.empty_cache()