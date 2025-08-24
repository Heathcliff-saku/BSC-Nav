from objnav_benchmark import GESObjectNavRobot
from LLMAgent import *
from memory_2 import VoxelTokenMemory
import numpy as np
from PIL import Image
from args import get_args
import torch
from ultralytics import YOLOWorld
import os

class LocalizeAgent(GESObjectNavRobot):
    def __init__(self, memory, habitat_benchmark_env=None):
        super().__init__(memory, habitat_benchmark_env)
        self.save_dir = "localize_results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def LocalizeTextPrompt(self, text_prompt):

        text_prompt_extend = imagenary_helper(self.client, text_prompt)
        print(text_prompt_extend)
        best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(text_prompt_extend)
        
        cluster_centers, _, _ = self.weighted_cluster_centers(top_k_positions, top_k_similarity)
        
        print("Extracted Loc Array using voxel memory:", cluster_centers)  
        np.save(f"{self.save_dir}/best_pos_topK_text_prompt.npy", np.array(top_k_positions))
        # np.save(f"{save_dir}/best_pos_centers_{safe_text_prompt}.npy", np.array(cluster_centers))

    def LocalizeImagePrompt(self, img):

        best_pos, top_k_positions, top_k_similarity = self.memory.voxel_localized(img)
        
        cluster_centers, _, _ = self.weighted_cluster_centers(top_k_positions, top_k_similarity)
        
        print("Extracted Loc Array using voxel memory:", cluster_centers)  
        np.save(f"{self.save_dir}/best_pos_topK_img_input.npy", np.array(top_k_positions))




if __name__ == "__main__":
    args = get_args()
    dinov2 = torch.hub.load('facebookresearch/dinov2', args.dino_size, source='github').to('cuda')
    yolow = YOLOWorld("yolov8x-worldv2.pt").to('cuda')
    yolow.set_classes(args.detect_classes)

    memory = VoxelTokenMemory(args, build_map=False, preload_dino=dinov2, preload_yolo=yolow)
    agent = LocalizeAgent(memory)

    agent.memory.load_memory(init_state=None, build_map=False)

    # agent.LocalizeTextPrompt("A modern, low-profile white porcelain toilet composed of a smooth oval bowl, a slim matching lid, and a compact rectangular tank with a small chrome side-mounted flush lever. The base flares gently where it meets the tan ceramic floor tiles, giving it a seamless, molded look. Glossy beige wall tiles rise behind it, making the toilet’s bright white glaze stand out. No decorative items sit on the tank; the surface is clean and bare. The seat and lid are both closed, their edges forming a neat, even line over the rim. Just above and to the side is a brushed-steel towel bar holding neatly folded white and aqua hand towels, reinforcing the bathroom’s tidy, contemporary feel.")
    # agent.LocalizeImagePrompt(Image.open("/home/orbit/桌面/Nav-2025/text.png"))

    agent.LocalizeTextPrompt("toilt")