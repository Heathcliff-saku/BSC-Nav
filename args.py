import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    # habitat env
    parser.add_argument("--dataset_dir", type=str, default='/home/orbit-new/桌面/orbit/shouwei_GES/VLDMap/Vlmaps/DistributionMap_dynamic/mp3d_datasource/v1/tasks/mp3d')
    parser.add_argument("--scene_dataset_config_file", type=str, default='/home/orbit-new/桌面/orbit/shouwei_GES/VLDMap/Vlmaps/DistributionMap_dynamic/mp3d_datasource/v1/tasks/mp3d/mp3d.scene_dataset_config.json')
    parser.add_argument("--scene_name", type=str, default='5q7pvUzZiYa') # 5LpN3gDmAk7 5q7pvUzZiYa
    
    # agent sensor
    parser.add_argument("--color_sensor", action="store_false")
    parser.add_argument("--depth_sensor", action="store_false")
    parser.add_argument("--semantic_sensor", action="store_false")
    
    parser.add_argument("--sensor_height", type=float, default=1.5)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--lidar_fov", type=int, default=360)
    parser.add_argument("--depth_img_for_lidar_n", type=int, default=20)
    
    # actions
    parser.add_argument("--move_forward", type=float, default=0.1)
    parser.add_argument("--turn_left", type=int, default=5)
    parser.add_argument("--turn_right", type=int, default=5)
    
    
    # memory
    parser.add_argument("--dino_size", type=str, default='dinov2_vitl14_reg')
    parser.add_argument("--memory_path", type=str, default='/home/orbit-new/桌面/Nav-2025/memory')
    
    
    parser.add_argument("--floor_height", type=float, default=-0.5)
    parser.add_argument("--map_height", type=float, default=2.0)
    
    parser.add_argument("--cell_size", type=float, default=0.05)
    parser.add_argument("--grid_size", type=float, default=1000)
    
    parser.add_argument("--base_forward_axis", type=list, default=[0, 0, -1])
    parser.add_argument("--base_left_axis", type=list, default=[-1, 0, 0])
    parser.add_argument("--base_up_axis", type=list, default=[0, 1, 0])
    parser.add_argument("--base2cam_rot", type=list, default=[1, 0, 0, 0, -1, 0, 0, 0, -1])
    parser.add_argument("--cam_calib_mat", type=list, default=[540, 0, 540, 0, 540, 360, 0, 0, 1])
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10)
    parser.add_argument("--depth_sample_rate", type=int, default=100)
    
    
    # agent
    parser.add_argument("--explore_first", action="store_true")
    parser.add_argument("--load_memory_path", type=str, default='/home/orbit-new/桌面/Nav-2025/memory/5q7pvUzZiYa_2')
    
    args = parser.parse_args()
    
    return args

