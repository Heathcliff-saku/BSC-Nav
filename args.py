import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--no_record", action="store_true")

    parser.add_argument("--dataset", type=str, default='hm3d')  #'mp3d'
    # habitat env
    # MP3D
    # parser.add_argument("--dataset_dir", type=str, default='/home/orbit/桌面/Nav-2025/data/mp3d/mp3d_habitat/mp3d') 
    # parser.add_argument("--scene_dataset_config_file", type=str, default='/home/orbit/桌面/Nav-2025/data/mp3d/mp3d_habitat/mp3d/mp3d.scene_dataset_config.json')
    # HM3D
    parser.add_argument("--dataset_dir", type=str, default='/home/orbit/桌面/Nav-2025/data/scene_datasets/hm3d/hm3d/val') 
    parser.add_argument("--scene_dataset_config_file", type=str, default='/home/orbit/桌面/Nav-2025/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json')
    parser.add_argument("--scene_name", type=str, default='00873-bxsVRursffK') # 5LpN3gDmAk7 5q7pvUzZiYa kEZ7cmS4wCh 2azQ1b91cZZ 00873-bxsVRursffK
    
    
    # agent sensor
    parser.add_argument("--color_sensor", action="store_false")
    parser.add_argument("--depth_sensor", action="store_false")
    parser.add_argument("--semantic_sensor", action="store_false")
    
    parser.add_argument("--sensor_height", type=float, default=1.5)
    # 传感器图像大小
    # 需要使用LLM agent的图像大小
    parser.add_argument("--width", type=int, default=680)
    parser.add_argument("--height", type=int, default=680)
    parser.add_argument("--lidar_fov", type=int, default=360)
    parser.add_argument("--depth_img_for_lidar_n", type=int, default=20)
    
    # actions
    parser.add_argument("--move_forward", type=float, default=0.25)
    parser.add_argument("--move_backward", type=float, default=-0.1)
    parser.add_argument("--turn_left", type=int, default=30)
    parser.add_argument("--turn_right", type=int, default=30)
    
    
    # memory
    # 用于构建memory时提取token的图像大小
    # 以及导航时进行query的联想图像需要缩放的大小
    parser.add_argument("--query_width", type=int, default=224) 
    parser.add_argument("--query_height", type=int, default=224)
    
    parser.add_argument("--gen_width", type=int, default=512)
    parser.add_argument("--gen_height", type=int, default=512)
    parser.add_argument("--imagenary_num", type=int, default=3)
    parser.add_argument("--diffusion_id", type=str, default='stabilityai/stable-diffusion-3.5-medium') # stabilityai/stable-diffusion-3.5-medium stabilityai/stable-diffusion-3.5-large-turbo
    
    parser.add_argument("--dino_size", type=str, default='dinov2_vitl14_reg')
    parser.add_argument("--memory_path", type=str, default='/home/orbit/桌面/Nav-2025/memory')
    
    
    parser.add_argument("--floor_height", type=float, default=-10.0)
    parser.add_argument("--map_height", type=float, default=10.0)
    
    parser.add_argument("--cell_size", type=float, default=0.1)
    parser.add_argument("--grid_size", type=float, default=1000)
    
    parser.add_argument("--base_forward_axis", type=list, default=[0, 0, -1])
    parser.add_argument("--base_left_axis", type=list, default=[-1, 0, 0])
    parser.add_argument("--base_up_axis", type=list, default=[0, 1, 0])
    parser.add_argument("--base2cam_rot", type=list, default=[1, 0, 0, 0, -1, 0, 0, 0, -1])
    # parser.add_argument("--cam_calib_mat", type=list, default=[540, 0, 540, 0, 540, 360, 0, 0, 1])
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=10)
    parser.add_argument("--depth_sample_rate", type=int, default=1000)
    
    # parser.add_argument("--detect_classes", type=str, default="Sofa. Bed. Television. Desk. Chair. Dining table. Lamp. Bookshelf. Refrigerator. Microwave. Washing machine. Computer. TV. Food. Cup. Cookware. Sink. Bathtub. Towel. Mirror")
    # parser.add_argument("--detect_classes", type=list, default=["Sofa", "Bed", "Television", "Desk", "Chair", "Dining table", "Lamp", "Bookshelf", "Refrigerator", "Microwave", "Washing machine", "Computer", "TV", "Food", "Cup", "Cookware", "Sink", "Bathtub", "Towel", "Mirror"])
    # parser.add_argument("--detect_classes", type=list, default=["chair", "sofa", "potted plant", "bed", "toilet", "tv monitor"])
    parser.add_argument("--detect_classes", type=list, default=['seating', 'chest of drawers', 'bed', 'bathtub', 'clothes', 'toilet', 'stool', 'sofa', 'sink', 'tv monitor', 'picture', 'cushion', 'towel', 'shower', 'counter', 'fireplace', 'chair', 'table', 'gym equipment', 'cabinet', 'plant'])
    parser.add_argument("--detect_conf", type=float, default=0.55)
    # agent
    parser.add_argument("--explore_first", action="store_true")
    # parser.add_argument("--load_memory_path", type=str, default='/home/orbit/桌面/Nav-2025/memory/2azQ1b91cZZ_4')
    parser.add_argument("--load_memory_path", type=str, default='/home/orbit/桌面/Nav-2025/memory/objectnav/hm3d_v2/00814-p53SfW6mjZe_island_0')
    parser.add_argument("--quite", type=bool, default=True)

    # single-floor or whole house in loading memory
    parser.add_argument("--load_single_floor", action="store_true")
    
    # exploring
    parser.add_argument("--random_move_num", type=float, default=30)

    parser.add_argument("--use_only_working_memory", action="store_true")

    #-----------------------------------------------------------------
    # objectnav benchmark
    parser.add_argument("--nav_task", type=str, default='objnav') # objnav, imgnav, ovon, r2r
    
    HABITAT_ROOT_DIR = "/home/orbit/桌面/Nav-2025/third-party/habitat-lab"
    parser.add_argument("--HM3D_CONFIG_PATH", type=str, default=f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml")
    parser.add_argument("--MP3D_CONFIG_PATH", type=str, default=f"{HABITAT_ROOT_DIR}/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml")
    
    parser.add_argument("--HM3D_SCENE_PREFIX", type=str, default="/home/orbit/桌面/Nav-2025/data/scene_datasets/hm3d/")
    parser.add_argument("--HM3D_EPISODE_PREFIX", type=str, default="/home/orbit/桌面/Nav-2025/baselines/Pixel-Navigator/checkpoints/objectnav_hm3d_v2/val/val.json.gz")

    parser.add_argument("--MP3D_SCENE_PREFIX", type=str, default="/home/orbit/桌面/Nav-2025/data/mp3d/mp3d_habitat/")
    parser.add_argument("--MP3D_EPISODE_PREFIX", type=str, default="/home/orbit/桌面/Nav-2025/baselines/Pixel-Navigator/checkpoints/objectnav_mp3d_v1/val/val.json.gz")

    parser.add_argument("--image_hfov", type=int, default=90)

    parser.add_argument("--benchmark_dataset", type=str, default='hm3d')
    parser.add_argument("--eval_episodes", type=int, default=1000)    # hm3d-0.1 2000 / mp3d-0.1 2195  
    parser.add_argument("--max_episode_steps", type=int, default=5000)  
    parser.add_argument("--success_distance", type=float, default=1.0)  
    
    
    

    args = parser.parse_args()
    
    return args

