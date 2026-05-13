# BSC-Nav Physical Deployment (phy branch)

**Brain-inspired Spatial Cognition for Navigation — Real-World Robotic Deployment**

This branch contains the complete codebase for deploying BSC-Nav on a physical mobile robotic platform. It implements structured spatial memory (landmark memory + cognitive map) construction and retrieval for goal-directed navigation in real indoor environments.

---

## Table of Contents

- [System Overview](#system-overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Dependencies](#software-dependencies)
- [Installation](#installation)
- [System Architecture](#system-architecture)
- [Deployment Pipeline](#deployment-pipeline)
  - [Phase 1: Environment Setup & SLAM Mapping](#phase-1-environment-setup--slam-mapping)
  - [Phase 2: Data Collection (Spatial Memory Pre-construction)](#phase-2-data-collection-spatial-memory-pre-construction)
  - [Phase 3: Spatial Memory Construction](#phase-3-spatial-memory-construction)
  - [Phase 4: Navigation Task Execution](#phase-4-navigation-task-execution)
  - [Phase 5: Navigation with Manipulation (Optional)](#phase-5-navigation-with-manipulation-optional)
- [File Descriptions](#file-descriptions)
- [Configuration Reference](#configuration-reference)
- [Coordinate Systems](#coordinate-systems)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## System Overview

BSC-Nav deploys a brain-inspired spatial cognition framework on a physical robot to perform:
- **Object-Goal Navigation (OGN)**: Navigate to a target object category (e.g., "chair")
- **Text-Instance Navigation (TIN)**: Navigate to a specific object described in natural language (e.g., "A yellow armchair near the window")
- **Image-Instance Navigation (IIN)**: Navigate to a specific object shown in a query image
- **Mobile Manipulation**: Navigate to targets and perform grasping/manipulation actions

The system operates in two phases:
1. **Memory Construction** (offline, teleoperated): Build structured spatial memory (cognitive map + landmark memory) by manually guiding the robot through the environment.
2. **Task Execution** (online, autonomous): Given a navigation goal, retrieve candidate targets from spatial memory and autonomously navigate to them.

---

## Hardware Requirements

| Component | Specification | Purpose |
|-----------|--------------|----------|
| Locomotion Chassis | Agilex Ranger-mini 3.0 (Ackermann steering, zero-radius turning) | Mobile base |
| Industrial Computer | NVIDIA RTX 4090 GPU (24 GB VRAM), ≥16 GB RAM | On-board inference |
| Vision Sensor | Intel RealSense D435i × 2 (848×480, 30 FPS, 87° FOV) | RGB-D perception |
| LiDAR | 32-beam LiDAR + IMU | SLAM & localization |
| Robotic Arm (optional) | Franka Emika Research 3 | Manipulation tasks |
| Network | Local WiFi (robot ↔ industrial computer communication) | WebSocket/HTTP API |

**Camera mounting**: Primary camera at 1.5 m above ground level; secondary camera at manipulator end-effector.

---

## Software Dependencies

### System Environment
- Ubuntu 20.04 / 22.04
- Python 3.10+
- CUDA 11.8+ with cuDNN
- ROS Noetic (for SLAM & navigation stack)
- Robot middleware (Agilex SDK with rosbridge_server for WebSocket)

### Python Packages

```bash
# Core deep learning
torch>=2.0.0
torchvision>=0.15.0

# Vision models
ultralytics              # YOLO-World (yolov8x-worldv2)
transformers
diffusers                # Stable Diffusion 3.5 Medium
bitsandbytes             # 4-bit quantization for diffusion model

# Perception & processing
pyrealsense2             # Intel RealSense SDK
opencv-python>=4.8.0
numpy
scipy
scikit-learn             # DBSCAN clustering
open3d                   # 3D point cloud operations
h5py                     # HDF5 feature storage
Pillow
matplotlib

# Communication
websocket-client         # Robot WebSocket communication
requests                 # Robot HTTP API

# Utilities
psutil
tqdm
kneed                    # Knee point detection for adaptive clustering

# Manipulation (optional)
franky                   # Franka robot control library
```

### Pre-trained Models

| Model | Purpose | Storage |
|-------|---------|----------|
| DINOv2 ViT-L/14 with registers | Patch-level visual feature extraction for cognitive map | `~/.cache/torch/hub/facebookresearch_dinov2_main` |
| YOLO-World X (YOLOv8x-worldv2) | Open-vocabulary object detection for landmark memory | `yolov8x-worldv2.pt` (project root) |
| Stable Diffusion 3.5 Medium | Text-to-image generation for association-enhanced retrieval | `stabilityai/stable-diffusion-3.5-medium` (HuggingFace) |
| CLIP ViT-H/14 (MetaCLIP FullCC) | Text-image semantic matching for target verification | (loaded via OpenCLIP) |

### External APIs
- OpenAI GPT-4 / GPT-4o (for landmark memory retrieval reasoning & target verification)

---

## Installation

```bash
# Clone repository and switch to phy branch
git clone https://github.com/Heathcliff-saku/BSC-Nav.git
cd BSC-Nav
git checkout phy

# Create conda environment
conda create -n bscnav python=3.10 -y
conda activate bscnav

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install pyrealsense2 opencv-python numpy scipy scikit-learn open3d h5py \
    Pillow matplotlib ultralytics transformers diffusers bitsandbytes \
    websocket-client requests psutil tqdm kneed openai

# Download DINOv2 model (cache locally for offline use)
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')"

# Download YOLO-World weights
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x-worldv2.pt

# Download Stable Diffusion 3.5 Medium (requires HuggingFace token)
python -c "
from diffusers import StableDiffusion3Pipeline
StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-medium', torch_dtype=torch.bfloat16)
"
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BSC-Nav Physical System                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   camera.py  │    │  datacollector.py │    │ memory_creater.py│  │
│  │  RealSense   │───▶│  RGB-D + Pose    │───▶│  Cognitive Map   │  │
│  │  D435i       │    │  Synchronization  │    │  + Landmark Mem  │  │
│  └──────────────┘    └──────────────────┘    └──────────────────┘  │
│         │                                             │             │
│         │            ┌──────────────────┐             │             │
│         │            │    memory.py     │◀────────────┘             │
│         │            │  Memory Retrieval│                           │
│         │            │  + Diffusion     │                           │
│         │            └────────┬─────────┘                           │
│         │                     │                                     │
│         ▼                     ▼                                     │
│  ┌──────────────┐    ┌──────────────────┐                           │
│  │  client.py   │    │    Agent.py      │                           │
│  │  WS + HTTP   │◀──▶│  Navigation      │                           │
│  │  Robot API   │    │  Agent (NavAgent)│                           │
│  └──────┬───────┘    └────────┬─────────┘                           │
│         │                     │                                     │
│         ▼                     ▼                                     │
│  ┌──────────────┐    ┌──────────────────┐                           │
│  │ lowlevel.py  │    │Agent_client.py / │                           │
│  │ Navigation   │    │Agent_grasp.py    │                           │
│  │ Controller   │    │  Entry Points    │                           │
│  └──────────────┘    └──────────────────┘                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐   ┌──────────────────┐
│  Agilex Chassis  │   │   Franka Arm     │
│  (rosbridge WS)  │   │   (franky lib)   │
└──────────────────┘   └──────────────────┘
```

---

## Deployment Pipeline

### Phase 1: Environment Setup & SLAM Mapping

**Objective**: Build a 2D occupancy grid map of the environment using LiDAR-based SLAM.

1. **Power on the robot** and ensure all sensors (LiDAR, IMU, cameras) are operational.

2. **Launch the SLAM system** via the robot's built-in middleware:
   - The Agilex Ranger-mini uses a 32-beam LiDAR for 2D SLAM.
   - Access the robot's web interface or use the WebSocket API to initiate mapping.

3. **Teleoperate the robot** through the entire environment to build the occupancy grid:
   ```python
   from client import WSClient, HttpClient

   ws_url = "ws://192.168.1.102:9090"   # Robot's WebSocket endpoint
   http_url = "http://192.168.1.102/apiUrl"  # Robot's HTTP API

   ws = WSClient(ws_url, "mapper")
   ws.start_heartbeat_timer(2)
   ws.mapping_2d(idtype="start", filename="my_environment")
   # Teleoperate robot through the environment...
   ws.mapping_2d(idtype="stop", filename="my_environment")
   ```

4. **Save the map**: The SLAM system outputs a `.png` occupancy grid map and associated metadata (resolution, origin). The map is retrieved via:
   ```python
   http = HttpClient(http_url)
   http.login_()
   http.get_map_info("my_environment")  # Get map metadata
   http.get_map_png("my_environment")   # Download map PNG to ./map_source/
   ```

---

### Phase 2: Data Collection (Spatial Memory Pre-construction)

**Objective**: Collect synchronized RGB-D images and robot poses by teleoperating the robot through the environment.

1. **Connect the camera and navigation controller**:
   ```python
   from camera import OptimizedRealSenseCamera
   from lowlevel import RobotNavigationController
   from datacollector import OptimizedRealTimeDataCollector

   # Initialize camera (device_index depends on which D435i)
   camera = OptimizedRealSenseCamera(
       device_index=1,
       warmup_frames=30,
       enable_filters=True,
       depth_preset='high_accuracy'
   )

   # Initialize robot controller
   nav = RobotNavigationController(
       ws_url="ws://192.168.1.102:9090",
       http_url="http://192.168.1.102/apiUrl",
       map_name="my_environment"
   )

   # Start navigation (loads the SLAM map for localization)
   nav.start_navigation(
       map_name="my_environment",
       initial_pose=(716, 887, 0)  # (x_png, y_png, theta_degrees)
   )
   ```

2. **Initialize and start data collection**:
   ```python
   collector = OptimizedRealTimeDataCollector(
       camera=camera,
       nav_controller=nav,
       save_dir="realtime_mapping_data",
       collection_interval=0.5,   # Capture every 0.5s
       max_time_diff=0.1,         # Max pose-image time desync
       blur_threshold=100.0,      # Reject blurry frames
       enable_quality_filter=True
   )
   collector.start_collection()
   ```

3. **Teleoperate the robot** through the environment at walking pace. The collector saves:
   - `color/frame_XXXXXX.png` — RGB images
   - `depth/frame_XXXXXX.png` — 16-bit depth images (millimeters)
   - `poses/frame_XXXXXX.json` — Robot pose (position x/y, orientation quaternion, yaw)

4. **Stop collection** with `Ctrl+C`. Summary statistics are printed.

**Important notes**:
- The initial_pose `(x_png, y_png, theta)` is specified in the **PNG pixel coordinate system** of the SLAM map. You need to visually identify the robot's starting position on the map image.
- The depth valid range is filtered to [0.3, 8.0] m.
- Blurry frames (Laplacian variance < threshold) are automatically rejected.

---

### Phase 3: Spatial Memory Construction

**Objective**: Process collected RGB-D data to build the cognitive map (3D voxelized visual features) and landmark memory (detected objects with 3D positions).

```python
from memory_creater import PhysicalCognitiveMapBuilder, CameraIntrinsics
from scipy.spatial.transform import Rotation as R
import os

# Define camera intrinsics (from RealSense factory calibration)
camera_intrinsics = CameraIntrinsics(
    fx=607.965,
    fy=607.875,
    ppx=428.058,
    ppy=245.646,
    width=848,
    height=480
)

# Define camera-to-base transform
# Camera mounted at 1.5m height, facing forward
pitch_angle = 0  # degrees
camera_rotation = R.from_euler('y', pitch_angle, degrees=True)
optical_to_ros = R.from_euler('xyz', [-90, 0, -90], degrees=True)
combined_rotation = camera_rotation * optical_to_ros

camera_to_base_transform = {
    'translation': [-0.1, 0.0, -1.35],  # [x, y, z] offset from base_link
    'rotation': combined_rotation.as_quat().tolist()  # [qx, qy, qz, qw]
}

# Initialize builder
builder = PhysicalCognitiveMapBuilder(
    camera_intrinsics=camera_intrinsics,
    camera_to_base_transform=camera_to_base_transform,
    save_path="memory/my_environment",
    device='cuda:0',
    config={
        'cell_size': 0.2,           # Voxel resolution (meters)
        'grid_size': 2000,          # Grid dimension (covers 400×400 m²)
        'depth_range': {'min': 0.3, 'max': 8.0},
        'depth_sample_rate': 200,   # Subsample 1 per 200 depth points
        'query_height': 252,        # DINOv2 input height
        'query_width': 448,         # DINOv2 input width
        'patch_size': 14,           # DINOv2 patch size
        'dino_model': 'dinov2_vitl14_reg',
        'yolo_model': 'yolov8x-worldv2.pt',
        'detect_classes': [
            'chair', 'table', 'sink', 'couch', 'plant',
            'vending machine', 'trash bin', 'tv', 'kitchen island',
            'book', 'lamp', 'door', 'refrigerator', 'bed',
            'toilet', 'microwave'
        ],
        'detect_conf': 0.5,         # YOLO detection confidence
        'token_dim': 1024,          # DINOv2 feature dimension
        'cache_size': 10,           # Buffer capacity per voxel
        'floor_height': -15.0,      # Min height (meters)
        'map_height': 15,           # Max height (meters)
        'sensor_height': 1.5        # Camera mount height
    }
)

# Process collected data
session_dir = "realtime_mapping_data/realtime_session_XXXXXXXX_XXXXXX"
builder.build_from_session(session_dir)  # Processes all frames

# Save memory to disk
builder.save_memory()
```

**Output files** (saved to `memory/my_environment/`):
- `feat.h5df` — HDF5 database storing DINOv2 patch features indexed by voxel coordinates
- `grid_rgb_pos.npy` — Voxel positions (N×3 array)
- `grid_rgb.npy` — Voxel RGB colors (N×3 array)
- `weight.npy` — Voxel observation weights
- `occupied_ids.npy` — 3D occupancy grid
- `cv_map.npy` — 2D top-down visualization map
- `max_id.npy` — Total number of occupied voxels
- `long_memory.json` — Landmark memory: list of detected objects with categories, 3D coordinates, confidence scores, and descriptions

**Key parameters**:
- `cell_size=0.2` m provides adequate spatial resolution for real-world deployment
- `depth_sample_rate=200` balances computation vs. coverage (1 point per 200 sampled from depth map)
- Features are stored with a **surprise-driven update**: new features are only added to a voxel if their cosine distance to existing neighborhood features exceeds 0.5

---

### Phase 4: Navigation Task Execution

**Objective**: Given a navigation goal (text, image, or category), autonomously retrieve candidate targets from spatial memory and navigate to them.

#### Option A: Interactive Navigation Interface

```bash
python Agent_client.py
```

This launches an interactive CLI with options:
1. **Text Navigation** — Describe the target in natural language
2. **Image Navigation** — Provide a query image path
3. **Category Navigation** — Specify an object category

#### Option B: Programmatic Usage

```python
from Agent import NavAgent

# Initialize the navigation agent
agent = NavAgent(
    ws_url="ws://192.168.1.102:9090",
    http_url="http://192.168.1.102/apiUrl",
    map_name="my_environment",
    memory_path="memory/my_environment",
    initial_pose=(716, 887, 0),     # Robot start position in PNG coords
    camera_device_index=1,
    load_memory=True,
    load_diffusion=True             # Needed for text/TIN navigation
)

# Text-Instance Navigation
agent.navigate_to_text("A yellow armchair near the bookshelf")

# Image-Instance Navigation
from PIL import Image
query_img = Image.open("query_images/target_chair.jpg")
agent.navigate_to_image(query_img)

# Category Navigation (Object-Goal Navigation)
agent.navigate_to_category("refrigerator")
```

#### Navigation Execution Flow

1. **Working memory retrieval**:
   - For **category goals**: Query landmark memory via GPT-4 reasoning → returns candidate 3D coordinates
   - For **text instance goals**: Augment description with GPT-4o → generate visual prototype via Stable Diffusion 3.5 → encode with DINOv2 → match against cognitive map features → DBSCAN clustering → returns candidate coordinates
   - For **image goals**: Encode query image with DINOv2 → match against cognitive map → returns candidates

2. **Candidate ranking**: Composite score = λ × confidence + (1-λ) × (1 - distance/max_distance), with λ=0.5

3. **Low-level planning**: Convert 3D candidate coordinates to 2D map coordinates → find navigable point on occupancy grid → send navigation goal via A*/TEB planner

4. **Goal verification**: Upon arrival at each candidate:
   - Perform 360° rotation scan
   - Compute CLIP similarity between observations and goal
   - GPT-4o verifies target presence and generates affordance actions

5. **Success/Failure**: If verification passes → task complete. If all candidates fail → task failed.

**Runtime parameters**:
- Navigation timeout: 300 s per candidate
- Goal reached threshold: 0.5 m (stable for 5.0 s)
- Maximum candidates: K=3 (landmark memory) + Q=3 (cognitive map)
- Robot speed: 0.2 m/s linear, 0.3 rad/s angular

---

### Phase 5: Navigation with Manipulation (Optional)

**Objective**: Combine navigation with robotic arm manipulation for mobile manipulation tasks.

```bash
python Agent_grasp.py
```

This extends `Agent_client.py` with Franka arm control via the `franky` library. After reaching the target via navigation, the system:
1. GPT-4 decomposes natural language instructions into waypoint-action sequences
2. At each waypoint, executes manipulation primitives (grasp, place, pour)
3. Uses end-effector camera for fine-grained alignment

---

## File Descriptions

| File | Description |
|------|-------------|
| `camera.py` | `OptimizedRealSenseCamera` class: threaded D435i capture with depth filtering, motion blur detection, and frame synchronization |
| `client.py` | Low-level WebSocket (`WSClient`) and HTTP (`HttpClient`) clients for robot API communication; coordinate transform utilities (PNG ↔ map coordinates) |
| `lowlevel.py` | `RobotNavigationController`: high-level wrapper for robot navigation control, pose tracking, goal sending, and status monitoring via ROS topics |
| `datacollector.py` | `OptimizedRealTimeDataCollector`: synchronized RGB-D + pose data collection with async I/O, quality filtering, and performance monitoring |
| `memory_creater.py` | `PhysicalCognitiveMapBuilder`: processes RGB-D frames to build 3D voxelized cognitive map (DINOv2 features) and landmark memory (YOLO-World detections) with surprise-driven updates |
| `memory.py` | `Memory` class (extends `PhysicalCognitiveMapBuilder`): adds memory loading, retrieval (voxel localization, diffusion-based imagination, DBSCAN clustering), and GPT integration |
| `Agent.py` | `NavAgent`: top-level navigation agent integrating memory retrieval, path planning, navigable point search, goal verification, and trajectory recording |
| `Agent_client.py` | Interactive CLI entry point for navigation tasks (text/image/category) without manipulation |
| `Agent_grasp.py` | Interactive CLI entry point for navigation + manipulation tasks (uses Franka arm via `franky`) |
| `checker.py` | Trajectory visualization tools for session replay and analysis |
| `utils.py` | Utility functions: depth-to-pointcloud projection, coordinate transforms, camera matrix computation, adaptive DBSCAN clustering |
| `test.py` | Quick YOLO-World model loading test |

---

## Configuration Reference

### Navigation Configuration (`Agent_client.py` / `Agent_grasp.py`)

```python
NAVIGATION_CONFIG = {
    "ws_url": "ws://192.168.1.102:9090",     # Robot WebSocket address
    "http_url": "http://192.168.1.102/apiUrl", # Robot HTTP API address
    "map_name": "my_environment",              # SLAM map name
    "memory_path": "memory/my_environment",    # Pre-built spatial memory
    "initial_pose": (716, 887, 0),             # (x_png, y_png, theta_deg)
    "camera_device_index": 1                   # RealSense device index
}
```

### Cognitive Map Parameters

| Parameter | Simulation | Real-World | Description |
|-----------|-----------|------------|-------------|
| `cell_size` | 0.1 m | 0.2 m | Voxel spatial resolution |
| `grid_size` | 1000 | 2000 | Grid dimension |
| `depth_sample_rate` | 1000 | 200 | Depth subsampling rate |
| `query_height × width` | 224×224 | 252×448 | DINOv2 input resolution |
| `detect_conf` | 0.55 | 0.5 | YOLO detection threshold |
| `cache_size` (B) | 10 | 10 | Per-voxel feature buffer |
| Surprise threshold (τ) | 0.5 | 0.5 | Feature novelty threshold |

### Navigation Agent Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `navigation_timeout` | 300 s | Per-goal timeout |
| `position_threshold` | 0.5 m | Goal reached distance |
| `stable_duration` | 5.0 s | Time to confirm arrival |
| `search_radius` | 40 px | Initial navigable point search radius |
| `max_search_radius` | 100 px | Maximum search radius |
| `safe_distance` | 10 px | Obstacle clearance margin |
| `obstacle_threshold` | 100 | Grayscale value for obstacle |

---

## Coordinate Systems

The system uses multiple coordinate frames:

1. **World/Map Coordinate** (meters): Origin at SLAM initialization point; used by the robot's internal navigation stack (ROS `map` frame).

2. **PNG Pixel Coordinate**: Origin at top-left of the occupancy grid map image. Related to world coordinates by:
   ```
   png_x = (world_x - originX) / resolution
   png_y = gridHeight - (world_y - originY) / resolution
   ```

3. **Voxel Grid Coordinate**: Origin at grid center. Related to world coordinates by:
   ```
   grid_row = grid_size/2 + int(world_y / cell_size)
   grid_col = grid_size/2 + int(world_x / cell_size)
   grid_h   = int(world_z / cell_size)
   ```

4. **Camera Coordinate**: Standard pinhole model (z-forward, x-right, y-down). Transformed to base_link via the fixed `camera_to_base_transform`.

---

## Troubleshooting

### Common Issues

**1. "No RealSense devices detected"**
- Check USB 3.0 connection
- Run `rs-enumerate-devices` to verify device visibility
- Try different `device_index` values (0 or 1)

**2. Navigation goal fails / robot stuck**
- The local planner's costmap inflation may block narrow passages
- Check that `initial_pose` is correctly specified (compare robot position on map)
- SLAM drift can cause coordinate misalignment — re-calibrate initial pose

**3. Memory construction produces empty/sparse features**
- Verify depth data quality: check for excessive depth holes in collected data
- Ensure camera intrinsics match the actual device calibration
- Verify the `camera_to_base_transform` is correct (incorrect transform causes features to be projected to wrong locations)

**4. Target retrieval returns incorrect locations**
- For text queries: check that Stable Diffusion generates reasonable visual prototypes
- Increase `imaginary_num` (default 3) for more diverse visual queries
- Verify `long_memory.json` contains expected detected objects

**5. GPU out of memory**
- Reduce `preload_features=False` (slower retrieval, less VRAM)
- Use smaller DINOv2 model (e.g., `dinov2_vitb14_reg`, 768-dim features)
- Enable CPU offloading for diffusion model (already configured by default)

**6. WebSocket connection failed**
- Verify robot IP address is reachable: `ping 192.168.1.102`
- Ensure `rosbridge_server` is running on the robot
- Check that the robot's navigation middleware is started

### Known Failure Modes

- **Impassable regions**: Costmap inflation blocks narrow passages → robot times out
- **Calibration drift**: Accumulated SLAM drift over long trajectories causes coordinate offset
- **Dynamic disturbances**: Moved furniture or pedestrians invalidate spatial memory

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ruan2025bscnav,
  title={From reactive to cognitive: Brain-inspired spatial intelligence for embodied agents},
  author={Ruan, Shouwei and Wang, Liyuan and Kang, Caixin and Zhu, Qihui and Liu, Songming and Wei, Xingxing and Su, Hang},
  journal={},
  year={2025}
}
```

---

## License

Please refer to the main repository for license information.
