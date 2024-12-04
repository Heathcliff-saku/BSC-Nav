from PIL import Image
import cv2
from habitat_sim.utils.common import d3_40_colors_rgb
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def show_obs(obs):
    # 获取 RGB 图像并转换为 BGR 格式
    bgr = cv2.cvtColor(obs["color_sensor"], cv2.COLOR_RGB2BGR)

    # 获取深度图像，并归一化到 0-255 范围以便显示
    depth = obs["depth_sensor"]
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # 获取语义图像，并将其转换为彩色图像
    semantic_obs = obs["semantic_sensor"]
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_colored = np.array(semantic_img.convert("RGB"))
    semantic_colored = cv2.cvtColor(semantic_colored, cv2.COLOR_RGB2BGR)

    # 将三种图像调整为相同大小
    h, w = bgr.shape[:2]
    depth_resized = cv2.resize(depth_normalized, (w, h))
    semantic_resized = cv2.resize(semantic_colored, (w, h))

    # 合并 RGB、深度和语义图像
    combined_image = np.hstack((bgr, cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR), semantic_resized))

    # 调整窗口大小，使其小于屏幕宽度
    screen_width = 1600  # 假设屏幕宽度为 800 像素
    scale_factor = min(screen_width / combined_image.shape[1], 1.0)
    new_width = int(combined_image.shape[1] * scale_factor)
    new_height = int(combined_image.shape[0] * scale_factor)
    combined_image_resized = cv2.resize(combined_image, (new_width, new_height))

    # 显示拼接图像
    cv2.imshow("RGB | Depth | Semantic", combined_image_resized)

def keyboard_control_fast():
    k = cv2.waitKey(1)
    if k == ord("a"):
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("w"):
        action = "move_forward"
    elif k == ord("q"):
        action = "stop"
    elif k == ord(" "):
        return k, "record"
    elif k == -1:
        return k, None
    else:
        return -1, None
    return k, action

def keyboard_control_nav():
    k = cv2.waitKey(1)
    if k == ord("a"):
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("w"):
        action = "move_forward"
    elif k == ord("q"):
        action = "stop"
    elif k == ord("r"):
        action = "nav"
    elif k == ord(" "):
        return k, "record"
    elif k == -1:
        return k, None
    else:
        return -1, None
    return k, action


def plot_token_matching(query_img, ref_img, similarities_2d):
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')  # 关闭坐标轴
    
    axes[1].imshow(ref_img)
    axes[1].set_title("Ref Image")
    axes[1].axis('off')  # 关闭坐标轴

    # 显示相似度分布
    im = axes[2].imshow(similarities_2d.cpu().numpy(), cmap='inferno')
    axes[2].set_title("Similarity Map")
    axes[2].axis('off')  # 关闭坐标轴
    fig.colorbar(im, ax=axes[1])  # 添加颜色条

    plt.tight_layout()
    plt.show()
    


def cvt_pose_vec2tf(pos_quat_vec: np.ndarray) -> np.ndarray:
    """
    pos_quat_vec: (px, py, pz, qx, qy, qz, qw)
    """
    pose_tf = np.eye(4)
    pose_tf[:3, 3] = pos_quat_vec[:3].flatten()
    rot = R.from_quat(pos_quat_vec[3:].flatten())
    pose_tf[:3, :3] = rot.as_matrix()
    return pose_tf


def get_sim_cam_mat(h,w):
    
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / 2.0
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat


def depth2pc(depth, fov=90, intr_mat=None, min_depth=0.1, max_depth=10):
    """
    Return 3xN array and the mask of valid points in [min_depth, max_depth]
    """

    h, w = depth.shape

    cam_mat = intr_mat
    if intr_mat is None:
        cam_mat = get_sim_cam_mat_with_fov(h, w, fov)
    # cam_mat[:2, 2] = 0
    cam_mat_inv = np.linalg.inv(cam_mat)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = x.reshape((1, -1))[:, :] + 0.5
    y = y.reshape((1, -1))[:, :] + 0.5
    z = depth.reshape((1, -1))[:, :]

    p_2d = np.vstack([x, y, np.ones_like(x)])
    pc = cam_mat_inv @ p_2d
    pc = pc * z
    mask = pc[2, :] > min_depth

    mask = np.logical_and(mask, pc[2, :] < max_depth)
    # pc = pc[:, mask]
    return pc, mask


def get_sim_cam_mat_with_fov(h, w, fov):
    cam_mat = np.eye(3)
    cam_mat[0, 0] = cam_mat[1, 1] = w / (2.0 * np.tan(np.deg2rad(fov / 2)))
    cam_mat[0, 2] = w / 2.0
    cam_mat[1, 2] = h / 2.0
    return cam_mat


def transform_pc(pc, pose):
    """
    pose: the pose of the camera coordinate where the pc is in
    """
    # pose_inv = np.linalg.inv(pose)

    pc_homo = np.vstack([pc, np.ones((1, pc.shape[1]))])

    pc_global_homo = pose @ pc_homo

    return pc_global_homo[:3, :]

def base_pos2grid_id_3d(gs, cs, x_base, y_base, z_base):
    row = int(gs / 2 - int(x_base / cs))
    col = int(gs / 2 - int(y_base / cs))
    h = int(z_base / cs)
    return [row, col, h]


def project_point(cam_mat, p):
    new_p = cam_mat @ p.reshape((3, 1))
    z = new_p[2, 0]
    new_p = new_p / new_p[2, 0]
    x = int(new_p[0, 0] - 0.5)
    y = int(new_p[1, 0] - 0.5)
    return x, y, z

    