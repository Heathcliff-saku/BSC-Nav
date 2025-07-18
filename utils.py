from PIL import Image
import cv2
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from kneed import KneeLocator

# def show_obs(obs):
#     # 获取 RGB 图像并转换为 BGR 格式
#     bgr = cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR)
#     bgr_back = cv2.cvtColor(obs["back_rgb"], cv2.COLOR_RGB2BGR) 
    
#     # 获取深度图像，并归一化到 0-255 范围以便显示
#     depth = obs["depth"]
#     depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
#     depth_normalized = depth_normalized.astype(np.uint8)

#     # 获取语义图像，并将其转换为彩色图像
#     semantic_obs = obs["semantic"]
#     semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
#     semantic_img.putpalette(d3_40_colors_rgb.flatten())
#     semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
#     semantic_colored = np.array(semantic_img.convert("RGB"))
#     semantic_colored = cv2.cvtColor(semantic_colored, cv2.COLOR_RGB2BGR)

#     # 将三种图像调整为相同大小
#     h, w = bgr.shape[:2]
#     depth_resized = cv2.resize(depth_normalized, (w, h))
#     semantic_resized = cv2.resize(semantic_colored, (w, h))

#     # 合并 RGB、深度和语义图像
#     combined_image = np.hstack((bgr, bgr_back, cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR), semantic_resized))

#     # 调整窗口大小，使其小于屏幕宽度
#     screen_width = 1600  # 假设屏幕宽度为 800 像素
#     scale_factor = min(screen_width / combined_image.shape[1], 1.0)
#     new_width = int(combined_image.shape[1] * scale_factor)
#     new_height = int(combined_image.shape[0] * scale_factor)
#     combined_image_resized = cv2.resize(combined_image, (new_width, new_height))

#     # 显示拼接图像
#     cv2.imshow("RGB1 | RGB2 | Depth | Semantic", combined_image_resized)

def keyboard_control_fast():
    k = cv2.waitKey(1)
    if k == ord("a"):
        action = "turn_left"
    elif k == ord("d"):
        action = "turn_right"
    elif k == ord("w"):
        action = "move_forward"
    elif k == ord("s"):
        action = "move_backward"
    elif k == ord("q"):
        action = "stop"
    elif k == ord(" "):
        return k, "record"
    elif k == -1:
        return k, None
    else:
        return -1, None
    return k, action

def keyboard_control_objectnav():
    k = cv2.waitKey(1)
    if k == ord("a"):
        action = 2
    elif k == ord("d"):
        action = 3
    elif k == ord("w"):
        action = 1
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
    elif k == ord("s"):
        action = "move_backward"
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
    # 修正版本：保持坐标系一致性
    row = int(gs / 2 + int(y_base / cs))  
    col = int(gs / 2 + int(x_base / cs))  
    h = int(z_base / cs)
    return [row, col, h]

def grid_id_3d2base_pos(gs, cs, best_pos):
    """
    将网格坐标批量还原为真实坐标
    
    参数:
    gs: 网格大小 (grid size)
    cs: 单元格大小 (cell size)  
    best_pos: numpy数组，形状为[N, 3]，每行为[row, col, h]
    
    返回:
    numpy数组，形状为[N, 3]，每行为[x_base, y_base, z_base]
    """
    # 确保输入是numpy数组
    best_pos = np.array(best_pos)
    
    # 提取行、列、高度索引
    rows = best_pos[:, 0]  # 所有行的第0列
    cols = best_pos[:, 1]  # 所有行的第1列
    hs = best_pos[:, 2]    # 所有行的第2列
    
    # 批量计算真实坐标
    x_base = (cols - gs / 2) * cs
    y_base = (rows - gs / 2) * cs
    z_base = hs * cs
    
    # 组合结果
    result = np.column_stack([x_base, y_base, z_base])
    return result


def project_point(cam_mat, p):
    new_p = cam_mat @ p.reshape((3, 1))
    z = new_p[2, 0]
    new_p = new_p / new_p[2, 0]
    x = int(new_p[0, 0] - 0.5)
    y = int(new_p[1, 0] - 0.5)
    return x, y, z




def adaptive_clustering(points, confidences, visualize=False):
    """
    对3D坐标点进行自适应聚类，并计算每个簇的平均坐标和置信度
    
    参数:
        points: numpy数组，形状为[N, 3]，表示N个点的xyz坐标
        confidences: numpy数组，形状为[N]，表示每个点的置信度
        visualize: 布尔值，是否可视化聚类过程和结果
    
    返回:
        cluster_centers: numpy数组，每个簇的平均坐标
        cluster_confidences: numpy数组，每个簇的平均置信度
        labels: 每个点所属的簇标签
    """
    # 1. 自适应确定eps参数
    # 使用k近邻计算最佳eps值，采用"拐点"法
    k = min(len(points) - 1, max(5, int(np.sqrt(len(points)))))  # 自适应设置k值
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # 排序距离以便找到拐点
    dist_desc = np.sort(distances[:, -1])  # 使用到第k个近邻的距离
    
    if visualize:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(dist_desc)), dist_desc)
        plt.xlabel('Points sorted by distance')
        plt.ylabel('k-th nearest neighbor distance')
        plt.title('K-distance Graph')
        plt.grid(True)
        plt.show()
    
    # 使用KneeLocator找到拐点（如果没有明显拐点，使用启发式方法）
    try:
        kneedle = KneeLocator(range(len(dist_desc)), 
                              dist_desc, 
                              S=1.0, 
                              curve='convex', 
                              direction='increasing')
        # 使用较小的eps值以分离相近的簇
        eps = dist_desc[kneedle.knee] * 0.5 if kneedle.knee else np.median(dist_desc)
    except:
        # 如果拐点检测失败，使用启发式方法：取距离的均值和标准差
        eps = np.mean(dist_desc) + np.std(dist_desc)
    
    # 2. 自适应确定min_samples参数
    # 降低簇形成的门槛，倾向于识别更多的簇
    min_samples = max(2, min(len(points) // 20, int(np.log(len(points)) / 1.5)))
    
    print(f"自适应参数: eps={eps:.4f}, min_samples={min_samples}")
    
    # 3. 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    
    # 统计簇的数量（不包括噪声点，标签为-1）
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"识别出的簇数量: {n_clusters}")
    
    # 评估簇的数量是否合适，若太少或没找到簇则尝试调整参数
    expected_min_clusters = max(1, int(np.sqrt(len(points)) / 3))  # 基于点数量估计可能的簇数
    
    if n_clusters == 0 or (len(points) > 20 and n_clusters < expected_min_clusters):
        if n_clusters == 0:
            print("未找到簇，尝试调整参数...")
            eps *= 1.5  # 增加eps
            min_samples = max(2, min_samples - 1)  # 减小min_samples但不低于2
        else:
            print(f"检测到的簇数({n_clusters})少于预期({expected_min_clusters})，尝试调整参数以增加簇数...")
            eps *= 0.7  # 减小eps以分离更多簇
            min_samples = max(2, min_samples - 1)  # 减小min_samples但不低于2
        
        print(f"调整后的参数: eps={eps:.4f}, min_samples={min_samples}")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"调整后识别出的簇数量: {n_clusters}")
    
    # 4. 计算每个簇的平均坐标和置信度
    cluster_centers = []
    cluster_confidences = []
    
    for i in range(n_clusters):
        cluster_mask = (labels == i)
        cluster_points = points[cluster_mask]
        cluster_conf = confidences[cluster_mask]
        
        # 使用置信度加权的平均坐标
        weighted_center = np.average(cluster_points, axis=0, weights=cluster_conf)
        avg_confidence = np.mean(cluster_conf)
        
        cluster_centers.append(weighted_center)
        cluster_confidences.append(avg_confidence)
    
    # 如果存在噪声点（标签为-1），可以选择性处理
    noise_mask = (labels == -1)
    if np.any(noise_mask):
        print(f"噪声点数量: {np.sum(noise_mask)}/{len(points)}")
    
    # 可视化结果
    if visualize and n_clusters > 0:
        visualize_clusters(points, labels, np.array(cluster_centers))
    
    return np.array(cluster_centers), np.array(cluster_confidences), labels

def visualize_clusters(points, labels, centers):
    """可视化聚类结果"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原始点
    colors = plt.cm.jet(np.linspace(0, 1, len(set(labels)) - (1 if -1 in labels else 0)))
    for i, label in enumerate(set(labels)):
        if label == -1:
            # 噪声点
            ax.scatter(points[labels == label, 0], 
                       points[labels == label, 1], 
                       points[labels == label, 2], 
                       c='k', marker='o', s=10, alpha=0.3)
        else:
            # 簇内点
            ax.scatter(points[labels == label, 0], 
                       points[labels == label, 1], 
                       points[labels == label, 2], 
                       c=[colors[i]], marker='o', s=30)
    
    # 绘制簇中心
    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], 
               c='red', marker='*', s=200, edgecolor='k')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Clustering Results')
    plt.show()