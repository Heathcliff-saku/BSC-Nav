import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def compute_transparency_weights(grid_pos, best_points, radius=50.0, falloff_rate=2.0):
    """
    计算每个点的透明度权重，基于到最近匹配点的距离
    距离越远，透明度越高（权重越低）
    
    Args:
        grid_pos: 点云位置 (N, 3)
        best_points: 匹配点位置 (M, 3)
        radius: 开始变透明的半径
        falloff_rate: 透明度变化速率
    
    Returns:
        weights: 每个点的不透明度权重 (N,)，范围[0, 1]，0表示完全透明
    """
    # 计算每个点到所有匹配点的距离
    distances = cdist(grid_pos, best_points)
    
    # 找到每个点到最近匹配点的距离
    min_distances = np.min(distances, axis=1)
    
    # 使用sigmoid函数计算权重，实现平滑过渡
    # 在radius内权重接近1（不透明），在radius外逐渐减少到0（透明）
    weights = 1 / (1 + np.exp(falloff_rate * (min_distances - radius) / radius))
    
    return weights


def cluster_best_points(best_points, eps=10.0, min_samples=3):
    """
    对best点进行聚类，找出聚类中心
    
    Args:
        best_points: 匹配点位置 (M, 3)
        eps: DBSCAN的邻域半径
        min_samples: 形成聚类的最小点数
    
    Returns:
        cluster_centers: 聚类中心列表
        cluster_labels: 每个点的聚类标签
    """
    # 使用DBSCAN进行聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(best_points)
    labels = clustering.labels_
    
    # 计算每个聚类的中心
    cluster_centers = []
    unique_labels = set(labels)
    unique_labels.discard(-1)  # 移除噪声点标签
    
    for label in unique_labels:
        mask = labels == label
        cluster_points = best_points[mask]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)
    
    # 如果没有聚类，使用所有点的中心
    if len(cluster_centers) == 0:
        cluster_centers = [np.mean(best_points, axis=0)]
    
    return np.array(cluster_centers), labels


def compute_gradient_colors(best_points, cluster_centers, colormap='plasma'):
    """
    根据点到最近聚类中心的距离计算渐变颜色
    
    Args:
        best_points: 匹配点位置 (M, 3)
        cluster_centers: 聚类中心 (K, 3)
        colormap: matplotlib颜色映射名称
    
    Returns:
        colors: RGB颜色数组 (M, 3)
    """
    # 计算每个点到所有聚类中心的距离
    distances = cdist(best_points, cluster_centers)
    
    # 找到每个点到最近聚类中心的距离
    min_distances = np.min(distances, axis=1)
    
    # 归一化距离到[0, 1]范围
    if np.max(min_distances) > np.min(min_distances):
        normalized_distances = (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances))
    else:
        normalized_distances = np.zeros_like(min_distances)
    
    # 使用颜色映射
    cmap = cm.get_cmap(colormap)
    colors = cmap(normalized_distances)[:, :3]  # 只取RGB，不要alpha
    
    return colors


def visualize_with_material_transparency(pc, rgb, best,
                                        transparency_radius=50.0,
                                        falloff_rate=2.0,
                                        cluster_eps=10.0,
                                        colormap='plasma'):
    """
    方法3: 使用Open3D的新渲染系统和材质透明度
    需要Open3D 0.13+版本
    
    Args:
        pc: 点云位置
        rgb: 点云颜色
        best: 强调点位置
        transparency_radius: 透明度半径
        falloff_rate: 透明度衰减率
        cluster_eps: 聚类半径
        colormap: 颜色映射 (可选: 'plasma', 'viridis', 'hot', 'spring', 'summer', 'cool')
    """

    rgb = rgb / 255.0
    
    # 计算透明度权重
    weights = compute_transparency_weights(pc, best, transparency_radius, falloff_rate)
    
    # 对best点进行聚类并计算渐变颜色
    cluster_centers, cluster_labels = cluster_best_points(best, eps=cluster_eps)
    gradient_colors = compute_gradient_colors(best, cluster_centers, colormap)
    
    # 创建应用程序
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    
    # 创建窗口
    window = app.create_window("Material Transparency with Gradient Highlights", 1024, 768)
    
    # 创建3D场景
    scene = o3d.visualization.gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(scene)
    
    # 根据透明度权重分组点云
    high_opacity_mask = weights > 0.7
    medium_opacity_mask = (weights > 0.3) & (weights <= 0.7)
    low_opacity_mask = weights <= 0.3
    
    # 高不透明度点云
    if np.any(high_opacity_mask):
        high_pcd = o3d.geometry.PointCloud()
        high_indices = np.where(high_opacity_mask)[0]
        high_pcd.points = o3d.utility.Vector3dVector(pc[high_indices])
        high_pcd.colors = o3d.utility.Vector3dVector(rgb[high_indices])
        
        mat_high = o3d.visualization.rendering.MaterialRecord()
        mat_high.base_color = [1.0, 1.0, 1.0, 1.0]  # 完全不透明
        mat_high.shader = "defaultUnlit"
        
        scene.scene.add_geometry("high_opacity", high_pcd, mat_high)
        mat_high.point_size = 8
    
    # 中等透明度点云
    if np.any(medium_opacity_mask):
        medium_pcd = o3d.geometry.PointCloud()
        medium_indices = np.where(medium_opacity_mask)[0]
        medium_pcd.points = o3d.utility.Vector3dVector(pc[medium_indices])
        medium_pcd.colors = o3d.utility.Vector3dVector(rgb[medium_indices])
        
        mat_medium = o3d.visualization.rendering.MaterialRecord()
        mat_medium.base_color = [1.0, 1.0, 1.0, 0.6]  # 60%不透明
        mat_medium.shader = "defaultUnlit"
        
        scene.scene.add_geometry("medium_opacity", medium_pcd, mat_medium)
        mat_medium.point_size = 5
    
    # 低不透明度点云
    if np.any(low_opacity_mask):
        low_pcd = o3d.geometry.PointCloud()
        low_indices = np.where(low_opacity_mask)[0]
        low_pcd.points = o3d.utility.Vector3dVector(pc[low_indices])
        low_pcd.colors = o3d.utility.Vector3dVector(rgb[low_indices])
        
        mat_low = o3d.visualization.rendering.MaterialRecord()
        mat_low.base_color = [1.0, 1.0, 1.0, 0.3]  # 30%不透明
        mat_low.shader = "defaultUnlit"
        
        scene.scene.add_geometry("low_opacity", low_pcd, mat_low)
        mat_medium.point_size = 3
    
    # 添加带渐变颜色的匹配点
    highlight_pcd = o3d.geometry.PointCloud()
    highlight_pcd.points = o3d.utility.Vector3dVector(best)
    highlight_pcd.colors = o3d.utility.Vector3dVector(gradient_colors)
    
    mat_highlight = o3d.visualization.rendering.MaterialRecord()
    mat_highlight.base_color = [1.0, 1.0, 1.0, 1.0]
    mat_highlight.shader = "defaultUnlit"
    mat_highlight.point_size = 15
    
    scene.scene.add_geometry("highlights", highlight_pcd, mat_highlight)
    
    # 添加聚类中心点（可选，用更大的点表示）
    if len(cluster_centers) > 0:
        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector(cluster_centers)
        # 聚类中心使用白色
        center_colors = np.array([[1.0, 1.0, 1.0]] * len(cluster_centers))
        center_pcd.colors = o3d.utility.Vector3dVector(center_colors)
        
        mat_center = o3d.visualization.rendering.MaterialRecord()
        mat_center.base_color = [1.0, 1.0, 1.0, 1.0]
        mat_center.shader = "defaultUnlit"
        mat_center.point_size = 30.0
        
        scene.scene.add_geometry("cluster_centers", center_pcd, mat_center)
    
    # 设置相机
    bbox = high_pcd.get_axis_aligned_bounding_box() if 'high_pcd' in locals() else o3d.geometry.AxisAlignedBoundingBox()
    scene.setup_camera(60, bbox, bbox.get_center())
    scene.scene.set_background([1.0, 1.0, 1.0, 0])
    
    # 运行应用
    app.run()


# 主函数
if __name__ == "__main__":
    # 加载数据
    grid_pos = np.load("/home/orbit/桌面/Nav-2025/memory/objectnav/hm3d_v2/00814-p53SfW6mjZe_island_0/grid_rgb_pos.npy")
    grid_rgb = np.load("/home/orbit/桌面/Nav-2025/memory/objectnav/hm3d_v2/00814-p53SfW6mjZe_island_0/grid_rgb.npy")
    best = np.load('/home/orbit/桌面/Nav-2025/localize_results/best_pos_topK_img_input.npy')

    print("\n4. Attempting material transparency with gradient highlights...")
    # 可以尝试不同的颜色映射
    # 明亮色系选项: 'plasma', 'viridis', 'hot', 'spring', 'summer', 'cool'
    visualize_with_material_transparency(
        grid_pos,
        grid_rgb,
        best,
        transparency_radius=30.0,
        falloff_rate=2.0,
        cluster_eps=5.0,  # 调整聚类半径
        colormap='hot'  # 使用plasma色彩映射（紫-粉-黄的渐变）
    )