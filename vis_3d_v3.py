import open3d as o3d
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm


def cluster_best_points(best_points, eps=10.0, min_samples=3):
    """
    对best点进行聚类，找出聚类中心
    
    Args:
        best_points: 匹配点位置 (M, 3)
        eps: DBSCAN的邻域半径
        min_samples: 形成聚类的最小点数
    
    Returns:
        cluster_centers: 聚类中心列表
        cluster_info: 聚类信息字典
    """
    # 使用DBSCAN进行聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(best_points)
    labels = clustering.labels_
    
    # 计算每个聚类的中心和信息
    cluster_centers = []
    cluster_info = {}
    unique_labels = set(labels)
    unique_labels.discard(-1)  # 移除噪声点标签
    
    for label in unique_labels:
        mask = labels == label
        cluster_points = best_points[mask]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)
        cluster_info[label] = {
            'center': center,
            'size': np.sum(mask),
            'points': cluster_points
        }
    
    # 统计噪声点
    noise_count = np.sum(labels == -1)
    if noise_count > 0:
        print(f"噪声点数量: {noise_count}")
    
    return np.array(cluster_centers), cluster_info, labels


def compute_transparency_weights_from_clusters(grid_pos, cluster_centers, radius=50.0, falloff_rate=2.0):
    """
    基于聚类中心计算场景点云的透明度权重
    
    Args:
        grid_pos: 点云位置 (N, 3)
        cluster_centers: 聚类中心位置 (K, 3)
        radius: 开始变透明的半径
        falloff_rate: 透明度变化速率
    
    Returns:
        weights: 每个点的不透明度权重 (N,)
    """
    if len(cluster_centers) == 0:
        return np.ones(len(grid_pos))
    
    # 计算每个点到所有聚类中心的距离
    distances = cdist(grid_pos, cluster_centers)
    
    # 找到每个点到最近聚类中心的距离
    min_distances = np.min(distances, axis=1)
    
    # 使用sigmoid函数计算权重
    weights = 1 / (1 + np.exp(falloff_rate * (min_distances - radius) / radius))
    
    return weights


def visualize_scene_with_clusters(pc, rgb, best,
                                  transparency_radius=50.0,
                                  falloff_rate=2.0,
                                  cluster_eps=10.0,
                                  min_samples=3,
                                  cluster_size=30.0,
                                  show_cluster_spheres=True):
    """
    显示场景点云，但只显示best点的聚类中心
    
    Args:
        pc: 场景点云位置
        rgb: 场景点云颜色
        best: 需要聚类的点位置
        transparency_radius: 透明度半径
        falloff_rate: 透明度衰减率
        cluster_eps: 聚类半径
        min_samples: 形成聚类的最小点数
        cluster_size: 聚类中心点的大小
        show_cluster_spheres: 是否显示聚类球体
    """
    
    rgb = rgb / 255.0
    
    # 对best点进行聚类，获取聚类中心
    cluster_centers, cluster_info, labels = cluster_best_points(best, eps=cluster_eps, min_samples=min_samples)
    
    print(f"\n聚类统计:")
    print(f"原始点数: {len(best)}")
    print(f"聚类数量: {len(cluster_centers)}")
    for label, info in cluster_info.items():
        print(f"聚类 {label}: {info['size']} 个点")
    
    # 基于聚类中心计算场景点云的透明度
    weights = compute_transparency_weights_from_clusters(pc, cluster_centers, transparency_radius, falloff_rate)
    
    # 创建应用程序
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    
    # 创建窗口
    window = app.create_window("Scene with Cluster Centers", 1024, 768)
    
    # 创建3D场景
    scene = o3d.visualization.gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)
    window.add_child(scene)
    
    # 根据透明度权重分组场景点云
    high_opacity_mask = weights > 0.7
    medium_opacity_mask = (weights > 0.3) & (weights <= 0.7)
    low_opacity_mask = weights <= 0.3
    
    # 高不透明度场景点云
    if np.any(high_opacity_mask):
        high_pcd = o3d.geometry.PointCloud()
        high_indices = np.where(high_opacity_mask)[0]
        high_pcd.points = o3d.utility.Vector3dVector(pc[high_indices])
        high_pcd.colors = o3d.utility.Vector3dVector(rgb[high_indices])
        
        mat_high = o3d.visualization.rendering.MaterialRecord()
        mat_high.base_color = [1.0, 1.0, 1.0, 1.0]  # 完全不透明
        mat_high.shader = "defaultUnlit"
        mat_high.point_size = 4.0
        
        scene.scene.add_geometry("high_opacity", high_pcd, mat_high)
    
    # 中等透明度场景点云
    if np.any(medium_opacity_mask):
        medium_pcd = o3d.geometry.PointCloud()
        medium_indices = np.where(medium_opacity_mask)[0]
        medium_pcd.points = o3d.utility.Vector3dVector(pc[medium_indices])
        medium_pcd.colors = o3d.utility.Vector3dVector(rgb[medium_indices])
        
        mat_medium = o3d.visualization.rendering.MaterialRecord()
        mat_medium.base_color = [1.0, 1.0, 1.0, 0.5]  # 50%不透明
        mat_medium.shader = "defaultUnlit"
        mat_medium.point_size = 3.0
        
        scene.scene.add_geometry("medium_opacity", medium_pcd, mat_medium)
    
    # 低不透明度场景点云
    if np.any(low_opacity_mask):
        low_pcd = o3d.geometry.PointCloud()
        low_indices = np.where(low_opacity_mask)[0]
        low_pcd.points = o3d.utility.Vector3dVector(pc[low_indices])
        low_pcd.colors = o3d.utility.Vector3dVector(rgb[low_indices])
        
        mat_low = o3d.visualization.rendering.MaterialRecord()
        mat_low.base_color = [1.0, 1.0, 1.0, 0.2]  # 20%不透明
        mat_low.shader = "defaultUnlit"
        mat_low.point_size = 2.0
        
        scene.scene.add_geometry("low_opacity", low_pcd, mat_low)
    
    # 添加聚类中心点
    if len(cluster_centers) > 0:
        # 使用亮红色作为聚类中心的颜色
        bright_red = [1.0, 0.0, 0.0]  # 纯红色
        center_colors = np.array([bright_red] * len(cluster_centers))
        
        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector(cluster_centers)
        center_pcd.colors = o3d.utility.Vector3dVector(center_colors)
        
        mat_center = o3d.visualization.rendering.MaterialRecord()
        mat_center.base_color = [1.0, 1.0, 1.0, 1.0]
        mat_center.shader = "defaultUnlit"
        mat_center.point_size = cluster_size * 1.5  # 增大1.5倍，更显著
        
        scene.scene.add_geometry("cluster_centers", center_pcd, mat_center)
        
        # 可选：为每个聚类添加半透明球体以显示聚类范围
        if show_cluster_spheres:
            for i, (label, info) in enumerate(cluster_info.items()):
                # 计算聚类的标准差作为球体半径
                cluster_points = info['points']
                distances_to_center = np.linalg.norm(cluster_points - info['center'], axis=1)
                sphere_radius = np.mean(distances_to_center) + np.std(distances_to_center)
                
                # 创建球体
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
                sphere.translate(info['center'])
                
                # 设置球体颜色（红色，但半透明）
                sphere_color = [1.0, 0.0, 0.0, 0.2]  # 红色，20%不透明度
                sphere.paint_uniform_color([1.0, 0.0, 0.0])
                
                mat_sphere = o3d.visualization.rendering.MaterialRecord()
                mat_sphere.base_color = sphere_color
                mat_sphere.shader = "defaultUnlit"
                
                scene.scene.add_geometry(f"cluster_sphere_{label}", sphere, mat_sphere)
    
    # 设置相机
    if 'high_pcd' in locals():
        bbox = high_pcd.get_axis_aligned_bounding_box()
    else:
        all_points = o3d.geometry.PointCloud()
        all_points.points = o3d.utility.Vector3dVector(pc)
        bbox = all_points.get_axis_aligned_bounding_box()
    
    scene.setup_camera(60, bbox, bbox.get_center())
    scene.scene.set_background([0.1, 0.1, 0.1, 1.0])  # 深色背景，让红色更显著
    
    # 运行应用
    app.run()


# 主函数
if __name__ == "__main__":
    # 加载数据
    grid_pos = np.load("/home/orbit/桌面/Nav-2025/memory/objectnav/hm3d_v2/00873-bxsVRursffK_island_0/grid_rgb_pos.npy")
    grid_rgb = np.load("/home/orbit/桌面/Nav-2025/memory/objectnav/hm3d_v2/00873-bxsVRursffK_island_0/grid_rgb.npy")
    best = np.load('/home/orbit/桌面/Nav-2025/localize_results/best_pos_topK_a toilet..npy')

    print("\n显示场景与聚类中心...")
    visualize_scene_with_clusters(
        grid_pos,
        grid_rgb,
        best,
        transparency_radius=30.0,     # 透明度半径
        falloff_rate=2.0,            # 透明度衰减率
        cluster_eps=5.0,             # 聚类半径
        min_samples=3,               # 最小点数
        cluster_size=30.0,           # 聚类中心点的大小
        show_cluster_spheres=False   # 是否显示聚类范围球体
    )