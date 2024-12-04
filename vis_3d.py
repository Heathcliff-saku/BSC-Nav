import open3d as o3d
import numpy as np



def _visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray, best):
    # grid_rgb = rgb / 255.0
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # pcd.colors = o3d.utility.Vector3dVector(grid_rgb)
    # o3d.visualization.draw_geometries([pcd])
    
    rgb = rgb / 255.0
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    # 创建一个用于高亮显示的点云
    highlight_pcd = o3d.geometry.PointCloud()
    highlight_pcd.points = o3d.utility.Vector3dVector(best)
    highlight_color = np.array([[1.0, 0.0, 0.0]] * len(best))  # 红色
    highlight_pcd.colors = o3d.utility.Vector3dVector(highlight_color)
    
    # 使用 Visualizer 来设置点大小
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(highlight_pcd)
    
    # 获取渲染选项并设置点大小
    render_option = vis.get_render_option()
    render_option.point_size = 5.0  # 设置普通点大小
    render_option.line_width = 5.0  # 设置线宽
    render_option.background_color = np.array([0, 0, 0])  # 设置背景颜色为黑色

    # 渲染
    vis.run()
    vis.destroy_window()
        
    

grid_pos = np.load("/home/orbit-new/桌面/Nav-2025/memory/5q7pvUzZiYa_1_2_3_4_5_6_7_8_9_10_11/grid_pos.npy")
grid_rgb = np.load("/home/orbit-new/桌面/Nav-2025/memory/5q7pvUzZiYa_1_2_3_4_5_6_7_8_9_10_11/grid_rgb.npy")
best = np.load('best_pos.npy')

_visualize_rgb_map_3d(grid_pos, grid_rgb, best)