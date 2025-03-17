import open3d as o3d
import numpy as np



def _visualize_rgb_map_3d(pc: np.ndarray, rgb: np.ndarray, best, center):
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
    
    highlight_pcd_2 = o3d.geometry.PointCloud()
    highlight_pcd_2.points = o3d.utility.Vector3dVector(center)
    highlight_color_2 = np.array([[1.0, 0.0, 0.0]] * len(center))  # 红色
    highlight_pcd_2.colors = o3d.utility.Vector3dVector(highlight_color_2)
    
    # 使用 Visualizer 来设置点大小
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(highlight_pcd)
    vis.add_geometry(highlight_pcd_2)
    
    # 获取渲染选项并设置点大小
    render_option = vis.get_render_option()
    render_option.point_size = 8.0  # 设置普通点大小
    render_option.line_width = 8.0  # 设置线宽
    # render_option.background_color = np.array([0, 0, 0])  # 设置背景颜色为黑色S

    # 渲染
    vis.run()
    vis.destroy_window()


grid_pos = np.load("/home/orbit/桌面/Nav-2025/memory/imgnav/hm3d/00821-eF36g7L6Z9M_island_0/grid_rgb_pos_floor_0.npy")
grid_rgb = np.load("/home/orbit/桌面/Nav-2025/memory/imgnav/hm3d/00821-eF36g7L6Z9M_island_0/grid_rgb_floor_0.npy")
best = np.load('''/home/orbit/桌面/Nav-2025/memory/imgnav/hm3d/00821-eF36g7L6Z9M_island_0/best_pos_centers_chair.npy''')
center = np.load('''/home/orbit/桌面/Nav-2025/memory/imgnav/hm3d/00821-eF36g7L6Z9M_island_0/best_pos_topK_chair.npy''')
# best = np.array([[
#             380,
#             597,
#             115
#         ]])
# center = best

_visualize_rgb_map_3d(grid_pos, grid_rgb, best, center)