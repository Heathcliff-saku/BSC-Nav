import pandas as pd
import numpy as np

def compute_metrics(csv_file):
    # 定义列名
    column_names = ['success', 'spl', 'distance_to_goal', 'object_goal', 'id', 'island', 'island_area', 
                    'long_memory_query', 'working_memory_query', 'search_point']
    
    # 读取CSV文件（没有表头）
    df = pd.read_csv(csv_file, header=None, names=column_names)
    
    # 移除distance_to_goal列中为inf的行
    df = df[~df['distance_to_goal'].apply(np.isinf)]
    
    # 计算总的指标
    total_success_rate = df['success'].mean()
    total_avg_spl = df['spl'].mean()
    total_avg_distance_to_goal = df['distance_to_goal'].mean()

    # 按物体类别分组统计
    grouped = df.groupby('object_goal').agg(
        success_rate=('success', 'mean'),
        avg_spl=('spl', 'mean'),
        avg_distance_to_goal=('distance_to_goal', 'mean')
    ).reset_index()

    # 输出总的指标
    print(f"总的成功率: {total_success_rate:.4f}")
    print(f"总的平均 SPL: {total_avg_spl:.4f}")
    print(f"总的平均距离到目标: {total_avg_distance_to_goal:.4f}")
    
    # 输出每个物体类别的指标
    print("\n每个物体类别的指标:")
    print(grouped)

# 示例：调用函数并提供CSV文件路径
csv_file = 'objnav_hm3d_v2_results.csv'  # 这里需要指定你的CSV文件路径
compute_metrics(csv_file)
