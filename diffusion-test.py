#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from PIL import Image
import argparse

def copy_high_resolution_images(source_dir, target_dir, min_width, min_height):
    """
    将分辨率大于指定值的图像从源文件夹复制到目标文件夹
    
    参数:
        source_dir (str): 源文件夹路径
        target_dir (str): 目标文件夹路径
        min_width (int): 最小宽度（像素）
        min_height (int): 最小高度（像素）
    """
    # 确保目标文件夹存在
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"创建目标文件夹: {target_dir}")
    
    # 支持的图像格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    
    # 统计数据
    total_images = 0
    copied_images = 0
    
    # 遍历源文件夹中的所有文件
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # 检查是否为图像文件
            if file_ext in supported_formats:
                total_images += 1
                try:
                    # 打开图像并获取尺寸
                    with Image.open(file_path) as img:
                        width, height = img.size
                        
                        # 检查图像分辨率是否满足条件
                        if width >= min_width or height >= min_height:
                            # 计算相对路径以保持文件夹结构
                            rel_path = os.path.relpath(root, source_dir)
                            target_subdir = os.path.join(target_dir, rel_path)
                            
                            # 确保目标子文件夹存在
                            if not os.path.exists(target_subdir):
                                os.makedirs(target_subdir)
                            
                            # 复制文件
                            target_file_path = os.path.join(target_subdir, file)
                            shutil.copy2(file_path, target_file_path)
                            copied_images += 1
                            print(f"已复制: {file} ({width}x{height})")
                            
                except Exception as e:
                    print(f"处理 {file} 时出错: {e}")
    
    print(f"\n完成! 共处理 {total_images} 个图像，复制了 {copied_images} 个高分辨率图像。")

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='将高分辨率图像从一个文件夹复制到另一个文件夹')
    parser.add_argument('--source_dir', help='源文件夹路径')
    parser.add_argument('--target_dir', help='目标文件夹路径')
    parser.add_argument('--width', type=int, default=500, help='最小宽度（像素），默认1920')
    parser.add_argument('--height', type=int, default=500, help='最小高度（像素），默认1080')
    
    args = parser.parse_args()
    
    print(f"源文件夹: {args.source_dir}")
    print(f"目标文件夹: {args.target_dir}")
    print(f"最小分辨率要求: {args.width}x{args.height} 像素")
    print("开始处理...\n")
    
    copy_high_resolution_images(args.source_dir, args.target_dir, args.width, args.height)

if __name__ == "__main__":
    main()