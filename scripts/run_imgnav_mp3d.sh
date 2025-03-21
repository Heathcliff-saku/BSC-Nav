#!/bin/bash

PYTHON_SCRIPT="imagenav_benchmark.py"  # 替换为您实际的Python文件名
LOG_FILE="run_imgnav_mp3d.txt"
MAX_RETRIES=30
RETRY_WAIT=5  # 失败后等待重启的秒数

retry_count=0
while [ $retry_count -lt $MAX_RETRIES ]; do
    # 添加日期时间戳到日志
    echo "================ 第 $((retry_count+1)) 次尝试，开始于 $(date) =================" >> "$LOG_FILE"
    
    # 运行Python脚本，它会自动检测CSV文件并从正确的位置开始
    python -u "$PYTHON_SCRIPT" --no_record --no_vis --load_single_floor \
    --dataset 'mp3d' --benchmark_dataset 'mp3d' \
    --MP3D_EPISODE_PREFIX '/home/orbit/桌面/Nav-2025/data_episode/imagenav/pointnav_mp3d_v1/val/val.json.gz' \
    --MP3D_CONFIG_PATH '/home/orbit/桌面/Nav-2025/third-party/habitat-lab/habitat-lab/habitat/config/benchmark/nav/imagenav/imagenav_mp3d.yaml' \
    --dataset_dir '/home/orbit/桌面/Nav-2025/data/mp3d/mp3d_habitat/mp3d' \
    --scene_dataset_config_file /home/orbit/桌面/Nav-2025/data/mp3d/mp3d_habitat/mp3d/mp3d.scene_dataset_config.json \
    --scene_name '5LpN3gDmAk7' \
    --eval_episodes 1000 "$@" >> "$LOG_FILE" 2>&1
    
    # 检查脚本退出状态
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "运行成功完成!"
        break
    else
        retry_count=$((retry_count+1))
        echo "运行出错，退出代码: $EXIT_CODE"
        
        if [ $retry_count -lt $MAX_RETRIES ]; then
            echo "将在 $RETRY_WAIT 秒后重试... (尝试 $retry_count/$MAX_RETRIES)"
            sleep $RETRY_WAIT
        else
            echo "达到最大重试次数。请检查日志文件: $LOG_FILE"
        fi
    fi
done