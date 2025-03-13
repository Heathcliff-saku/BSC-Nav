import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def visualize_detection(image_path, results):
    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 设置字体
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

    for result in results:
        scores = result['scores']
        labels = result['labels']
        boxes = result['boxes']

        for score, label, box in zip(scores, labels, boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = box

            # 绘制边界框
            draw.rectangle(((x1, y1), (x2, y2)), outline="red", width=2)

            # 绘制标签和置信度
            text = f"{label}: {score:.2f}"
            text_size = draw.textbbox((x1, y1), text, font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            draw.rectangle(((x1, y1 - text_height), (x1 + text_width, y1)), fill="red")
            draw.text((x1, y1 - text_height), text, fill="white", font=font)

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()

# 使用示例
model_id = "IDEA-Research/grounding-dino-tiny"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_path = '/home/orbit-new/桌面/orbit/shouwei_GES/VLDMap/Vlmaps/DistributionMap_dynamic/vlmaps/data/vlmaps_dataset/5q7pvUzZiYa_1/rgb/000226.png'
image = Image.open(image_path)

text = "a oven. a TV. a Bookcase"
inputs = processor(images=image, text=text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.3,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

# 可视化检测结果
visualize_detection(image_path, results)