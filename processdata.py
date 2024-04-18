import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 主路径
base_path = 'res'

# 获取所有文件夹
folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

# 用于存储图片名称的列表
data = []


# 处理图片，创建一个新的正方形图片并调整大小
def process_and_resize_image(image_path):
    # 打开图像
    image = Image.open(image_path)
    # 获取图像尺寸
    width, height = image.size
    # 计算新图像的大小（正方形边长）
    new_size = max(width, height)
    # 创建一个黑色的正方形背景图
    new_image = Image.new("RGB", (new_size, new_size), (0, 0, 0))
    # 计算原图像贴在新图像中心的位置
    left = (new_size - width) // 2
    top = (new_size - height) // 2
    new_image.paste(image, (left, top))
    # 调整图像大小到256x256
    new_image = new_image.resize((256, 256), Image.LANCZOS)
    # 保存图像
    new_image.save(image_path)


# 读取图片名称
for folder in folders:
    all_images = [f for f in os.listdir(folder) if f.startswith('all_')]
    main_images = [f for f in os.listdir(folder) if f.startswith('main_')]

    # 确保图片成对出现
    for all_image in all_images:
        suffix = all_image.split('_')[1]
        main_image = f'main_{suffix}'
        if main_image in main_images:
            all_image_path = os.path.join(folder, all_image)
            main_image_path = os.path.join(folder, main_image)
            process_and_resize_image(all_image_path)
            process_and_resize_image(main_image_path)
            # 在添加到列表之前去除 base_path
            relative_all_image_path = all_image_path.replace(base_path, '').lstrip('/')
            relative_main_image_path = main_image_path.replace(base_path, '').lstrip('/')
            # 使用相对路径添加数据到列表
            data.append(
                (os.path.splitext(all_image)[0], relative_all_image_path, relative_main_image_path))  # 移除.png后缀，并添加到列表

# 创建DataFrame
df = pd.DataFrame(data, columns=['text', 'image', 'sketch'])

# 分割训练集和验证集
train, valid = train_test_split(df, test_size=0.1, random_state=42)  # 90%训练集

# 可以将训练集保存到CSV文件
train.to_csv('train_dataset.csv', index=False)

print("训练集DataFrame已保存为 'train_dataset.csv'")
