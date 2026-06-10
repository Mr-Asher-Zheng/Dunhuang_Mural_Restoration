import os
import random
import shutil
from pathlib import Path

landscape_data_path = r'D:\作业\毕业设计\Landscape_painting_GAN\data'

gt_save_dir = "./samples/Evaluation/LandscapeTest/gt"
pred_save_dir = "./samples/Evaluation/LandscapeTest/pred"
input_save_dir = "./samples/Evaluation/LandscapeTest/input"
line_save_dir = "./samples/Evaluation/LandscapeTest/line"
mask_save_dir = "./samples/Evaluation/LandscapeTest/mask"

# 从landscape_data_path随机抽取100张图像，保存到gt_save_dir
seed = 42
random.seed(seed)

# 1. 创建所有输出文件夹（不存在则自动创建）
for dir_path in [gt_save_dir, pred_save_dir, input_save_dir, line_save_dir, mask_save_dir]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)


# 2. 获取原始数据文件夹中所有图片文件
def get_image_files(folder_path):
    """获取文件夹下所有图片文件的完整路径"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # 支持的图片格式
    image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return image_files


# 获取所有图片路径
all_images = get_image_files(landscape_data_path)
print(f"原始数据集总图片数量：{len(all_images)}")

# 3. 随机抽取100张图片
sample_num = 100
if len(all_images) < sample_num:
    raise ValueError(f"图片数量不足！需要{sample_num}张，仅找到{len(all_images)}张")

selected_images = random.sample(all_images, sample_num)
print(f"已随机抽取 {len(selected_images)} 张图片")

# 4. 将抽取的图片复制到gt_save_dir
for img_path in selected_images:
    # 获取文件名
    img_name = os.path.basename(img_path)
    # 目标保存路径
    save_path = os.path.join(gt_save_dir, img_name)
    # 复制文件
    shutil.copy2(img_path, save_path)

print(f"✅ 抽取完成！100张图片已保存至：{gt_save_dir}")
print(f"其他文件夹已创建就绪：{pred_save_dir}\n{input_save_dir}\n{line_save_dir}\n{mask_save_dir}")
