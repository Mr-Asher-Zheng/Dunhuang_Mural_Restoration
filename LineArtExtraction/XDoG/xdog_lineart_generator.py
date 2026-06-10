from XDoG import *

# 读取训练集和验证集图片
train_images_dir = r"D:\作业\毕业设计\DH\image_mask\train\images"
val_images_dir = r"D:\作业\毕业设计\DH\image_mask\val\images"

# # 创建保存线条图的目录
# train_lines_dir = r"D:\作业\毕业设计\DH\image_mask\train\lines"
# val_lines_dir = r"D:\作业\毕业设计\DH\image_mask\val\lines"
# os.makedirs(train_lines_dir, exist_ok=True)
# os.makedirs(val_lines_dir, exist_ok=True)


# train_lines_dir1 = r'D:\作业\毕业设计\DH\line1\train'
# val_lines_dir1 = r'D:\作业\毕业设计\DH\line1\val'
# os.makedirs(train_lines_dir1, exist_ok=True)
# os.makedirs(val_lines_dir1, exist_ok=True)
# train_lines_dir2 = r"D:\作业\毕业设计\DH\line2\train"
# val_lines_dir2 = r"D:\作业\毕业设计\DH\line2\val"
# os.makedirs(train_lines_dir1, exist_ok=True)
# os.makedirs(val_lines_dir1, exist_ok=True)

gt_save_dir = r"D:\作业\毕业设计\DH\kaggle\data\gt"
line1_dir = r"D:\作业\毕业设计\DH\kaggle\data\line1"
line2_dir = r"D:\作业\毕业设计\DH\kaggle\data\line2"
line3_dir = r"D:\作业\毕业设计\DH\kaggle\data\line3"


# gt_save_dir = "../../ImageRestoration/samples/Evaluation/LandscapeTest/gt"
# line_save_dir = "../../ImageRestoration/samples/Evaluation/LandscapeTest/line"


# 读取目录下图片并生成线条图
def line_art_extraction(input_dir, output_dir, sigma1=1.0, sigma2=1.2, phi=0.1, epsilon=0, sharpen=100):
    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg'))]
    print(all_images[:5])  # 打印前5个文件名以确认读取正确
    for img_name in tqdm(all_images, desc=f"Processing images in {input_dir}"):
        img_path = os.path.join(input_dir, img_name)
        with Image.open(img_path) as img:
            image_tensor = transform(img).unsqueeze(0)
            xdog_result = XDoGFilter(image_tensor, sigma1=sigma1, sigma2=sigma2, phi=phi, epsilon=epsilon,
                                     sharpen=sharpen)
            xdog_result_img = xdog_result.squeeze().detach().numpy()
            xdog_result_img = (xdog_result_img).astype(np.uint8)
            pil_img_xdog = Image.fromarray(xdog_result_img)
            pil_img_xdog.save(os.path.join(output_dir, img_name))


# line_art_extraction(train_images_dir, train_lines_dir)
# line_art_extraction(val_images_dir, val_lines_dir)
# line_art_extraction(gt_save_dir, line_save_dir)

# line_art_extraction(train_images_dir, train_lines_dir1, sigma1=1, sigma2=2, phi=0.005, epsilon=0, sharpen=50)
# line_art_extraction(val_images_dir, val_lines_dir1, sigma1=1, sigma2=2, phi=0.005, epsilon=0, sharpen=50)

# line_art_extraction(train_images_dir, train_lines_dir2, sigma1=1, sigma2=10, phi=0.5, epsilon=200, sharpen=100)
# line_art_extraction(val_images_dir, val_lines_dir2, sigma1=1, sigma2=10, phi=0.5, epsilon=200, sharpen=100)

# line_art_extraction(gt_save_dir, line1_dir, sigma1=1, sigma2=2, phi=0.005, epsilon=0, sharpen=50)
# line_art_extraction(gt_save_dir, line2_dir, sigma1=1, sigma2=10, phi=0.5, epsilon=200, sharpen=100)

line_art_extraction(gt_save_dir, line3_dir, sigma1=1, sigma2=10, phi=0.005, epsilon=0, sharpen=50)
