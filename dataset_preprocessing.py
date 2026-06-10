import os
from PIL import Image
from tqdm import tqdm


# ==================== 图片放缩或裁剪为512x512 ====================
# 读取某个文件夹下的所有图片文件，放缩或裁剪为512x512保存到另一个文件夹
# 要求：
# 不拉伸图片
# 遍历每一张图片
# 任意一边 小于 256 直接跳过
# 若该图片分辨率非常大，如1024*1024，则可裁剪为4张512*512的图片，
# 若该图片为长方形如512*256，则裁剪为两张256*256的图片然后尽量无损放大到512*512，
# 若该图片不可能是刚刚好是512的倍数，这种情况下就尽量裁剪
def crop_images(img_path, save_dir, size=512):
    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    with Image.open(img_path) as img:
        img = img.convert("RGB")
        # 原始图片宽高
        W, H = img.size

        # 情况0：任意一边 < 64 → 直接跳过
        if W < 256 or H < 256:
            print("skipping (too small):", img_path)
            return

        # 情况1：其中一边<512 取最短边为基准裁剪成正方形然后放大
        if W < size or H < size:
            small_side = min(W, H)

            num_x = W // small_side
            num_y = H // small_side

            count = 0
            for i in range(num_y):
                for j in range(num_x):
                    x = j * small_side
                    y = i * small_side
                    crop = img.crop((x, y, x + small_side, y + small_side))
                    square = crop.resize((size, size), Image.Resampling.LANCZOS)

                    save_path = os.path.join(save_dir, f"{img_name}_crop{count}.png")
                    square.save(save_path, "PNG")
                    count += 1
            return

        # 情况2：两边均>=512且<1024 -> 取最短边为基准，在图片中心裁剪成正方形然后放大
        if max(W, H) < size * 2:
            small_side = min(W, H)

            left = (W - small_side) // 2
            top = (H - small_side) // 2

            crop = img.crop((left, top, left + small_side, top + small_side))
            crop = crop.resize((size, size), Image.Resampling.LANCZOS)

            save_path = os.path.join(save_dir, f"{img_name}_crop0.png")
            crop.save(save_path, "PNG")
            return

        # 情况3：任意一边>=1024且另一边>=512 -> 裁剪成512*512
        if W >= size * 2 or H >= size * 2:
            count = 0
            for h in range(0, H - size + 1, size):
                for w in range(0, W - size + 1, size):
                    crop = img.crop((w, h, w + size, h + size))
                    save_path = os.path.join(save_dir, f"{img_name}_crop{count}.png")
                    crop.save(save_path, "PNG")
                    count += 1
            return

        print("Unhandled case for image:", img_path)


def process(image_dir, save_dir, size=512):
    os.makedirs(save_dir, exist_ok=True)

    image_list = sorted([os.path.join(image_dir, f)
                         for f in os.listdir(image_dir) if f.endswith(('png', 'jpg'))])

    print("Total images found:", len(image_list))
    print(image_list)

    for img_path in tqdm(image_list, desc="Processing Images"):
        crop_images(img_path, save_dir, size=size)


# MuralDH不用裁剪，其数据集中的图片已经是512x512的了，所以不需要进行裁剪处理

# =========================================================================================
# Dunhuang_Grottoes_Painting
# 1、
Dunhuang_Grottoes_Painting_image_dir_1 = r"../Dunhuang_Grottoes_Painting/train/train_GT"
Dunhuang_Grottoes_Painting_save_dir_1 = r"../Dunhuang_Grottoes_Painting/train/train_GT_cropped"
process(Dunhuang_Grottoes_Painting_image_dir_1, Dunhuang_Grottoes_Painting_save_dir_1, size=512)
# 2、
Dunhuang_Grottoes_Painting_image_dir_2 = r"../Dunhuang_Grottoes_Painting/test/test_GT"
Dunhuang_Grottoes_Painting_save_dir_2 = r"../Dunhuang_Grottoes_Painting/test/test_GT_cropped"
process(Dunhuang_Grottoes_Painting_image_dir_2, Dunhuang_Grottoes_Painting_save_dir_2, size=512)
# =========================================================================================

# =========================================================================================
# DhMurals-inpainting-dataset
# 1、
DhMurals_inpainting_dataset_image_dir_1 = r"../DhMurals-inpainting-dataset/train/images"
DhMurals_inpainting_dataset_save_dir_1 = r"../DhMurals-inpainting-dataset/train/images_cropped"
process(DhMurals_inpainting_dataset_image_dir_1, DhMurals_inpainting_dataset_save_dir_1, size=512)
# 2、
DhMurals_inpainting_dataset_image_dir_2 = r"../DhMurals-inpainting-dataset/test/images"
DhMurals_inpainting_dataset_save_dir_2 = r"../DhMurals-inpainting-dataset/test/images_cropped"
process(DhMurals_inpainting_dataset_image_dir_2, DhMurals_inpainting_dataset_save_dir_2, size=512)
# 3、
DhMurals_inpainting_dataset_image_dir_3 = r"../DhMurals-inpainting-dataset/val/images"
DhMurals_inpainting_dataset_save_dir_3 = r"../DhMurals-inpainting-dataset/val/images_cropped"
process(DhMurals_inpainting_dataset_image_dir_3, DhMurals_inpainting_dataset_save_dir_3, size=512)
# =========================================================================================

# =========================================================================================
# Dunhuang_Faces
# 1、
Dunhuang_Faces_image_dir_1 = r"../Dunhuang_Faces/Dunhuang_Faces-20K/face_imgs"
Dunhuang_Faces_save_dir_1 = r"../Dunhuang_Faces/Dunhuang_Faces-20K/face_imgs_cropped"
process(Dunhuang_Faces_image_dir_1, Dunhuang_Faces_save_dir_1, size=512)
# 2、
Dunhuang_Faces_image_dir_2 = r"../Dunhuang_Faces/Dunhuang_Faces-20K/original_mural_imgs"
Dunhuang_Faces_save_dir_2 = r"../Dunhuang_Faces/Dunhuang_Faces-20K/original_mural_imgs_cropped"
process(Dunhuang_Faces_image_dir_2, Dunhuang_Faces_save_dir_2, size=512)
# =========================================================================================
