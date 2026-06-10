import os
import random
import shutil
import PIL.Image as Image
from tqdm import tqdm

# ========================= 图片放缩或裁剪为512x512 ==============================
# img_path = r"before_resize.png"
# image = Image.open(img_path)
# resized_image = image.resize((512, 512), Image.Resampling.LANCZOS)
# save_path = r"after_resize.png"
# resized_image.save(save_path)
# =============================================================================

# ==================================== 划分数据集 ========================================
# # 划分Dunhuang_Faces的数据集为训练集和验证集
# total_images_dir = r"D:\作业\毕业设计\Dunhuang_Faces\Dunhuang_Faces-20K\face_imgs_cropped"
# total_masks_dir = r"D:\作业\毕业设计\Dunhuang_Faces\Dunhuang_Faces-20K\seg_pred"
# val_images_dir = r"D:\作业\毕业设计\Dunhuang_Faces\Dunhuang_Faces-20K\face_imgs_cropped_val"
# val_masks_dir = r"D:\作业\毕业设计\Dunhuang_Faces\Dunhuang_Faces-20K\seg_pred_val"
# os.makedirs(val_images_dir, exist_ok=True)
# os.makedirs(val_masks_dir, exist_ok=True)
#
# # 读取total_images_dir里的所有图片，打乱并抽取10%的图片作为验证集移动到val_images_dir
# # 且对应的mask也移动到val_masks_dir
# all_images = [f for f in os.listdir(total_images_dir) if f.lower().endswith(('png', 'jpg'))]
#
# random.seed(42)
# random.shuffle(all_images)
#
# val_count = int(len(all_images) * 0.1)
# val_images = all_images[:val_count]
#
# print(f"总图片数量: {len(all_images)}，验证集数量: {val_count}")
#
# for img_name in tqdm(val_images, desc="Moving Validation Images"):
#     src_img = os.path.join(total_images_dir, img_name)
#     dst_img = os.path.join(val_images_dir, img_name)
#     shutil.move(src_img, dst_img)
#
#     mask_name = os.path.splitext(img_name)[0] + ".png"  # 假设mask都是png格式
#     src_mask = os.path.join(total_masks_dir, mask_name)
#     dst_mask = os.path.join(val_masks_dir, mask_name)
#
#     if os.path.exists(src_mask):
#         shutil.move(src_mask, dst_mask)
#     else:
#         print(f"警告: 未找到对应的mask文件 {src_mask}，跳过移动该mask。")
#
# print("验证集划分完成。")
# =======================================================================================


# ======================== 整合数据集1.0 ===================================================
def get_files(src_dir):
    return sorted([f for f in os.listdir(src_dir) if f.endswith(('png', 'jpg'))])


# ======================== 整合数据集DH 的 images 和 masks 文件夹 =========================
dst_image_dir_train = r"../DH/image_mask/train/images"
dst_mask_dir_train = r"../DH/image_mask/train/masks"
dst_image_dir_val = r"../DH/image_mask/val/images"
dst_mask_dir_val = r"../DH/image_mask/val/masks"
os.makedirs(dst_image_dir_train, exist_ok=True)
os.makedirs(dst_mask_dir_train, exist_ok=True)
os.makedirs(dst_image_dir_val, exist_ok=True)
os.makedirs(dst_mask_dir_val, exist_ok=True)
# =====================================================================================


# 将
# D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\train\images
# D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\train\masks
# 中的文件复制到 ../DH/image_mask/images 和 ../DH/image_mask/masks 中
# 并重命名为 MuralDH_00001.png 这种格式

# =================================================================================
# MuralDH
# train
MuralDH_train_src_image_dir_1 = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\train\images"
MuralDH_train_src_mask_dir_1 = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\train\labels"

MuralDH_train_src_image_dir_2 = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural512"
MuralDH_train_src_mask_dir_2 = r"D:\作业\毕业设计\MuralDH\MuralDH\seg_pred"

image_files_1 = get_files(MuralDH_train_src_image_dir_1)
mask_files_1 = get_files(MuralDH_train_src_mask_dir_1)
image_files_2 = get_files(MuralDH_train_src_image_dir_2)
mask_files_2 = get_files(MuralDH_train_src_mask_dir_2)

all_pairs = []
for img, msk in zip(image_files_1, mask_files_1):
    all_pairs.append(
        (os.path.join(MuralDH_train_src_image_dir_1, img), os.path.join(MuralDH_train_src_mask_dir_1, msk)))

for img, msk in zip(image_files_2, mask_files_2):
    all_pairs.append(
        (os.path.join(MuralDH_train_src_image_dir_2, img), os.path.join(MuralDH_train_src_mask_dir_2, msk)))

for idx, (image_path, mask_path) in enumerate(all_pairs, start=1):
    new_file_name = f"MuralDH_{idx:05d}.png"

    shutil.copy(image_path, os.path.join(dst_image_dir_train, new_file_name))
    shutil.copy(mask_path, os.path.join(dst_mask_dir_train, new_file_name))
print("Data integration completed.")

all_pairs.clear()

# val
MuralDH_val_src_image_dir_1 = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\test\images"
MuralDH_val_src_mask_dir_1 = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\test\labels"
image_files_1 = get_files(MuralDH_val_src_image_dir_1)
mask_files_1 = get_files(MuralDH_val_src_mask_dir_1)

all_pairs = []
for img, msk in zip(image_files_1, mask_files_1):
    all_pairs.append((os.path.join(MuralDH_val_src_image_dir_1, img), os.path.join(MuralDH_val_src_mask_dir_1, msk)))

for idx, (image_path, mask_path) in enumerate(all_pairs, start=1):
    new_file_name = f"MuralDH_{idx:05d}.png"

    shutil.copy(image_path, os.path.join(dst_image_dir_val, new_file_name))
    shutil.copy(mask_path, os.path.join(dst_mask_dir_val, new_file_name))

print("Data integration completed.")
# =================================================================================

all_pairs.clear()

# =================================================================================
# Dunhuang_Grottoes_Painting
# train
Dunhuang_Grottoes_Painting_train_src_image_dir_1 = r"D:\作业\毕业设计\Dunhuang_Grottoes_Painting\train\train_GT_cropped"
Dunhuang_Grottoes_Painting_train_src_mask_dir_1 = r"D:\作业\毕业设计\Dunhuang_Grottoes_Painting\train\seg_pred"
image_files_1 = get_files(Dunhuang_Grottoes_Painting_train_src_image_dir_1)
mask_files_1 = get_files(Dunhuang_Grottoes_Painting_train_src_mask_dir_1)
all_pairs = []
for img, msk in zip(image_files_1, mask_files_1):
    all_pairs.append((os.path.join(Dunhuang_Grottoes_Painting_train_src_image_dir_1, img),
                      os.path.join(Dunhuang_Grottoes_Painting_train_src_mask_dir_1, msk)))

for idx, (image_path, mask_path) in enumerate(all_pairs, start=1):
    new_file_name = f"Dunhuang_Grottoes_Painting_{idx:05d}.png"

    shutil.copy(image_path, os.path.join(dst_image_dir_train, new_file_name))
    shutil.copy(mask_path, os.path.join(dst_mask_dir_train, new_file_name))

print("Data integration completed.")

all_pairs.clear()

# val
Dunhuang_Grottoes_Painting_val_src_image_dir_1 = r"D:\作业\毕业设计\Dunhuang_Grottoes_Painting\test\test_GT_cropped"
Dunhuang_Grottoes_Painting_val_src_mask_dir_1 = r"D:\作业\毕业设计\Dunhuang_Grottoes_Painting\test\seg_pred"
image_files_1 = get_files(Dunhuang_Grottoes_Painting_val_src_image_dir_1)
mask_files_1 = get_files(Dunhuang_Grottoes_Painting_val_src_mask_dir_1)

all_pairs = []
for img, msk in zip(image_files_1, mask_files_1):
    all_pairs.append((os.path.join(Dunhuang_Grottoes_Painting_val_src_image_dir_1, img),
                      os.path.join(Dunhuang_Grottoes_Painting_val_src_mask_dir_1, msk)))

for idx, (image_path, mask_path) in enumerate(all_pairs, start=1):
    new_file_name = f"Dunhuang_Grottoes_Painting_{idx:05d}.png"

    shutil.copy(image_path, os.path.join(dst_image_dir_val, new_file_name))
    shutil.copy(mask_path, os.path.join(dst_mask_dir_val, new_file_name))

print("Data integration completed.")
# =================================================================================

all_pairs.clear()

# =================================================================================
# DhMurals-inpainting-dataset
# train
DhMurals_inpainting_dataset_train_src_image_dir_1 = r"D:\作业\毕业设计\DhMurals-inpainting-dataset\train\images_cropped"
DhMurals_inpainting_dataset_train_src_mask_dir_1 = r"D:\作业\毕业设计\DhMurals-inpainting-dataset\train\seg_pred"
image_files_1 = get_files(DhMurals_inpainting_dataset_train_src_image_dir_1)
mask_files_1 = get_files(DhMurals_inpainting_dataset_train_src_mask_dir_1)
all_pairs = []
for img, msk in zip(image_files_1, mask_files_1):
    all_pairs.append((os.path.join(DhMurals_inpainting_dataset_train_src_image_dir_1, img),
                      os.path.join(DhMurals_inpainting_dataset_train_src_mask_dir_1, msk)))

for idx, (image_path, mask_path) in enumerate(all_pairs, start=1):
    new_file_name = f"DhMurals-inpainting-dataset_{idx:05d}.png"

    shutil.copy(image_path, os.path.join(dst_image_dir_train, new_file_name))
    shutil.copy(mask_path, os.path.join(dst_mask_dir_train, new_file_name))

print("Data integration completed.")

all_pairs.clear()

# val
DhMurals_inpainting_dataset_val_src_image_dir_1 = r"D:\作业\毕业设计\DhMurals-inpainting-dataset\val\images_cropped"
DhMurals_inpainting_dataset_val_src_mask_dir_1 = r"D:\作业\毕业设计\DhMurals-inpainting-dataset\val\seg_pred"
DhMurals_inpainting_dataset_val_src_image_dir_2 = r"D:\作业\毕业设计\DhMurals-inpainting-dataset\test\images_cropped"
DhMurals_inpainting_dataset_val_src_mask_dir_2 = r"D:\作业\毕业设计\DhMurals-inpainting-dataset\test\seg_pred"
image_files_1 = get_files(DhMurals_inpainting_dataset_val_src_image_dir_1)
mask_files_1 = get_files(DhMurals_inpainting_dataset_val_src_mask_dir_1)

image_files_2 = get_files(DhMurals_inpainting_dataset_val_src_image_dir_2)
mask_files_2 = get_files(DhMurals_inpainting_dataset_val_src_mask_dir_2)

all_pairs = []
for img, msk in zip(image_files_1, mask_files_1):
    all_pairs.append((os.path.join(DhMurals_inpainting_dataset_val_src_image_dir_1, img),
                      os.path.join(DhMurals_inpainting_dataset_val_src_mask_dir_1, msk)))

for img, msk in zip(image_files_2, mask_files_2):
    all_pairs.append((os.path.join(DhMurals_inpainting_dataset_val_src_image_dir_2, img),
                      os.path.join(DhMurals_inpainting_dataset_val_src_mask_dir_2, msk)))

for idx, (image_path, mask_path) in enumerate(all_pairs, start=1):
    new_file_name = f"DhMurals-inpainting-dataset_{idx:05d}.png"

    shutil.copy(image_path, os.path.join(dst_image_dir_val, new_file_name))
    shutil.copy(mask_path, os.path.join(dst_mask_dir_val, new_file_name))

print("Data integration completed.")
# =================================================================================

all_pairs.clear()

# =================================================================================
# Dunhuang_Faces
# train
Dunhuang_Faces_train_src_image_dir_1 = r"D:\作业\毕业设计\Dunhuang_Faces\Dunhuang_Faces-20K\face_imgs_cropped"
Dunhuang_Faces_train_src_mask_dir_1 = r"D:\作业\毕业设计\Dunhuang_Faces\Dunhuang_Faces-20K\seg_pred"
image_files_1 = get_files(Dunhuang_Faces_train_src_image_dir_1)
mask_files_1 = get_files(Dunhuang_Faces_train_src_mask_dir_1)
all_pairs = []
for img, msk in zip(image_files_1, mask_files_1):
    all_pairs.append((os.path.join(Dunhuang_Faces_train_src_image_dir_1, img),
                      os.path.join(Dunhuang_Faces_train_src_mask_dir_1, msk)))

for idx, (image_path, mask_path) in enumerate(all_pairs, start=1):
    new_file_name = f"Dunhuang_Faces_{idx:05d}.png"

    shutil.copy(image_path, os.path.join(dst_image_dir_train, new_file_name))
    shutil.copy(mask_path, os.path.join(dst_mask_dir_train, new_file_name))

print("Data integration completed.")

all_pairs.clear()

# val
Dunhuang_Faces_val_src_image_dir_1 = r"D:\作业\毕业设计\Dunhuang_Faces\Dunhuang_Faces-20K\face_imgs_cropped_val"
Dunhuang_Faces_val_src_mask_dir_1 = r"D:\作业\毕业设计\Dunhuang_Faces\Dunhuang_Faces-20K\seg_pred_val"

image_files_1 = get_files(Dunhuang_Faces_val_src_image_dir_1)
mask_files_1 = get_files(Dunhuang_Faces_val_src_mask_dir_1)

all_pairs = []
for img, msk in zip(image_files_1, mask_files_1):
    all_pairs.append((os.path.join(Dunhuang_Faces_val_src_image_dir_1, img),
                      os.path.join(Dunhuang_Faces_val_src_mask_dir_1, msk)))

for idx, (image_path, mask_path) in enumerate(all_pairs, start=1):
    new_file_name = f"Dunhuang_Faces_{idx:05d}.png"

    shutil.copy(image_path, os.path.join(dst_image_dir_val, new_file_name))
    shutil.copy(mask_path, os.path.join(dst_mask_dir_val, new_file_name))

print("Data integration completed.")
all_pairs.clear()
# =================================================================================
# =================================================================================


# # ========================= 整合数据集2.0 =========================
# def collect_files_dict(folder):
#     file = [f for f in os.listdir(folder) if f.endswith(('png', 'jpg'))]
#     return {os.path.splitext(f)[0]: os.path.join(folder, f) for f in file}
#
#
# def collect_pairs(image_dir, mask_dir):
#     image_dict = collect_files_dict(image_dir)
#     mask_dict = collect_files_dict(mask_dir)
#
#     common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
#
#     pairs = [(image_dict[key], mask_dict[key]) for key in common_keys]
#
#     print(f"[INFO] {image_dir}")
#     print(f"Matched pairs: {len(pairs)}")
#
#     return pairs
#
#
# def integrate_dataset(src_pairs_list, dst_img_dir, dst_mask_dir, prefix, start_idx=1):
#     os.makedirs(dst_img_dir, exist_ok=True)
#     os.makedirs(dst_mask_dir, exist_ok=True)
#
#     idx = start_idx
#     for pairs in src_pairs_list:
#         for img_path, mask_path in tqdm(pairs):
#             new_file_name = f"{prefix}_{idx:05d}.png"
#             shutil.copy(img_path, os.path.join(dst_img_dir, new_file_name))
#             shutil.copy(mask_path, os.path.join(dst_mask_dir, new_file_name))
#             idx += 1
#
#     print(f"[DONE] Total images: {idx - start_idx}")
#     return idx
#
#
# # ==================== 路径配置 ====================
#
# dst_image_dir_train = r"../DH/image_mask/train/images"
# dst_mask_dir_train = r"../DH/image_mask/train/masks"
#
# dst_image_dir_val = r"../DH/image_mask/val/images"
# dst_mask_dir_val = r"../DH/image_mask/val/masks"
#
# # ==================== MuralDH train ====================
# train_pairs_1 = collect_pairs(
#     r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\train\images",
#     r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\train\labels"
# )
#
# train_pairs_2 = collect_pairs(
#     r"D:\作业\毕业设计\MuralDH\MuralDH\Mural512",
#     r"D:\作业\毕业设计\MuralDH\MuralDH\seg_pred"
# )
#
# integrate_dataset(
#     [train_pairs_1, train_pairs_2],
#     dst_image_dir_train,
#     dst_mask_dir_train,
#     prefix="MuralDH",
#     start_idx=1
# )
#
# # ==================== MuralDH val ====================
# val_pairs = collect_pairs(
#     r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\test\images",
#     r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\test\labels"
# )
#
# integrate_dataset(
#     [val_pairs],
#     dst_image_dir_val,
#     dst_mask_dir_val,
#     prefix="MuralDH",
#     start_idx=1
# )
# # ===========================================================
