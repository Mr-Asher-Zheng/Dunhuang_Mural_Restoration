import os
import re
import shutil
import torch.nn.functional as F
import torch
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim, ssim

ControlNet_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\ControlNet\gt'
ControlNet_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\ControlNet\pred'
ControlNet_mask_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\ControlNet\mask'

FinetuneControlNet_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\FinetuneControlNet\gt'
FinetuneControlNet_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\FinetuneControlNet\pred'
FinetuneControlNet_mask_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\FinetuneControlNet\mask'
# FinetuneControlNet_gt_dir = r'D:\作业\毕业设计\论文\答辩文件\图片\copy\gt'
# FinetuneControlNet_pred_dir = r'D:\作业\毕业设计\论文\答辩文件\图片\copy\pred'

MISF_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\MISF\gt'
MISF_pred_dir = r'D:\作业\毕业设计\misf-main\data\result'
MISF_mask_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\MISF\mask'

EdgeConnect_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\EdgeConnect\gt'
EdgeConnect_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\EdgeConnect\pred'
EdgeConnect_mask_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\EdgeConnect\mask'

Lama_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\Lama\gt'
Lama_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\Lama\pred'
Lama_mask_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\Lama\mask'

use_mask_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\用mask不用轮廓\gt'
use_mask_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\用mask不用轮廓\pred'
use_mask_mask_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\用mask不用轮廓\mask'

use_line_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\用轮廓不用mask\gt'
use_line_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\用轮廓不用mask\pred'
use_line_mask_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\用轮廓不用mask\mask'

landscape_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\LandscapeTest\gt'
landscape_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\LandscapeTest\pred'
landscape_mask_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\LandscapeTest\mask'

line1_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\line1\gt'
line1_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\line1\pred'

line2_gt_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\line2\gt'
line2_pred_dir = r'D:\作业\毕业设计\Diffusion\ImageRestoration\samples\Evaluation\line2\pred'

# 遍历文件
for filename in os.listdir(Lama_pred_dir):
    # 只处理以 _mask.png 结尾的文件
    if filename.endswith('_mask.png'):
        # 去掉 _mask 加上 _pred
        new_filename = filename.replace('_mask.png', '_pred.png')

        # 完整路径
        old_path = os.path.join(Lama_pred_dir, filename)
        new_path = os.path.join(Lama_pred_dir, new_filename)

        # 安全重命名（不覆盖已存在文件）
        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"✅ 已修改：{filename} → {new_filename}")
        else:
            print(f"⚠️ 已存在，跳过：{new_filename}")


def psnr(img1, img2, max_value=1.0, eps=1e-5):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(max_value / torch.sqrt(mse + eps))


image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])
mask_transform = transforms.Compose([
    transforms.ToTensor(),
])


# 读取gt和pred目录下的所有图像文件，并计算SSIM
# 目录下的文件名字都是：
# xxxx_gt.png 和 xxxx_pred.png
# xxxx是一样的
def calculate_ssim(gt_dir, pred_dir, mask_dir=None):
    ssim_values = []
    ms_ssim_values = []
    PSNR_values = []
    MSE_values = []

    # gt_image_list = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith(('png', 'jpg'))])
    # pred_image_list = sorted([os.path.join(pred_dir, f) for f in os.listdir(pred_dir) if f.endswith(('png', 'jpg'))])

    gt_files = [f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg'))]
    for file in gt_files:
        if not file.endswith("_gt.png"):
            continue

        base_name = file.replace("_gt.png", "")

        gt_path = os.path.join(gt_dir, f"{base_name}_gt.png")
        pred_path = os.path.join(pred_dir, f"{base_name}_pred.png")
        mask_path = os.path.join(mask_dir, f"{base_name}_mask.png") if mask_dir else None

        if not os.path.exists(pred_path):
            print(f"❌ Missing: {base_name}")
            continue

        gt_image = Image.open(gt_path).convert("RGB")
        pred_image = Image.open(pred_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L") if mask_dir else None

        gt_tensor = image_transform(gt_image).unsqueeze(0)
        pred_tensor = image_transform(pred_image).unsqueeze(0)
        mask_tensor = mask_transform(mask_image).unsqueeze(0) if mask_dir else None

        mask_tensor = (mask_tensor > 0).float() if mask_dir else None

        mask_ratio = mask_tensor.mean() if mask_dir else None

        # 计算 SSIM
        ssim_score = ssim(gt_tensor, pred_tensor, data_range=2.0, size_average=True)
        ms_ssim_score = ms_ssim(gt_tensor, pred_tensor, data_range=2.0, size_average=True)
        PSNR_score = psnr(gt_tensor, pred_tensor)
        MSE_score = F.mse_loss(gt_tensor, pred_tensor)

        if mask_dir:
            ssim_values.append(ssim_score.item() * mask_ratio.item())
            ms_ssim_values.append(ms_ssim_score.item() * mask_ratio.item())
            PSNR_values.append(PSNR_score.item() * mask_ratio.item())
        else:
            ssim_values.append(ssim_score.item())
            ms_ssim_values.append(ms_ssim_score.item())
            PSNR_values.append(PSNR_score.item())
            MSE_values.append(MSE_score.item())

    # 平均
    avg_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    avg_ms_ssim = sum(ms_ssim_values) / len(ms_ssim_values) if ms_ssim_values else 0
    PSNR = sum(PSNR_values) / len(PSNR_values) if PSNR_values else 0
    MSE = sum(MSE_values) / len(MSE_values) if MSE_values else 0

    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MS-SSIM: {avg_ms_ssim:.4f}")
    print(f"Average PSNR: {PSNR:.4f} dB")
    print(f"Average MSE: {MSE:.4f}")


print("用mask不用轮廓:")
calculate_ssim(use_mask_gt_dir, use_mask_pred_dir)
print("-----------------------------------")
print("用轮廓不用mask:")
calculate_ssim(use_line_gt_dir, use_line_pred_dir)
print("-----------------------------------")

print("line1")
calculate_ssim(line1_gt_dir, line1_pred_dir)
print("------------------------------------")

print("line2")
calculate_ssim(line2_gt_dir, line2_pred_dir)

print("controlnet:")
calculate_ssim(ControlNet_gt_dir, ControlNet_pred_dir)
print("-----------------------------------")
print("fine-tune controlnet:")
calculate_ssim(FinetuneControlNet_gt_dir, FinetuneControlNet_pred_dir)
print("-----------------------------------")
print("MISF:")
calculate_ssim(MISF_gt_dir, MISF_pred_dir)
print("-----------------------------------")
print("EdgeConnect:")
calculate_ssim(EdgeConnect_gt_dir, EdgeConnect_pred_dir)
print("-----------------------------------")
print("Lama:")
calculate_ssim(Lama_gt_dir, Lama_pred_dir)
print("-----------------------------------")
print("landscape:")
calculate_ssim(landscape_gt_dir, landscape_pred_dir)
print("-----------------------------------")

print("==============================================================")

print("用mask不用轮廓:")
calculate_ssim(use_mask_gt_dir, use_mask_pred_dir, use_mask_mask_dir)
print("-----------------------------------")
print("用轮廓不用mask:")
calculate_ssim(use_line_gt_dir, use_line_pred_dir, use_line_mask_dir)
print("-----------------------------------")

print("controlnet:")
calculate_ssim(ControlNet_gt_dir, ControlNet_pred_dir, ControlNet_mask_dir)
print("-----------------------------------")
print("fine-tune controlnet:")
calculate_ssim(FinetuneControlNet_gt_dir, FinetuneControlNet_pred_dir, FinetuneControlNet_mask_dir)
print("-----------------------------------")
print("MISF:")
calculate_ssim(MISF_gt_dir, MISF_pred_dir, MISF_mask_dir)
print("-----------------------------------")
print("EdgeConnect:")
calculate_ssim(EdgeConnect_gt_dir, EdgeConnect_pred_dir, EdgeConnect_mask_dir)
print("-----------------------------------")
print("Lama:")
calculate_ssim(Lama_gt_dir, Lama_pred_dir, Lama_mask_dir)

# gt_save_dir = "./samples/Evaluation/FinetuneControlNet/gt"
# pred_save_dir = "./samples/Evaluation/FinetuneControlNet/pred"
# input_save_dir = "./samples/Evaluation/FinetuneControlNet/input"
# line_save_dir = "./samples/Evaluation/FinetuneControlNet/line"
# mask_save_dir = "./samples/Evaluation/FinetuneControlNet/mask"
#
# name_list = ['Dunhuang_Faces_00511', 'Dunhuang_Faces_00471', 'Dunhuang_Faces_00643',
#              'DhMurals-inpainting-dataset_00002', 'MuralDH_00139', 'MuralDH_00033', 'MuralDH_00062',
#              'Dunhuang_Faces_00362', 'Dunhuang_Faces_00599', 'DhMurals-inpainting-dataset_00015',
#              'Dunhuang_Faces_00317', 'MuralDH_00038', 'MuralDH_00160', 'Dunhuang_Faces_00478', 'Dunhuang_Faces_00534',
#              'Dunhuang_Faces_00244', 'DhMurals-inpainting-dataset_00095', 'Dunhuang_Faces_00241', 'MuralDH_00095',
#              'DhMurals-inpainting-dataset_00076', 'Dunhuang_Faces_00355', 'Dunhuang_Faces_00532',
#              'Dunhuang_Faces_00680', 'Dunhuang_Faces_00452', 'Dunhuang_Faces_00674', 'Dunhuang_Faces_00352',
#              'Dunhuang_Faces_00022', 'Dunhuang_Grottoes_Painting_00051', 'DhMurals-inpainting-dataset_00024',
#              'Dunhuang_Faces_00199', 'Dunhuang_Faces_00343', 'Dunhuang_Faces_00260',
#              'DhMurals-inpainting-dataset_00082', 'Dunhuang_Faces_00391', 'Dunhuang_Faces_00496',
#              'Dunhuang_Faces_00223', 'DhMurals-inpainting-dataset_00081', 'DhMurals-inpainting-dataset_00096',
#              'Dunhuang_Grottoes_Painting_00070', 'DhMurals-inpainting-dataset_00017', 'MuralDH_00070',
#              'Dunhuang_Grottoes_Painting_00061', 'DhMurals-inpainting-dataset_00110', 'Dunhuang_Faces_00417',
#              'Dunhuang_Grottoes_Painting_00050', 'DhMurals-inpainting-dataset_00112',
#              'Dunhuang_Grottoes_Painting_00006', 'DhMurals-inpainting-dataset_00038', 'Dunhuang_Faces_00629',
#              'DhMurals-inpainting-dataset_00097', 'DhMurals-inpainting-dataset_00136', 'Dunhuang_Faces_00503',
#              'DhMurals-inpainting-dataset_00089', 'DhMurals-inpainting-dataset_00122', 'MuralDH_00162', 'MuralDH_00140',
#              'MuralDH_00192', 'Dunhuang_Faces_00300', 'Dunhuang_Faces_00491', 'Dunhuang_Faces_00331', 'MuralDH_00043',
#              'MuralDH_00098', 'MuralDH_00166', 'Dunhuang_Faces_00353', 'DhMurals-inpainting-dataset_00060',
#              'Dunhuang_Grottoes_Painting_00099', 'Dunhuang_Faces_00370', 'Dunhuang_Faces_00181', 'Dunhuang_Faces_00161',
#              'Dunhuang_Faces_00588', 'Dunhuang_Faces_00005', 'Dunhuang_Faces_00021', 'Dunhuang_Faces_00401',
#              'Dunhuang_Faces_00266', 'DhMurals-inpainting-dataset_00004', 'Dunhuang_Faces_00236',
#              'Dunhuang_Faces_00337', 'MuralDH_00136', 'Dunhuang_Faces_00445', 'Dunhuang_Faces_00399',
#              'DhMurals-inpainting-dataset_00114', 'Dunhuang_Faces_00187', 'Dunhuang_Faces_00550',
#              'Dunhuang_Grottoes_Painting_00010', 'Dunhuang_Faces_00489', 'Dunhuang_Faces_00378', 'Dunhuang_Faces_00635',
#              'DhMurals-inpainting-dataset_00109', 'MuralDH_00078', 'MuralDH_00200', 'Dunhuang_Faces_00470',
#              'Dunhuang_Faces_00443', 'Dunhuang_Faces_00224', 'Dunhuang_Grottoes_Painting_00091',
#              'DhMurals-inpainting-dataset_00143', 'Dunhuang_Faces_00095', 'DhMurals-inpainting-dataset_00139',
#              'Dunhuang_Faces_00575', 'DhMurals-inpainting-dataset_00018', 'MuralDH_00013']
#
# # gt_save_dir文件夹里图片的名字：Dunhuang_Faces_00511_gt.png
# # pred_save_dir文件夹里图片的名字：Dunhuang_Faces_00511_pred.png
# # ...
#
# target_root = 'D:\作业\毕业设计\论文\答辩文件\图片\copy'
#
# # 按照原来结构复制list到path下
# # 5个文件夹的配置：源路径 + 后缀
# folders = [
#     (gt_save_dir, "gt"),
#     (pred_save_dir, "pred"),
#     (input_save_dir, "input"),
#     (line_save_dir, "line"),
#     (mask_save_dir, "mask"),
# ]
#
# # 开始复制
# for src_dir, suffix in folders:
#     # 目标子文件夹：copy/gt, copy/pred ...
#     target_dir = os.path.join(target_root, suffix)
#     os.makedirs(target_dir, exist_ok=True)
#
#     # 遍历需要复制的文件名
#     for name in name_list:
#         # 构造文件名：xxx_gt.png / xxx_pred.png ...
#         filename = f"{name}_{suffix}.png"
#         src_path = os.path.join(src_dir, filename)
#         dst_path = os.path.join(target_dir, filename)
#
#         # 如果源文件存在，就复制
#         if os.path.exists(src_path):
#             shutil.copy2(src_path, dst_path)
#             print(f"✅ 已复制：{filename}")
#         else:
#             print(f"⚠️ 不存在，跳过：{src_path}")
#
# print("\n🎉 全部复制完成！")
