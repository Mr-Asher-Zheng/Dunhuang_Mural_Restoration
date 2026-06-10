from PIL import Image, ImageDraw, ImageFont
import os

# =============================================================================
# images_dir = r"D:\作业\毕业设计\DH\samples\seg\images"
# SAM_dir = r"D:\作业\毕业设计\DH\samples\seg\SAM"
# finetune_SAM_dir = r"D:\作业\毕业设计\DH\samples\seg\finetune_SAM"
# finetune_SAM_Adapter_dir = r"D:\作业\毕业设计\DH\samples\seg\finetune_SAM-Adapter"
#
# # 每个文件夹都有6张图像，
# # images_dir是原图，SAM_dir是SAM的分割结果，finetune_SAM_dir是微调SAM的分割结果，finetune_SAM_Adapter_dir是微调SAM-Adapter的分割结果
# # 先将每个文件夹里的图片横向拼接，然后将四个文件夹的拼接结果竖向拼接（按上面文件顺序），最后显示出来
# # 在最左边显示名字，分别是"原图"、"SAM"、"微调SAM"、"微调SAM-Adapter"
# dirs = [
#     ("原图", images_dir),
#     ("SAM", SAM_dir),
#     ("微调SAM", finetune_SAM_dir),
#     ("微调SAM-Adapter", finetune_SAM_Adapter_dir)
# ]
# =============================================================================


# =============================================================================
input_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/input"
origin_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/gt"
no_finetune_ControlNet_dir = r"./ImageRestoration/samples/ddpm/ControlNet_ddpm/pred"
finetune_ControlNet_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/pred"

dirs = [
    ("输入", input_dir),
    ("原图", origin_dir),
    ("无微调ControlNet", no_finetune_ControlNet_dir),
    ("微调ControlNet", finetune_ControlNet_dir),
]
# =============================================================================

# =============================================================================
# input_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/input"
# origin_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/gt"
# no_finetune_ControlNet_dir = r"./ImageRestoration/samples/ddpm/扩散完后直接恢复完好区域/pred"
# finetune_ControlNet_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/pred"
#
# dirs = [
#     ("输入", input_dir),
#     ("原图", origin_dir),
#     ("（a）", no_finetune_ControlNet_dir),
#     ("（b）", finetune_ControlNet_dir),
# ]
# =============================================================================


# =============================================================================
# input_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/input"
# origin_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/gt"
# no_finetune_ControlNet_dir = r"./ImageRestoration/samples/ddpm/不膨胀/pred"
# finetune_ControlNet_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/pred"
#
# dirs = [
#     ("输入", input_dir),
#     ("原图", origin_dir),
#     ("无高斯膨胀", no_finetune_ControlNet_dir),
#     ("有高斯膨胀", finetune_ControlNet_dir),
# ]
# =============================================================================


# =============================================================================
# input_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/input"
# origin_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/gt"
# use_mask_dir = r"./ImageRestoration/samples/ddpm/用mask不用轮廓/pred"
# use_line_dir = r"./ImageRestoration/samples/ddpm/用轮廓不用mask/pred"
# use_both_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/pred"
#
# dirs = [
#     ("输入", input_dir),
#     ("原图", origin_dir),
#     ("有掩码，无轮廓", use_mask_dir),
#     ("有轮廓，无掩码", use_line_dir),
#     ("有掩码，有轮廓", use_both_dir),
# ]
# =============================================================================

# =============================================================================
# input_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/input"
# origin_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/gt"
# MISF_dir = r"./ImageRestoration/samples/ddpm/MISF_ddpm/pred"
# EdgeConnect_dir = r"./ImageRestoration/samples/ddpm/EdgeConnect_ddpm/pred"
# Lama_dir = r"./ImageRestoration/samples/ddpm/Lama_ddpm/pred"
# ControlNet_dir = r"./ImageRestoration/samples/ddpm/Finetune_ddpm/pred"
# dirs = [
#     ("输入", input_dir),
#     ("原图", origin_dir),
#     ("MISF", MISF_dir),
#     ("EdgeConnect", EdgeConnect_dir),
#     ("Lama", Lama_dir),
#     ("微调ControlNet", ControlNet_dir),
# ]
# =============================================================================

# =============================================================================
# input_dir = r"./ImageRestoration/samples/Evaluation/LandscapeTest/论文展示/输入"
# origin_dir = r"./ImageRestoration/samples/Evaluation/LandscapeTest/论文展示/原图"
# pred_dir = r"./ImageRestoration/samples/Evaluation/LandscapeTest/论文展示/输出"
#
# dirs = [
#     ("输入", input_dir),
#     ("原图", origin_dir),
#     ("微调ControlNet", pred_dir),
# ]
# =============================================================================

gap = 20  # 图片间距
left_text_width = 400  # 左侧文字区域宽度


def load_images(folder):
    files = sorted(os.listdir(folder))
    images = [Image.open(os.path.join(folder, f)).convert("RGB") for f in files]
    return images


def concat_horizontal(images):
    w, h = images[0].size
    total_w = len(images) * w + (len(images) - 1) * gap

    canvas = Image.new("RGB", (total_w, h), (255, 255, 255))

    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += w + gap

    return canvas


rows = []
labels = []

for name, folder in dirs:
    imgs = load_images(folder)
    row = concat_horizontal(imgs)
    rows.append(row)
    labels.append(name)

row_w, row_h = rows[0].size
total_h = len(rows) * row_h + (len(rows) - 1) * gap
total_w = row_w + left_text_width

final_img = Image.new("RGB", (total_w, total_h), (255, 255, 255))

draw = ImageDraw.Draw(final_img)

# 字体
try:
    font = ImageFont.truetype("simhei.ttf", 40)
except:
    font = ImageFont.load_default()

y = 0
for i, row in enumerate(rows):
    final_img.paste(row, (left_text_width, y))

    text = labels[i]
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    text_x = (left_text_width - text_w) // 2
    text_y = y + (row_h - text_h) // 2

    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    draw.rectangle([left_text_width, y, left_text_width + row_w, y + row_h], outline=(0, 0, 0), width=2)

    y += row_h + gap

final_img.show()
final_img.save("compare.png", dpi=(300, 300))
