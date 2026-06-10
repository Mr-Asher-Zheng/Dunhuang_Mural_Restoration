import gradio as gr
import os
import cv2
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch import nn
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
import numpy as np


class Adapter(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = dim // reduction
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        # print(x.shape)
        return x + self.adapter(x)


class AdapterBlock(nn.Module):
    def __init__(self, block, dim):
        super().__init__()
        self.block = block
        self.adapter = Adapter(dim)

    def forward(self, x):
        out = self.block(x)
        out = self.adapter(out)
        return out


class SAMAdapterModel(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        # 加载官方的模型
        self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        # self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint)

        self.image_encoder = self.sam.image_encoder
        self.prompt_encoder = self.sam.prompt_encoder
        self.mask_decoder = self.sam.mask_decoder

        # 冻结原始权重
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        # 在每个Transformer block 末尾插 Adapter
        dim = self.image_encoder.blocks[0].norm1.normalized_shape[0]
        self.image_encoder.blocks = nn.ModuleList([
            AdapterBlock(block, dim) for block in self.image_encoder.blocks
        ])

        # 调整位置编码的形状，使其匹配当前输入图像的尺寸
        self.image_encoder.pos_embed = self.resize_pos_embed(
            self.image_encoder.pos_embed, H=512, W=512
        )

    def resize_pos_embed(self, pos_embed, H, W):
        # 获取原始位置编码的高度和宽度
        # [1, 64, 64, 768]
        # 原始的位置编码是针对 1024x1024 图像设计的，经过 patch embedding 后（分割成16×16的小方块，1024÷16）变为 64x64，
        # 每个 16×16 patch 都有自己的一条 768 维的位置向量
        _, H_old, W_old, C = pos_embed.shape

        # 计算新的高度和宽度，因为输入别的尺寸的图像，匹配不了原来的patch
        # 512//16 512//16
        H_new, W_new = H // 16, W // 16

        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, C, H_old, W_old]

        # 把预训练的二维位置编码当作多通道图片，用双线性插值的方法重新采样到目标分辨率
        pos_embed_resized = F.interpolate(
            pos_embed, size=(H_new, W_new), mode='bilinear', align_corners=False
        )

        # [1, 32, 32, 768]
        pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1)  # [1, H_new, W_new, C]

        # 把一个普通张量包装成可训练的模型参数并返回
        return nn.Parameter(pos_embed_resized)

    def forward(self, images):
        # [16, 3, 512, 512]
        B, C, H, W = images.shape

        # # 调整位置编码的形状，使其匹配当前输入图像的尺寸
        # self.image_encoder.pos_embed = self.resize_pos_embed(self.image_encoder.pos_embed, H, W)

        # SAM image_encoder 提取图像特征
        features = self.image_encoder(images)  # [B, C, H/16, W/16]
        # print(features.shape)

        # print(self.prompt_encoder)
        # SAM mask_decoder 需要 prompt embedding；这里直接给空（全图分割）
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        image_pe = self.prompt_encoder.get_dense_pe()

        # print(image_pe.shape)
        # print(sparse_embeddings.shape)
        # print(dense_embeddings.shape)

        H_feat = features.shape[-2]
        W_feat = features.shape[-1]
        image_pe = F.interpolate(image_pe, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
        dense_embeddings = F.interpolate(dense_embeddings, size=(H_feat, W_feat), mode='bilinear', align_corners=False)

        # print(image_pe.shape)
        # print(dense_embeddings.shape)
        #
        # print("=======================================")

        mask, _ = self.mask_decoder(
            image_embeddings=features,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        mask = F.interpolate(
            mask, size=(H, W),  # 这里的 H, W 就是原图 512, 512
            mode="bilinear", align_corners=False
        )
        return mask


def load_models_status(sam_file, seg_file, res_file, sr_file, api_key):
    msg = []
    if sam_file:
        msg.append(f"✔ SAM基础模型已加载: {sam_file.name}")
    else:
        msg.append("❌ SAM基础模型未加载")

    if seg_file:
        msg.append(f"✔ 损坏区域检测模型已加载: {seg_file.name}")
    else:
        msg.append("❌ 损坏区域检测模型未加载")

    if res_file:
        msg.append(f"✔ 图像修复模型已加载: {res_file.name}")
    else:
        msg.append("❌ 图像修复模型未加载")

    if sr_file:
        msg.append(f"✔ 超分辨率模型已加载: {sr_file.name}")
    else:
        msg.append("❌ 超分辨率模型未加载")

    if api_key:
        msg.append("✔ 通义万相API Key已设置")
    else:
        msg.append("❌ 通义万相API Key未设置")

    return "\n".join(msg)


def clear_all():
    return None, None, None, None, "", "已清空所有内容！"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SAMAdapterModel(checkpoint=r"D:\作业\毕业设计\Diffusion\ImageSeg\model\sam_vit_h_4b8939.pth")
model.load_state_dict(torch.load(r"D:\作业\毕业设计\Diffusion\ImageSeg\model\best_iou_model.pth", weights_only=True))
model = model.to(device)
model.eval()
print("加载模型完成")
print(next(model.parameters()).device)


def Image_seg_detect(
        # sam_model_path, finetuned_sam_model_path,
        image_path, size, device
):
    # print("sam_model_path:", sam_model_path)
    # print("finetuned_sam_model_path:", finetuned_sam_model_path)
    print("image_path:", image_path)

    # model = SAMAdapterModel(checkpoint=sam_model_path)
    # model.load_state_dict(torch.load(finetuned_sam_model_path, weights_only=True))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    transformed = transform(image=image)
    image = transformed['image']

    image = image.unsqueeze(0).to(device)
    print("image:")
    print(image.shape)
    print(image.device)

    with torch.no_grad():
        # [8, 1, 512, 512]
        pred_masks = model(image)
        print("pred_masks:")
        print(pred_masks.shape)

        # [8, 512, 512, 1]
        pred_masks = torch.clamp(pred_masks, -1, 1).detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
        print(pred_masks.shape)

        # [8, 512, 512]
        if pred_masks.shape[-1] == 1:
            pred_masks = pred_masks.squeeze(-1)
            print(pred_masks.shape)

    return pred_masks[0]


# sam_model_path = "../ImageSeg/model/sam_vit_h_4b8939.pth"
# finetuned_sam_model_path = "../ImageSeg/model/best_iou_model.pth"
# img = r"C:\Users\ASUS\Desktop\img_898crop_0_1.png"
# image = Image_seg_detect(sam_model_path, finetuned_sam_model_path, img, 512, "cuda")
# print(image)

if __name__ == "__main__":
    with gr.Blocks(title="敦煌壁画修复系统") as demo:
        with gr.Tab(label="导入模型"):
            gr.Markdown(
                """
                上传或配置需要使用的模型文件，点击 **应用** 后生效。
                """
            )
            with gr.Column():
                with gr.Group():
                    with gr.Accordion("损坏区域检测模型", open=True):
                        with gr.Row():
                            SAM_model_file = gr.File(label="上传SAM基础模型", file_types=[".pth", ".pt"])
                            image_seg_model_file = gr.File(label="上传损坏区域检测模型", file_types=[".pth", ".pt"])
                with gr.Group():
                    with gr.Accordion("图像修复模型", open=True):
                        image_restoration_model_file = gr.File(label="上传图像修复模型", file_types=[".pth", ".pt"])
                with gr.Group():
                    with gr.Accordion("超分辨率模型", open=True):
                        image_sr_model_file = gr.File(label="上传超分辨率模型", file_types=[".pth", ".pt"])
                with gr.Group():
                    with gr.Accordion("通义万相API", open=True):
                        api_key = gr.Textbox(label="API Key", placeholder="输入API Key")

                load_button = gr.Button("应用")
                clear_button = gr.Button("清空")
                status = gr.Textbox(label="状态信息", interactive=False)

                load_button.click(
                    load_models_status,
                    inputs=[
                        SAM_model_file,
                        image_seg_model_file,
                        image_restoration_model_file,
                        image_sr_model_file,
                        api_key,
                    ],
                    outputs=status
                )

                clear_button.click(
                    clear_all,
                    inputs=[],
                    outputs=[
                        SAM_model_file,
                        image_seg_model_file,
                        image_restoration_model_file,
                        image_sr_model_file,
                        api_key,
                        status
                    ]
                )

        with gr.Tab(label="模型推理"):
            with gr.Column():
                # 展示图像修复过程
                with gr.Group():
                    with gr.Row(equal_height=True):
                        # 展示损坏区域检测的输入图像和输出图像
                        pass

                with gr.Accordion("损坏区域检测模块", open=True):
                    gr.Markdown(
                        """
                        1. 上传需要修复的敦煌壁画图像。
                        2. 系统将使用导入的模型进行损坏区域检测。
                        3. 下载检测结果。
                        """
                    )
                    with gr.Row():
                        input_image = gr.Image(label="上传敦煌壁画图像", type="filepath")
                        output_mask = gr.Image(label="检测结果")
                    size = gr.Number(value=512, label="输入图像尺寸")
                    device = gr.Dropdown(choices=["cpu", "cuda"], value="cuda", label="运行设备")

                    Image_seg_detect_button = gr.Button("开始损坏区域检测")
                    try:
                        Image_seg_detect_button.click(
                            Image_seg_detect,
                            inputs=[
                                # SAM_model_file,
                                # image_seg_model_file,
                                input_image,
                                size,
                                device
                            ],
                            outputs=output_mask
                        )
                    except Exception as e:
                        print("设置损坏区域检测按钮失败:", e)

                with gr.Accordion("XDoG线稿提取模块", open=True):
                    gr.Markdown(
                        """
                        1. 上传需要修复的敦煌壁画图像。
                        2. 系统将使用导入的模型进行损坏区域检测。
                        3. 下载检测结果。
                        """
                    )

    # 运行
    demo.launch()
