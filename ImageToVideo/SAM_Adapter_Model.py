from torch import nn
from segment_anything import sam_model_registry
import torch.nn.functional as F


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
    def __init__(self, model_type, checkpoint):
        super().__init__()
        # 加载官方的模型
        # self.sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)

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
