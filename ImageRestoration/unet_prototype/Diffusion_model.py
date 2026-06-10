import math
import time

import torch
from tensorboardX import SummaryWriter
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    """
    在扩散模型中，时间步 t 是关键信息（不同 t 对应不同噪声程度），但 t 本身是单个数值（如 t=500），无法直接输入卷积网络。
    需通过 "时间嵌入"（如正弦余弦编码）将 t 转换为 time_emb_dim 维的向量，再注入网络各层，让模型感知当前处于哪个扩散步骤。

    Args：
        dim: 嵌入向量的维度（整数），决定输出 embedding 的长度
        time: 时间步张量，形状为 (batch_size,)，里面是每个样本的时间步编号

    Returns:
        embeddings: 时间步的正弦/余弦嵌入向量，形状为 (batch_size, dim)
            - 前半部分 (dim // 2): sin(time * freq)
            - 后半部分 (dim // 2): cos(time * freq)
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: 时间步张量，形状为 (batch_size,)
        Returns:
            embeddings: 时间步的正弦/余弦嵌入向量，形状为 (batch_size, dim)
        """
        device = time.device
        # 将维度分成两半，分别用于sin和cos
        half_dim = self.dim // 2
        # print(half_dim)

        # 计算不同频率的指数衰减
        step = math.log(10000) / (half_dim - 1)
        # 生成频率序列
        freqs = torch.exp(torch.arange(half_dim, device=device) * -step)
        # print(time[:, None])
        # print(freqs[None, :])

        # 将时间步与频率序列相乘
        # 对应time * freq
        # [batch, dim//2]
        embeddings = time[:, None] * freqs[None, :]
        # print(embeddings)

        # 拼接sin和cos得到最终的嵌入向量
        # [batch, dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        self.block = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),

            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, t_emb):
        # print(x)
        # print(t_emb)
        t = self.time_mlp(t_emb)[:, :, None, None]
        # print(t)
        h = self.block(x)

        h = h + t

        h = h + self.res_conv(x)

        return h


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, downsample=True):
        super().__init__()
        self.res_block = ResnetBlock(in_channels, out_channels, time_emb_dim)

        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x, t_emb):
        x = self.res_block(x, t_emb)
        x = self.downsample(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, upsample=True):
        super().__init__()
        self.res_block = ResnetBlock(in_channels, out_channels, time_emb_dim)

        if upsample:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = nn.Identity()

    def forward(self, x, t_emb):
        x = self.res_block(x, t_emb)
        x = self.upsample(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, time_emb_dim):
        """
        Args:
            in_channels: 输入图像的通道数
            out_channels: 输出图像的通道数
            base_channels: 基础通道数，决定网络的宽度
            time_emb_dim: 时间嵌入的维度
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim

        # 各层通道数设计
        self.down_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8,
                              base_channels * 16]
        self.up_channels = list(reversed(self.down_channels))

        # 时间嵌入层模块
        self.time_embedding = SinusoidalPositionEmbeddings(dim=time_emb_dim)
        # Multi-Layer Perceptron（多层感知机）
        # 正弦/余弦嵌入是固定的、非学习的（基于数学公式生成），只能提供时间步的基础周期性信息。
        # MLP中的线性层和激活函数是可学习的，能对原始嵌入进行非线性变换，提取更复杂的时间特征。
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 初始卷积层
        self.init_conv = nn.Conv2d(in_channels, self.down_channels[0], kernel_size=3, padding=1)

        # 下采样层
        self.downs = nn.ModuleList([])
        # in_ch = self.down_channels[0]
        # for out_ch in self.down_channels[1:]:
        #     self.downs.append(DownsampleBlock(in_ch, out_ch, time_emb_dim, downsample=True))
        #     in_ch = out_ch

        for i in range(len(self.down_channels) - 1):
            self.downs.append(
                DownsampleBlock(self.down_channels[i], self.down_channels[i+1], time_emb_dim, downsample=True))

        # bottleneck
        self.bottleneck = ResnetBlock(self.down_channels[-1], self.down_channels[-1], time_emb_dim)

        # 上采样层
        self.ups = nn.ModuleList([])
        in_ch = self.up_channels[0]
        for out_ch in self.up_channels[1:]:
            self.ups.append(UpsampleBlock(in_ch * 2, out_ch, time_emb_dim, upsample=True))
            in_ch = out_ch

        # 输出卷积层
        self.out_conv = nn.Conv2d(self.up_channels[-1], out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        Args:
            x: 输入图像张量，形状为 (batch_size, in_channels, height, width)
            t: 时间步张量，形状为 (batch_size,)
        Returns:

        """
        # print(x.shape)
        # print(t.shape)
        # 时间嵌入
        # [16, 128]
        t_emb = self.time_embedding(t)
        # [16, 128]
        t_emb = self.time_mlp(t_emb)

        # 初始卷积
        # [16, 64, 256, 256]
        x = self.init_conv(x)

        # 下采样 + 保存 skip connections
        skips = []
        for down in self.downs:
            x = down(x, t_emb)
            skips.append(x)

        # bottleneck
        x = self.bottleneck(x, t_emb)

        # 上采样 + 融合 skip connections
        for up, skip in zip(self.ups, reversed(skips)):
            x = torch.cat((x, skip), dim=1)
            x = up(x, t_emb)

        # 输出卷积
        x = self.out_conv(x)
        return x


if __name__ == "__main__":
    batch_size = 4
    time_steps = torch.arange(batch_size)  # 示例时间步张量 [0, 1, 2, 3]
    embedding_dim = 4  # 嵌入维度

    sample_data = torch.randn(batch_size, 3, 64, 64)

    model = Unet(in_channels=3, out_channels=3, base_channels=32, time_emb_dim=embedding_dim)
    out = model(sample_data, time_steps)

    write = writer = SummaryWriter("./logs/diff")
    write.add_graph(model, (sample_data, time_steps))
