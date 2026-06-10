# from VQVAE import *
import math
import time
import netron
import torch
from torch import nn
from tensorboardX import SummaryWriter


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


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample,
                 num_heads, num_layers, norm_channels, attn=False, cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ) for i in range(num_layers)
        ])

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(self.t_emb_dim, out_channels)
                ) for _ in range(num_layers)
            ])

        # print("t_emb_layers", self.t_emb_layers)

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers)
        ])

        # print("first", self.resnet_conv_first)
        # print("second", self.resnet_conv_second)

        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ])

            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) \
            if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None, context=None):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)

            out = out + self.residual_input_conv[i](resnet_input)

            if self.attn:
                batch_size, channels, h, w = out.shape
                # in_attn = self.attention_norms[i](out)
                # in_attn = in_attn.reshape(batch_size, channels, h * w).transpose(1, 2)
                # out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                # out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                # out = out + out_attn

                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used."
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels,
                 cross_attn=False, context_dim=None):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim
        self.context_dim = context_dim
        self.cross_attn = cross_attn

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_layers + 1)
        ])

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ) for _ in range(num_layers + 1)
            ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers + 1)
        ])

        self.attention_norms = nn.ModuleList([
            nn.GroupNorm(norm_channels, out_channels)
            for _ in range(num_layers)
        ])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])

        if self.cross_attn:
            assert context_dim is not None, "Context Dimension must be passed for cross attention"
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers + 1)
        ])

    def forward(self, x, t_emb=None, context=None):
        out = x

        resnet_input = out
        out = self.resnet_conv_first[0](out)
        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)

        for i in range(self.num_layers):
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used."
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class UnetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample,
                 num_heads, num_layers, norm_channels, attn=False, cross_attn=False, context_dim=None):
        super().__init__()
        self.t_emb_dim = t_emb_dim
        self.up_sample = up_sample
        self.num_layers = num_layers
        self.attn = attn
        self.cross_attn = cross_attn
        self.context_dim = context_dim

        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_layers)
        ])

        if self.t_emb_dim is not None:
            self.t_emb_layers = nn.ModuleList([
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                ) for _ in range(num_layers)
            ])

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers)
        ])

        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ])

            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])

        if self.cross_attn:
            assert context_dim is not None, "Context dimension must be passed for cross attention."
            self.cross_attention_norms = nn.ModuleList(
                [nn.GroupNorm(norm_channels, out_channels)
                 for _ in range(num_layers)]
            )
            self.cross_attentions = nn.ModuleList(
                [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                 for _ in range(num_layers)]
            )
            self.context_proj = nn.ModuleList(
                [nn.Linear(context_dim, out_channels)
                 for _ in range(num_layers)]
            )

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2,
                                                 padding=1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None, context=None):
        # print("x input", x.shape)
        x = self.up_sample_conv(x)
        # print("x up_sample_conv", x.shape)
        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)
            # print("x concat", x.shape)

        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

            # Self-attention
            if self.attn:
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn
            # Cross-attention
            if self.cross_attn:
                assert context is not None, "context cannot be None if cross attention layers are used."
                batch_size, channels, h, w = out.shape
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.cross_attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                assert len(context.shape) == 3, \
                    "Context shape does not match B,_,CONTEXT_DIM"
                assert context.shape[0] == x.shape[0] and context.shape[-1] == self.context_dim, \
                    "Context shape does not match B,_,CONTEXT_DIM"
                context_proj = self.context_proj[i](context)
                out_attn, _ = self.cross_attentions[i](in_attn, context_proj, context_proj)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        return out


class ImageContextEncoder(nn.Module):
    def __init__(self, in_ch=1, context_dim=1280):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 320, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(320, context_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(context_dim, context_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(context_dim, context_dim, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, img):
        x = self.encoder(img)
        # print(x.shape)
        return x.flatten(2).transpose(1, 2)  # (B, H*W, C)


class Unet(nn.Module):
    def __init__(self, z_channels=4, line_cond=False, image_cross_attn=False):
        """
        Args:
            z_channels: 输入图像的通道数
        """
        super().__init__()
        # self.down_channels = [256, 384, 512, 768]
        self.down_channels = [320, 640, 1280, 1280]
        self.up_channels = list(reversed(self.down_channels))
        # self.mid_channels = [768, 512]
        self.mid_channels = [1280, 1280]
        # self.t_emb_dim = 512
        self.t_emb_dim = 320
        self.t_proj_dim = 1280
        self.down_sample = [True, True, True]
        self.up_sample = list(reversed(self.down_sample))
        self.num_down_layers = 2
        self.num_mid_layers = 2
        self.num_up_layers = 2
        self.attns = [True, True, True]
        self.cross_attn = [False, False, False]
        self.norm_channels = 32
        self.num_heads = 16
        self.conv_out_channels = 320
        self.context_dim = None

        # 时间嵌入层模块
        self.time_embedding = SinusoidalPositionEmbeddings(dim=self.t_emb_dim)

        # 时间嵌入投影模块
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_proj_dim),
            nn.SiLU(),
            nn.Linear(self.t_proj_dim, self.t_proj_dim),
        )

        self.line_cond = line_cond
        if self.line_cond:
            self.line_cond_in_channels = 3
            self.line_cond_out_channels = 4
            self.cond_conv_in = nn.Conv2d(in_channels=self.line_cond_in_channels,
                                          out_channels=self.line_cond_out_channels, kernel_size=1, bias=False)
            self.conv_in_concat = nn.Conv2d(z_channels + self.line_cond_out_channels,
                                            self.down_channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = nn.Conv2d(z_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.image_cross_attn = image_cross_attn
        if self.image_cross_attn:
            self.cross_attn = [True, True, True]
            self.context_dim = 1280

            self.image_context_encoder = ImageContextEncoder(in_ch=1, context_dim=self.context_dim)

        self.downs = nn.ModuleList([])
        # in_ch = self.down_channels[0]
        # for out_ch in self.down_channels[1:]:
        #     self.downs.append(DownBlock(in_ch, out_ch, self.t_emb_dim,))

        for i in range(len(self.down_channels) - 1):
            # print(self.down_channels[i], self.down_channels[i + 1])
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_proj_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        norm_channels=self.norm_channels,
                                        attn=self.attns[i],
                                        cross_attn=self.cross_attn[i],
                                        context_dim=self.context_dim))

        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_proj_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels,
                                      cross_attn=self.cross_attn[i],
                                      context_dim=self.context_dim))

        self.ups = nn.ModuleList([])
        for i in range(len(self.up_channels) - 1):
            # print(i, self.up_channels[i + 1],
            #       self.up_channels[i + 2] if i != len(self.up_channels) - 2 else self.conv_out_channels)
            self.ups.append(
                UnetUpBlock(self.up_channels[i + 1] * 2,
                            self.up_channels[i + 2] if i != len(self.up_channels) - 2 else self.conv_out_channels,
                            self.t_proj_dim,
                            up_sample=self.up_sample[i],
                            num_heads=self.num_heads,
                            num_layers=self.num_up_layers,
                            norm_channels=self.norm_channels,
                            attn=self.attns[i],
                            cross_attn=self.cross_attn[i],
                            context_dim=self.context_dim))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(self.norm_channels, self.conv_out_channels),
            nn.SiLU(),
            nn.Conv2d(self.conv_out_channels, z_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, t, cond_input=None):
        """
        Args:
            x: 输入图像张量，形状为 (batch_size, in_channels, height, width)
            t: 时间步张量，形状为 (batch_size,)
        Returns:

        """
        # print("x", x.shape)

        if self.line_cond:
            # print("cond", cond_input.shape)
            line_cond = torch.nn.functional.interpolate(cond_input, size=x.shape[2:])
            # print("line", line_cond.shape)
            line_cond = self.cond_conv_in(line_cond)
            # print("line cond conv", line_cond.shape)
            x = torch.cat([x, line_cond], dim=1)
            # print("x concat", x.shape)
            out = self.conv_in_concat(x)
            # print(out.shape)
        else:
            out = self.conv_in(x)
            # print(out.shape)

        if self.image_cross_attn:
            line_encoder = self.image_context_encoder(cond_input)
            # print("line context shape:", line_encoder.shape)

        t_emb = self.time_embedding(t)
        # print("time embedding:")
        # print(t_emb.shape)
        t_emb = self.t_proj(t_emb)
        # print(t_emb.shape)

        down_outs = []
        # print("Downsampling:")
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
            # print(f" After down block {idx}: {out.shape}")

        # print("所有down_outs的shape：", [d.shape for d in down_outs])

        # print("Bottleneck:")
        for idx, mid in enumerate(self.mids):
            out = mid(out, t_emb)
            # print(f" After mid block {idx}: {out.shape}")

        # print("Upsampling:")
        for idx, up in enumerate(self.ups):
            # print("===========================================")
            down_out = down_outs.pop()
            # print({idx}, down_out.shape, out.shape)
            out = up(out, down_out, t_emb)
            # print(f" After up block {idx}: {out.shape}")

        # print("Output layer:")
        out = self.conv_out(out)
        # print(out.shape)

        return out


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     batch_size = 1
#     time_steps = torch.arange(batch_size).to(device)  # 示例时间步张量 [0, 1, 2, 3]
#     print(time_steps)
#     embedding_dim = 4  # 嵌入维度
#
#     sample_data = torch.randn(batch_size, 3, 512, 512).to(device)
#     sample_line = torch.randn(batch_size, 1, 512, 512).to(device)
#
#     vae = VQVAE().to(device)
#     vae.load_state_dict(torch.load("./model/vqvae_autoencoder_best.pth"))
#     vae.eval()
#     for param in vae.parameters():
#         param.requires_grad = False
#
#     with torch.no_grad():
#         images, _, _ = vae.encode(sample_data)
#
#     print("vae_encode_output_shape", images.shape)
#
#     model = Unet(im_channels=3, line_cond=True).to(device)
#     print(model)
#     out = model(images, time_steps, cond_input=sample_line)
#     print(out.shape)
#
#     with torch.no_grad():
#         out = vae.decode(out)
#         print("vae_decode_output_shape", out.shape)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    time_steps = torch.arange(batch_size).to(device)
    images = torch.randn(batch_size, 4, 64, 64).to(device)
    sample_line = torch.randn(batch_size, 3, 512, 512).to(device)

    model = Unet(z_channels=4, line_cond=True, image_cross_attn=False).to(device)
    model.eval()
    print(model)

    out = model(images, time_steps, cond_input=sample_line)
    print(out.shape)

    # example_inputs = (images, time_steps, sample_line)
    # traced_model = torch.jit.trace(model, example_inputs)

    # torch.onnx.export(
    #     model,
    #     example_inputs,
    #     "unet_model.onnx",
    #     opset_version=17,
    #     input_names=['input_image', 'time_steps', 'cond_input'],
    #     output_names=['output_image'],
    #     dynamic_axes={
    #         'input_image': {0: 'batch_size', 2: 'height', 3: 'width'},
    #         'time_steps': {0: 'batch_size'},
    #         'cond_input': {0: 'batch_size', 2: 'height', 3: 'width'},
    #         'output_image': {0: 'batch_size', 2: 'height', 3: 'width'},
    #     }
    # )
    # # netron.start("unet_model.onnx")

    # writer = SummaryWriter("./logs/unet_model_graph")
    # with torch.no_grad():
    #     writer.add_graph(
    #         model,
    #         (images, time_steps, sample_line),
    #         use_strict_trace=False
    #     )
    #     writer.close()
