import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# class VectorQuantizerEMA(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
#         """
#         初始化 EMA Vector Quantizer
#
#         :param num_embeddings: Codebook 的大小 (K)
#         :param embedding_dim: 嵌入向量的维度 (D)
#         :param commitment_cost: 用于约束 Encoder 输出不偏离 Codebook 太远的 Loss 系数
#         :param decay: EMA 的衰减率 (gamma)，通常为 0.99
#         :param epsilon: 防止除以零的小数值
#         """
#         super(VectorQuantizerEMA, self).__init__()
#
#         self._embedding_dim = embedding_dim
#         self._num_embeddings = num_embeddings
#
#         self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
#         self._embedding.weight.data.normal_()
#         self._embedding.weight.requires_grad = False  # 不更新嵌入向量的梯度
#
#         self._commitment_cost = commitment_cost
#
#         self._decay = decay
#         self._epsilon = epsilon
#
#         # _ema_cluster_size: 记录每个嵌入向量被选中的次数
#         self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
#         # _ema_w: 记录分配给每个嵌入向量的输入向量之和
#         self.register_buffer('_ema_w', self._embedding.weight.data.clone())
#
#     def forward(self, inputs):
#         # (B, C, H, W) -> (B, H, W, C)
#         inputs = inputs.permute(0, 2, 3, 1).contiguous()
#         input_shape = inputs.shape
#
#         # (B, H, W, C) -> (B*H*W, C)
#         flat_input = inputs.view(-1, self._embedding_dim)
#
#         # 计算输入与嵌入向量之间的距离
#         distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
#                      + torch.sum(self._embedding.weight ** 2, dim=1)
#                      - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
#
#         # 获取最近的嵌入向量索引
#         encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
#
#         # 生成独热编码
#         encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
#         encodings.scatter_(1, encoding_indices, 1)
#
#         # 量化，根据索引从 Codebook 中取出向量
#         quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
#
#         # EMA 更新
#         if self.training:
#             # 计算当前 batch 每个 codebook vector 被选中的次数
#             _cluster_size = torch.sum(encodings, dim=0)
#             # 计算当前 batch 分配给每个 codebook vector 的输入向量之和
#             _dw = torch.matmul(encodings.t(), flat_input)
#
#             # 更新 Cluster Size
#             # N_new = decay * N_old + (1 - decay) * N_current
#             self._ema_cluster_size.data.mul_(self._decay).add_(_cluster_size, alpha=1 - self._decay)
#
#             # 更新 Embedding Sum
#             # W_sum_new = decay * W_sum_old + (1 - decay) * W_sum_current
#             self._ema_w.data.mul_(self._decay).add_(_dw, alpha=1 - self._decay)
#
#             # 归一化
#             # Weight = EMA_Sum / EMA_Count
#             # n 是所有 Codebook Vector 被选中的总次数
#             n = torch.sum(self._ema_cluster_size.data)
#
#             # 代码: self._ema_cluster_size + self._epsilon
#             # 作用: 为每个 Codebook 向量的使用计数 N_i 都加上一个极小的正数 ε。
#             # 目的: 防止某个向量 i 从未被选中，导致 N_i = 0。
#             # 如果 N_i=0，在最后一步除法中就会出现 m_i / 0，结果为 NaN 或 Inf，使训练崩溃。
#             # 拉普拉斯平滑确保了所有向量都有一个极小的、非零的权重。
#
#             _ema_cluster_size = (
#                     (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
#             )
#
#             self._embedding.weight.data.copy_(self._ema_w / _ema_cluster_size.unsqueeze(1))
#
#         # Loss
#         # 计算encoder输出(即inputs)和decoder输入(即quantized)之间的损失
#         e_latent_loss = F.mse_loss(quantized.detach(), inputs)
#         loss = self._commitment_cost * e_latent_loss
#
#         # 直通估计 Straight Through Estimator
#         # trick, 将decoder的输入对应的梯度复制，作为encoder的输出对应的梯度
#         quantized = inputs + (quantized - inputs).detach()
#
#         # 恢复形状 (B, H, W, C) -> (B, C, H, W)
#         avg_probs = torch.mean(encodings, dim=0)
#         perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-5)))
#
#         return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        """
        初始化 EMA Vector Quantizer

        :param num_embeddings: Codebook 的大小 (K)
        :param embedding_dim: 嵌入向量的维度 (D)
        :param commitment_cost: 用于约束 Encoder 输出不偏离 Codebook 太远的 Loss 系数
        :param decay: EMA 的衰减率 (gamma)，通常为 0.99
        :param epsilon: 防止除以零的小数值
        """
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._embedding.weight.requires_grad = False  # 不更新嵌入向量的梯度

        self._commitment_cost = commitment_cost

        self._decay = decay
        self._epsilon = epsilon

        # _ema_cluster_size: 记录每个嵌入向量被选中的次数
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        # _ema_w: 记录分配给每个嵌入向量的输入向量之和
        self.register_buffer('_ema_w', self._embedding.weight.data.clone())

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, inputs):
        input_dtype = inputs.dtype
        inputs = inputs.float()

        # (B, C, H, W) -> (B, H, W, C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # (B, H, W, C) -> (B*H*W, C)
        flat_input = inputs.view(-1, self._embedding_dim)

        codebook_weight = self._embedding.weight.float()

        # 计算输入与嵌入向量之间的距离
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # 获取最近的嵌入向量索引
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # 生成独热编码
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 量化，根据索引从 Codebook 中取出向量
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # EMA 更新
        if self.training:
            # 计算当前 batch 每个 codebook vector 被选中的次数
            _cluster_size = torch.sum(encodings, dim=0)
            # 计算当前 batch 分配给每个 codebook vector 的输入向量之和
            _dw = torch.matmul(encodings.t(), flat_input)

            # 更新 Cluster Size
            # N_new = decay * N_old + (1 - decay) * N_current
            self._ema_cluster_size.data.mul_(self._decay).add_(_cluster_size, alpha=1 - self._decay)

            # 更新 Embedding Sum
            # W_sum_new = decay * W_sum_old + (1 - decay) * W_sum_current
            self._ema_w.data.mul_(self._decay).add_(_dw, alpha=1 - self._decay)

            # 归一化
            # Weight = EMA_Sum / EMA_Count
            # n 是所有 Codebook Vector 被选中的总次数
            n = torch.sum(self._ema_cluster_size.data)

            # 代码: self._ema_cluster_size + self._epsilon
            # 作用: 为每个 Codebook 向量的使用计数 N_i 都加上一个极小的正数 ε。
            # 目的: 防止某个向量 i 从未被选中，导致 N_i = 0。
            # 如果 N_i=0，在最后一步除法中就会出现 m_i / 0，结果为 NaN 或 Inf，使训练崩溃。
            # 拉普拉斯平滑确保了所有向量都有一个极小的、非零的权重。

            _ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n
            )

            self._embedding.weight.data.copy_(self._ema_w / _ema_cluster_size.unsqueeze(1))

        # Loss
        # 计算encoder输出(即inputs)和decoder输入(即quantized)之间的损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # 直通估计 Straight Through Estimator
        # trick, 将decoder的输入对应的梯度复制，作为encoder的输出对应的梯度
        quantized = inputs + (quantized - inputs).detach()

        # 恢复形状 (B, H, W, C) -> (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        if input_dtype != torch.float32:
            quantized = quantized.to(input_dtype)

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-5)))

        return loss, quantized, perplexity, encoding_indices


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim,
                 down_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn
        self.t_emb_dim = t_emb_dim

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

        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(norm_channels, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers)
        ])

        # Multi-Head Self-Attention
        if self.attn:
            self.attention_norms = nn.ModuleList([
                nn.GroupNorm(norm_channels, out_channels)
                for _ in range(num_layers)
            ])

            self.attentions = nn.ModuleList([
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                for _ in range(num_layers)
            ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) \
            if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None):
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
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                # [b, c, h*w] -> [b, h*w, c]
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        out = self.down_sample_conv(out)
        return out


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.t_emb_dim = t_emb_dim

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
            nn.GroupNorm(norm_channels, out_channels) for _ in range(num_layers)
        ])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ])

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers + 1)
        ])

    def forward(self, x, t_emb=None):
        out = x
        resnet_input = out
        # 第一个 Resnet Block
        out = self.resnet_conv_first[0](out)

        if self.t_emb_dim is not None:
            out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]

        # 第二个 Resnet Block
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

            resnet_input = out
            out = self.resnet_conv_first[i + 1](out)
            if self.t_emb_dim is not None:
                out = out + self.t_emb_layers[i + 1](t_emb)[:, :, None, None]
            out = self.resnet_conv_second[i + 1](out)
            out = out + self.residual_input_conv[i + 1](resnet_input)

        return out


class VQVAEUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample,
                 num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.t_emb_dim = t_emb_dim
        self.attn = attn
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

        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1) \
            if self.up_sample else nn.Identity()

    def forward(self, x, out_down=None, t_emb=None):
        x = self.up_sample_conv(x)

        if out_down is not None:
            x = torch.cat([x, out_down], dim=1)

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
                in_attn = out.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                out = out + out_attn

        return out


class VQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        im_channels = 3
        self.down_channels = [128, 256, 512, 512]
        self.mid_channels = [512, 512]
        self.down_sample = [True, True, True]
        self.num_down_layers = 2
        self.num_mid_layers = 2
        self.num_up_layers = 2
        self.norm_channels = 32

        # 禁用 Encoder 的 Downblock 和 Decoder 的 Upblock 中的注意力机制
        self.attns = [False, False, False]
        # self.num_heads = 4
        self.num_heads = 8

        # Latent Dimension
        self.z_channels = 4
        self.codebook_size = 8192

        self.up_sample = list(reversed(self.down_sample))

        # ===================== Encoder =====================
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, stride=1, padding=(1, 1))

        # Downblock + Midblock
        self.encoder_downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                t_emb_dim=None, down_sample=self.down_sample[i],
                                                num_heads=self.num_heads,
                                                num_layers=self.num_down_layers,
                                                attn=self.attns[i],
                                                norm_channels=self.norm_channels))

        # print("Encoder Downblocks:", self.encoder_downs)

        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))

        # print("Encoder Midblocks:", self.encoder_mids)

        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)

        # Pre Quantization Convolution
        self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)

        # # Codebook
        # self.embedding = nn.Embedding(self.codebook_size, self.z_channels)

        self.quantize = VectorQuantizerEMA(
            num_embeddings=self.codebook_size,
            embedding_dim=self.z_channels,
            commitment_cost=0.25,
            decay=0.99,
            epsilon=1e-5
        )

        # ======================= Decoder =====================
        # Post Quantization Convolution
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.down_channels[-1], kernel_size=3, padding=(1, 1))

        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))


        self.decoder_ups = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_ups.append(VQVAEUpBlock(self.down_channels[i], self.down_channels[i - 1],
                                                 t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_up_layers,
                                                 attn=self.attns[i - 1],
                                                 norm_channels=self.norm_channels))

        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], im_channels, kernel_size=3, padding=1)

    # def quantize(self, x):
    #     B, C, H, W = x.shape
    #
    #     # B, C, H, W -> B, H, W, C
    #     x = x.permute(0, 2, 3, 1)
    #
    #     # B, H, W, C -> B, H*W, C
    #     x = x.reshape(x.size(0), -1, x.size(-1))
    #
    #     # find nearest embedding/codebook vector
    #     # 查找最近的嵌入码本向量
    #     # dist between (B, H*W, C) and (B, K, C) -> (B, H*W, K)
    #     dist = torch.cdist(x, self.embedding.weight[None, :].repeat((x.size(0), 1, 1)))
    #     # (B, H*W)
    #     min_encoding_indices = torch.argmin(dist, dim=-1)
    #
    #     # replace encoder output with nearest codebook
    #     # 用最近的码本替换编码器输出
    #     # quant_out -> B*H*W, C
    #     quant_out = torch.index_select(self.embedding.weight, 0, min_encoding_indices.view(-1))
    #
    #     # x -> B*H*W, C
    #     x = x.reshape((-1, x.size(-1)))
    #     commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
    #     codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
    #     quantize_losses = {
    #         'codebook_loss': codebook_loss,
    #         'commitment_loss': commitment_loss
    #     }
    #     # straight through estimation
    #     # 直通估计
    #     quant_out = x + (quant_out - x).detach()
    #
    #     # quant_out -> B, C, H, W
    #     quant_out = quant_out.reshape((B, H, W, C)).permute(0, 3, 1, 2)
    #     return quant_out, quantize_losses

    def encode(self, x):
        # print("x shape in encode:", x.shape)
        out = self.encoder_conv_in(x)
        # print("after encoder_conv_in:", out.shape)
        for idx, down in enumerate(self.encoder_downs):
            out = down(out)
            # print(f"after encoder_downs[{idx}]:", out.shape)
        for mid in self.encoder_mids:
            out = mid(out)
            # print("after encoder_mids:", out.shape)


        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)

        # out, quant_losses = self.quantize(out)
        # return out, quant_losses

        quant_loss, out, perplexity, _ = self.quantize(out)

        return out, quant_loss, perplexity

    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_ups):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out

    def forward(self, x):
        # z, quant_losses = self.encode(x)
        z, quant_loss, perplexity = self.encode(x)

        out = self.decode(z)

        # return out, z, quant_losses
        return out, z, quant_loss, perplexity


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((1, 3, 512, 512)).to(device)

    # dataset = torch.utils.data.TensorDataset(x)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    model = VQVAE().to(device)
    print(model)
    model.eval()
    output, z, quantize_losses, perplexity = model(x)
    print("==================================")
    print("output shape:", output.shape)
    print("z shape:", z.shape)

    # writer = SummaryWriter("./logs/vqvae_graph")
    # with torch.no_grad():
    #     writer.add_graph(model, x)
    #     writer.close()



#
#     # 4. 定义优化器
#     learning_rate = 1e-4
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#     print(model)
#     # output, z, quantize_losses = model(x)
#     # output, quant_losses = model.encode(x)
#
#     # 使用混合精度进行前向传播
#     # 初始化 GradScaler (用于训练中的梯度缩放，虽然这里仅演示前向，但在训练脚本中是必须的)
#     scaler = torch.amp.GradScaler('cuda')
#
#     train_iterator = iter(dataloader)
#     loader_tqdm = tqdm(range(10))
#
#     for step in loader_tqdm:
#         try:
#             batch = next(train_iterator)
#         except StopIteration:
#             train_iterator = iter(dataloader)
#             batch = next(train_iterator)
#
#         images = batch[0].to(device)
#
#         optimizer.zero_grad()
#
#         with torch.amp.autocast('cuda', dtype=torch.float16):
#             output, z, quant_loss, perplexity = model(images)
#             recon_loss = F.mse_loss(output, images)
#             total_loss = recon_loss + quant_loss
#
#         scaler.scale(total_loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         loader_tqdm.set_description(f"Step {step}: Total Loss: {total_loss.item():.6f}")
#
#     # output, quant_loss, perplexity = model.encode(x)
#     print(f"Input shape: {x.shape}")
#     print(f"Output shape: {output.shape}")
#     print(f"Latent z shape: {z.shape}")
#     print(f"Quantization Loss: {quant_loss.item():.6f}")
#     print(f"Reconstruction Loss: {recon_loss.item():.6f}")
#     print(f"Perplexity: {perplexity.item():.6f}")
#     print(f"Output dtype: {output.dtype}")  # 应该是 float16 (如果最后没转回的话)
