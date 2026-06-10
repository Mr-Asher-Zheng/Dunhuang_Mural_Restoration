import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Rather than taking IMG_CHANNELS x IMG_H x IMG_W all the way to
    1 scalar value , we instead predict grid of values.
    Where each grid is prediction of how likely
    the discriminator thinks that the image patch corresponding
    to the grid cell is real
    """
    def __init__(self, im_channels=3,
                 conv_channels=[64, 128, 256, 512],
                 kernels=[4, 4, 4, 4, 4],
                 strides=[2, 2, 2, 1, 1],
                 paddings=[1, 1, 1, 1, 1]):
        super().__init__()
        self.im_channels = im_channels
        activation = nn.LeakyReLU(0.2)
        # [3, 64, 128, 256, 512, 1]
        layers_dim = [self.im_channels] + conv_channels + [1]

        self.layers = nn.ModuleList()
        for i in range(len(layers_dim) - 1):
            is_first = (i == 0)
            is_last = (i == len(layers_dim) - 2)

            # 第一层和最后一层不使用归一化，中间层使用归一化
            use_norm = (not is_first) and (not is_last)
            # 最后一层不使用激活函数，中间层使用激活函数
            use_act = (not is_last)
            # 第一层使用偏置，其他层如果使用归一化则不使用偏置，否则使用偏置
            bias = not use_norm

            block = nn.Sequential(
                nn.Conv2d(layers_dim[i], layers_dim[i + 1],
                          kernel_size=kernels[i],
                          stride=strides[i],
                          padding=paddings[i],
                          bias=bias),
                nn.GroupNorm(32, layers_dim[i + 1]) if use_norm else nn.Identity(),
                activation if use_act else nn.Identity()
            )
            self.layers.append(block)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


if __name__ == "__main__":
    # 创建值在[-1,1]之间的随机图像张量，形状为(2,3,256,256)
    x = torch.randn(1, 3, 512, 512)

    model = Discriminator(im_channels=3)
    print(model)
    prob = model(x)
    print(prob.shape)
    # 查看取值范围：
    print(prob.min().item(), prob.max().item())
