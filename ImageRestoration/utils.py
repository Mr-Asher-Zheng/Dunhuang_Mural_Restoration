import os
import random
import time
import math
import matplotlib
import numpy as np
import warnings
from datetime import datetime
from tqdm import tqdm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from matplotlib import pyplot as plt
from prefetch_generator import BackgroundGenerator
import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms, models
import torchvision.transforms as T
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore", message=".*libpng warning: iCCP: known incorrect sRGB profile*")

matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 示例图
def image_sample(dataloader):
    sample = next(iter(dataloader))
    print(sample["image"].shape, sample["mask"].shape)

    image = sample["image"][0]  # [3,H,W]
    line = sample["line"][0]  # [3,H,W]
    mask = sample["mask"][0]  # [1,H,W]

    M_real = sample["M_real"][0]
    image_mask = sample["image_mask"][0]

    image_np = image.permute(1, 2, 0).cpu().numpy()
    line_np = line.permute(1, 2, 0).cpu().numpy()

    mask_np = mask.squeeze(0).cpu().numpy()
    M_real_np = M_real.squeeze(0).cpu().numpy()
    image_mask_np = image_mask.permute(1, 2, 0).cpu().numpy()

    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    mean_line = np.array([0.5])
    std_line = np.array([0.5])

    image_np = (image_np * std) + mean
    image_mask_np = (image_mask_np * std) + mean

    line_np = (line_np * std_line) + mean_line

    combined = image_np.copy()
    combined[mask_np > 0] = [1, 0, 0]

    # titles = ["Image", "Mask", "Line", "Image + Mask", "Real Mask", "Image With Mask"]
    # images = [image_np, mask_np, line_np, combined, M_real_np, image_mask_np]

    titles = ["Image", "Mask", "Line"]
    images = [image_np, M_real_np, line_np]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, img, title in zip(axs.flatten(), images, titles):
        ax.imshow(img, cmap="gray" if title == "Mask" or "Real Mask" else None)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def generate_dilated_mask(real_mask, max_kernel=15):
    # 随机选择一个3~max_kernel的奇数卷积核大小
    k = random.choice([i for i in range(3, max_kernel + 1) if i % 2 == 1])
    # k = 15
    kernel = torch.ones((1, 1, k, k))

    dilated = torch.nn.functional.conv2d(
        real_mask.unsqueeze(0),
        kernel,
        padding=k // 2
    ) > 0

    # print("dilated", dilated.shape)
    # time.sleep(60)
    return dilated.squeeze(0)


def generate_zone_mask(h, w, min_hw, xx, yy, num_points,
                       thick_min_weight=0.005, thick_max_weight=0.02,
                       base_hw=512, base_sigma=8):
    """
    :param h: 输入图像的高度
    :param w: 输入图像的宽度
    :param num_points: 线段的关键点数量
    :param thick_min_weight: 线段厚度最小权重
    :param thick_max_weight: 线段厚度最大权重
    :param base_hw: 以512为基准的尺寸
    :param base_sigma: 基准尺寸下的高斯模糊sigma
    :return:
    """

    # thickness 范围（和尺寸相关）
    thick_min = min_hw * thick_min_weight
    thick_max = min_hw * thick_max_weight

    # 游走式生成点
    start_x = random.randint(0, w - 1)
    start_y = random.randint(0, h - 1)

    points = [(start_x, start_y)]
    for _ in range(num_points - 1):
        # 随机方向（0 到 2π）
        angle = random.uniform(0, 2 * math.pi)
        # 随机步长（和图像尺寸相关，最短为尺寸的 5%，最长为 15%）
        step = random.uniform(0.05, 0.15) * min_hw

        # 计算下一个点坐标
        px = points[-1][0] + step * math.cos(angle)
        py = points[-1][1] + step * math.sin(angle)

        # 防止出界
        px = max(0, min(w - 1, px))
        py = max(0, min(h - 1, py))

        # 添加点
        points.append((px, py))

    # 初始化 mask
    mask = torch.zeros(h, w, dtype=torch.bool)

    # 逐段连接
    # num_points 个点
    # num_points - 1 段线
    for i in range(num_points - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]

        # print(f"Line segment {i}: ({x0}, {y0}) to ({x1}, {y1})")

        # 方向向量
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            continue

        # print(f"  Direction vector: ({dx}, {dy}), Length: {length}")
        # time.sleep(60)

        # 单位方向
        ux = dx / length
        uy = dy / length

        # 法向量（垂直方向）
        nx = -uy
        ny = ux

        # 每一段单独随机 thickness
        thickness = random.uniform(thick_min, thick_max)

        # 点到直线的距离（法向投影）
        dist = torch.abs(
            (xx - x0) * nx + (yy - y0) * ny
        )

        # 点在线段方向上的投影长度
        proj = (xx - x0) * ux + (yy - y0) * uy

        # 有限线段约束
        segment = (proj >= 0) & (proj <= length)

        # 当前线段 mask
        line = (dist < thickness) & segment

        # 叠加
        mask |= line

    # sigma 按比例缩放
    sigma = int(base_sigma * (min_hw / base_hw))

    # kernel_size 按比例计算，保证覆盖完整
    kernel_size = 6 * sigma + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    # print(f"Applying Gaussian Blur with kernel_size={kernel_size}, sigma={sigma}")

    blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    mask = mask.float().unsqueeze(0)
    mask_smooth = blur(mask)
    mask = (mask_smooth > 0.5).squeeze()

    return mask


def generate_crack_mask(h, w, min_hw, xx, yy, num_lines=5, base_hw=512):
    crack = torch.zeros((h, w), dtype=torch.bool)

    for _ in range(num_lines):
        # 1. 随机起点
        x0 = random.randint(0, w - 1)
        y0 = random.randint(0, h - 1)

        # 2. 随机方向（裂纹常见：偏直，但允许扰动）
        angle = random.uniform(0, math.pi)

        # 3. 裂纹长度（短一些更像真实裂纹）
        length = random.uniform(0.05, 0.25) * min_hw

        # 4. 极细 thickness（关键）
        base = min_hw / base_hw
        thickness = random.uniform(0.3, 0.6) * base

        # 方向单位向量
        ux = math.cos(angle)
        uy = math.sin(angle)

        # 法向量
        nx = -uy
        ny = ux

        # 点到直线的法向距离
        dist = torch.abs(
            (xx - x0) * nx + (yy - y0) * ny
        )

        # 在线段方向上的投影
        proj = (xx - x0) * ux + (yy - y0) * uy

        # 有限线段约束
        segment = (proj >= 0) & (proj <= length)

        # 当前裂纹线
        line = (dist < thickness) & segment

        # 叠加（允许重叠）
        crack |= line

    return crack


def generate_irregular_mask(h, w, base_hw=512):
    """
    h, w: 掩码尺寸
    num_strokes: 随机刷子块数量
    max_size: 每个块的最大尺寸
    """
    min_hw = min(h, w)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = torch.zeros((h, w), dtype=torch.bool)

    # 随机进行矩形区域损坏
    if random.random() < 0.5:
        min_weight = 20 * (min_hw / base_hw)
        max_weight = 90 * (min_hw / base_hw)
        # print(f"min_weight: {min_weight}, max_weight: {max_weight}")

        # 随机中心点
        center_x = random.randint(0, w - 1)
        center_y = random.randint(0, h - 1)
        # 随机形状大小
        size_x = random.randint(int(min_weight), int(max_weight))
        size_y = random.randint(int(min_weight), int(max_weight))

        # 随机角度（-45° 到 45° 比较自然）
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # 矩形中心
        cx = center_x + size_x / 2
        cy = center_y + size_y / 2
        # 坐标平移
        x = xx - cx
        y = yy - cy
        # 反向旋转坐标
        x_rot = x * cos_a + y * sin_a
        y_rot = -x * sin_a + y * cos_a
        # 旋转矩形 mask
        brush = (
                (torch.abs(x_rot) <= size_x / 2) &
                (torch.abs(y_rot) <= size_y / 2)
        )

        mask = mask | brush

    # 大区域损坏
    brush = generate_zone_mask(h, w, min_hw, xx, yy, num_points=100,
                               thick_min_weight=0.005, thick_max_weight=0.02,
                               base_hw=base_hw, base_sigma=8)
    # 逻辑或 |，保证只要有刷子覆盖的像素就标记为 1
    mask = mask | brush

    # 小点和细点，模拟小区域以及细小区域的损坏
    brush = generate_zone_mask(h, w, min_hw, xx, yy, num_points=50,
                               thick_min_weight=0.001, thick_max_weight=0.005,
                               base_hw=base_hw, base_sigma=4)
    mask = mask | brush

    # 细线条，模拟裂纹刮痕
    brush = generate_crack_mask(h, w, min_hw, xx, yy, num_lines=5,
                                base_hw=base_hw)
    # mask = mask | brush

    return mask.float().unsqueeze(0)


def generate_artificial_mask(real_mask):
    # print("shape", real_mask.shape)
    h, w = real_mask.shape[-2:]

    # 真实mask占全图比例
    real_damage_ratio = real_mask.mean().item()

    # 人造mask占全图比例上限
    max_artificial_damage_ratio = 0.4

    # 人工损坏比例随着真实损坏比例增加而减少
    target_artificial_ratio = max(0.0, max_artificial_damage_ratio * (1.0 - real_damage_ratio / 0.5))

    # 如果实际人造mask比例为0，直接返回真实mask，不添加人造mask
    if target_artificial_ratio <= 0:
        return torch.zeros_like(real_mask)

    # # if random.random() < 0.7:
    # M_a = generate_dilated_mask(real_mask, max_kernel=15)
    # else:
    M_a = generate_irregular_mask(h, w)

    # 确保人工掩码不会覆盖真实掩码区域
    M_a = M_a * (1 - real_mask)

    # 当前人工掩码数量
    current_count = M_a.sum().item()
    # 全图总像素数量
    total_count = M_a.numel()
    # 目标人工掩码数量
    target_count = int(total_count * target_artificial_ratio)

    if current_count > target_count:
        dynamic_block_size = max(1, h // 8)
        scale_h = max(1, h // dynamic_block_size)
        scale_w = max(1, w // dynamic_block_size)

        low_res_noise = torch.rand((1, 1, scale_h, scale_w))
        spatial_noise = F.interpolate(
            low_res_noise,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        # 根据噪声值在掩码区域内进行筛选，按照噪声值大小去掉threshold比例的掩码
        # 将当前掩码区域转化为布尔值，目的是忽略掩码外的噪声值（忽略掩码外区域，在当前掩码区域进行裁切）
        valid_mask_bool = M_a.bool().squeeze()
        # 提取掩码区域对应区域的噪声值
        valid_noise_values = spatial_noise[valid_mask_bool]
        # 计算保留比例，例如当前有10个，要保留5个，保留比例就是5/10
        keep_ratio = target_count / current_count
        # 计算阈值，以下是一个不严谨的说明，仅用于理解quantile的作用
        # 例如：[0.1~0.9]，随机生成10个噪声值
        # 逻辑上先排好序[0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # 要保留50%，阈值就是0.5，对应的噪声值是0.5
        threshold = torch.quantile(valid_noise_values, keep_ratio)
        # 小于阈值的保留，大于阈值的去掉
        # 由于 spatial_noise 具有空间相关性，该筛选会产生连续区域而非离散噪点
        selector = spatial_noise <= threshold
        selector = selector.float().unsqueeze(0)
        M_a = M_a * selector
    return M_a


class RestorationMuralDataset(Dataset):
    def __init__(self, image_dir, line_dir, mask_dir, size=256, select=False):
        all_image_list = sorted(
            [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png', 'jpg'))])
        all_line_list = sorted([os.path.join(line_dir, f) for f in os.listdir(line_dir) if f.endswith(('png', 'jpg'))])
        all_mask_list = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg'))])

        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.LANCZOS),
        ])

        # [0, 1]
        # [-1, 1]
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.line_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

        if select:
            self.image_list = []
            self.line_list = []
            self.mask_list = []

            # 损坏区域达50%的图像不加入数据集
            for img_p, line_p, mask_p in zip(all_image_list, all_line_list, all_mask_list):
                with Image.open(mask_p) as mask:
                    mask_pil = mask.convert("L").resize((self.size, self.size))
                    # print("mask type:", type(mask))

                    mask_np = np.array(mask_pil)
                    # print("mask_np type:", type(mask_np))
                    damage_ratio = (mask_np > 128).mean()
                    # print(f"damage_ratio type: {type(damage_ratio)}, value: {damage_ratio}")
                    # time.sleep(60)

                    if damage_ratio < 0.5:
                        self.image_list.append(img_p)
                        self.line_list.append(line_p)
                        self.mask_list.append(mask_p)

            assert len(self.image_list) == len(self.mask_list) == len(self.line_list), "图像数量、线稿数量和mask数量不一致！"
        else:
            self.image_list = all_image_list
            self.line_list = all_line_list
            self.mask_list = all_mask_list

            assert len(self.image_list) == len(self.mask_list) == len(self.line_list), "图像数量、线稿数量和mask数量不一致！"

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # image (512, 512) <class 'PIL.Image.Image'> 0 255
        # line (512, 512) <class 'PIL.Image.Image'> 0 255
        # mask (512, 512) <class 'PIL.Image.Image'> 0 255

        # line (512, 512, 3)
        # mask (512, 512)

        # after transform:
        # image== torch.Size([3, 512, 512]) <class 'torch.Tensor'> tensor(-1.) tensor(1.)
        # line== torch.Size([3, 512, 512]) <class 'torch.Tensor'> tensor(-1.) tensor(1.)
        # mask== torch.Size([1, 512, 512]) <class 'torch.Tensor'> tensor(0.) tensor(1.)
        image = Image.open(self.image_list[idx]).convert("RGB")
        line = Image.open(self.line_list[idx]).convert("RGB")
        mask = Image.open(self.mask_list[idx]).convert("L")

        # print("image",
        #       image.size,
        #       type(image),
        #       np.min(np.array(image)),
        #       np.max(np.array(image)), )
        # print("line",
        #       line.size,
        #       type(line),
        #       np.min(np.array(line)),
        #       np.max(np.array(line)), )
        # print("mask",
        #       mask.size,
        #       type(mask),
        #       np.min(np.array(mask)),
        #       np.max(np.array(mask)), )
        #
        # print("=====")
        # print("line", np.array(line).shape)
        # print("mask", np.array(mask).shape)

        # 统一变换
        image = self.transform(image)
        line = self.transform(line)
        mask = self.transform(mask)
        # 分别进行图像和掩码的后续变换
        image = self.image_transform(image)
        line = self.line_transform(line)
        mask = self.mask_transform(mask)

        # print("after transform:")
        # print("image==",
        #       image.shape,
        #       type(image),
        #       torch.min(image),
        #       torch.max(image), )
        # print("line==",
        #       line.shape,
        #       type(line),
        #       torch.min(line),
        #       torch.max(line), )
        # print("mask==",
        #       mask.shape,
        #       type(mask),
        #       torch.min(mask),
        #       torch.max(mask), )

        # 二值化到0和1
        mask = (mask > 0.5).float()
        # mask = torch.clamp(mask, 0, 1)

        M_real = mask.clone()
        M_art = generate_artificial_mask(M_real)
        # # print("M_art_unique:", torch.unique(M_art))
        # # 掩码（真实加伪造）
        mask = M_real + M_art
        # 反掩码
        anti_mask = 1 - mask

        # Image_real = image.clone()
        # image 的掩码区域使用白色填充，完好区域保持原图
        image_mask = image * anti_mask + mask

        # image = self.transform(image)
        # image = self.image_transform(image)

        # mask_unique = torch.unique(mask)
        # print("mask_unique:", mask_unique)
        # anti_mask_unique = torch.unique(anti_mask)
        # print("anti_mask_unique:", anti_mask_unique)

        # print(mask.min(), mask.max(), mask.dtype)
        # time.sleep(60)

        # 残缺区域（白色）为1，完好区域（黑色）为0
        # mask_label = (mask > 0).float()

        return {"image": image, "line": line, "mask": mask, "anti_mask": anti_mask,
                "M_real": M_real, "image_mask": image_mask, "name": os.path.basename(self.image_list[idx])}


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class NoiseScheduler(nn.Module):
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=1000):
        """
        Args:
            beta_start: β1，初始噪声水平
            beta_end: βT，最终噪声水平
            num_steps: T，扩散步数
        """
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps

        # β_t: 线性噪声调度
        # self.betas 存的是噪声调度表（每一步要加多少噪声）
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_steps))
        # α_t = 1 - β_t
        # self.alphas 存的是每一步剩下的“保留率”。
        self.register_buffer('alphas', 1.0 - self.betas)

        # α_bar_t = ∏(1-β_i) from i=1 to t
        # 当前步的累计保留率，即从第1步到第t步，数据还剩下多少比例没有被噪声破坏，累计连乘积会越来越小，原图保留的比例也越来越小
        self.register_buffer('alpha_bar', torch.cumprod(self.alphas, dim=0))

        # 以下两行代码为什么要开方？
        # 因为在扩散模型的数学推导中，噪声的添加和去除涉及到标准差的计算，而标准差是方差的平方根。
        # sqrt(α_bar_t)
        # 控制「保留多少原图」
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(self.alpha_bar))
        # sqrt(1-α_bar_t)
        # 控制「加多少噪声」
        self.register_buffer('sqrt_one_minus_alpha_bar', torch.sqrt(1.0 - self.alpha_bar))

        # ===================== 上面是前向扩散需要的参数 =====================
        # ===================== 下面是反向扩散需要的参数 =====================
        # 1/sqrt(α_bar_t)
        self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1.0 / self.alpha_bar))
        # sqrt(1/α_bar_t - 1)
        self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1.0 / self.alpha_bar - 1))

        # α_bar_(t-1)
        # 前一步的累计保留率
        self.register_buffer('alpha_bar_prev', torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]], dim=0))
        # 1/sqrt(α_t)
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))

        # 后验分布方差 σ_t^2
        self.register_buffer('posterior_var', self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar))
        # 后验分布均值系数1: β_t * sqrt(α_bar_(t-1))/(1-α_bar_t)
        self.register_buffer('posterior_mean_coef1',
                             self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar))
        # 后验分布均值系数2: (1-α_bar_(t-1)) * sqrt(α_t)/(1-α_bar_t)
        self.register_buffer('posterior_mean_coef2',
                             (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar))

    def get(self, var, t, x_shape):
        """获取指定时间步的变量值并调整形状
        Args:
            var: 要查询的变量
            t: 时间步
            x_shape: 目标形状
        Returns:
            调整后的变量值
        """
        # print("t")
        # print(t)

        # 从变量张量中收集指定时间步的值
        out = var[t]

        # print("out")
        # print(out)

        # 调整形状为[batch_size, 1, 1, 1],以便进行广播
        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

    # def add_noise(self, x, t):
    #     """向输入添加噪声
    #     实现公式： x_t = sqrt(α_bar_t) * x_0 + sqrt(1-α_bar_t) * ε, ε ~ N(0,I)
    #     Args:
    #         x: 输入图像 x_0
    #         t: 时间步
    #     Returns:
    #         (noisy_x, noise): 加噪后的图像 x_t 和使用噪声 ε
    #     """
    #     # 获取时间步 t 对应的 sqrt(α_bar_t)，原图权重
    #     sqrt_alpha_bar = self.get(self.sqrt_alpha_bar, t, x.shape)
    #     # 获取时间步 t 对应的 sqrt(1-α_bar_t)，噪声权重
    #     sqrt_one_minus_alpha_bar = self.get(self.sqrt_one_minus_alpha_bar, t, x.shape)
    #
    #     # 从标准正态分布中采样噪声 ε~N(0,I)
    #     noise = torch.randn_like(x)
    #
    #     # print(sqrt_alpha_bar.shape)
    #     # print(sqrt_one_minus_alpha_bar.shape)
    #     # print(x.shape)
    #     # print(noise.shape)
    #
    #     # 实现前向扩散过程：x_t = sqrt(α_bar_t) * x_0 + sqrt(1-α_bar_t) * ε
    #     return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise

    def add_noise(self, x, noise, t):
        """向输入添加噪声
        实现公式： x_t = sqrt(α_bar_t) * x_0 + sqrt(1-α_bar_t) * ε, ε ~ N(0,I)
        Args:
            x: 输入图像 x_0
            noise: 使用的噪声 ε
            t: 时间步
        Returns:
            (noisy_x, noise): 加噪后的图像 x_t 和使用噪声 ε
        """
        # 获取时间步 t 对应的 sqrt(α_bar_t)，原图权重
        sqrt_alpha_bar = self.get(self.sqrt_alpha_bar, t, x.shape)
        # 获取时间步 t 对应的 sqrt(1-α_bar_t)，噪声权重
        sqrt_one_minus_alpha_bar = self.get(self.sqrt_one_minus_alpha_bar, t, x.shape)

        # print(sqrt_alpha_bar.shape)
        # print(sqrt_one_minus_alpha_bar.shape)
        # print(x.shape)
        # print(noise.shape)

        # 实现前向扩散过程：x_t = sqrt(α_bar_t) * x_0 + sqrt(1-α_bar_t) * ε
        return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    创建一个计划，其学习率在优化器中设置的初始 lr 与 0 之间的余弦函数值后降低，
    在预热期期间，学习率在 0 和优化器中设置的初始 lr 之间线性增加。

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        为其调度学习率的优化器。
        num_warmup_steps (:obj:`int`):
        预热阶段的步骤数。
        num_training_steps (:obj:`int`):
        训练步骤的总数。
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        余弦 schedule 中的波数（默认值是在半余弦之后从最大值减少到 0）。
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        恢复训练时最后一个 epoch 的索引。

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # t/T 这个比值实际上是一个比例因子，
        # 当t=0（当前步数为0），学习率为0
        # 当t=num_warmup_steps（设定的预热步数），学习率达到设定的最大

        # decadence衰退阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        # progress = (t-T)/(n-T)
        # 当t=T的时候，说明已经执行完预热步骤，progress=0
        # 当t=n的时候，说明训练要完成了，progress=1

        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
        # 当progress=0，返回1（比例因子），对应预热阶段结束，衰退阶段开始，最大学习率
        # 当progress=1，返回0（比例因子），对应衰退阶段结束，学习率为0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def psnr(img1, img2, max_value=1.0, eps=1e-5):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(max_value / torch.sqrt(mse + eps))


def predict_x0(x_t, t, noise_pred, noise_scheduler):
    sqrt_alpha_bar = (noise_scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = (1 - noise_scheduler.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)

    # print("sqrt_alpha_bar:", sqrt_alpha_bar, sqrt_one_minus_alpha_bar.shape)
    # print("sqrt_one_minus_alpha_bar:", sqrt_one_minus_alpha_bar, sqrt_one_minus_alpha_bar.shape)
    # print("x_t:", x_t.shape)
    # print("noise_pred:", noise_pred.shape)

    x_0 = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
    return x_0


def plot_diffusion_steps(image, noise_scheduler, step_size=100):
    """绘制图像逐步加噪的过程
    Args:
        image: 原始图像
        noise_scheduler: 噪声调度器
        step_size: 每隔多少步绘制一次
    Returns:
        fig: 绘制的图像
    """
    num_images = noise_scheduler.num_steps // step_size
    # 创建一个图形对象，设置画布尺寸为宽15英寸、高3英寸
    fig = plt.figure(figsize=(15, 3))

    # 绘制原始图像
    # 在1行(num_images+1)列的网格中绘制第1个子图（原始图像）
    plt.subplot(1, num_images + 1, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original')

    # 绘制不同时间步的噪声图像
    for idx in range(num_images):
        t = torch.tensor([idx * step_size])
        # 分别绘制在t时刻的加噪图像
        noisy_image, _ = noise_scheduler.add_noise(image, t)
        plt.subplot(1, num_images + 1, idx + 2)
        plt.imshow(noisy_image)
        plt.axis('off')
        plt.title(f't={t.item()}')

    # 自动调整子图间距
    plt.tight_layout()
    plt.show()
    # return fig
