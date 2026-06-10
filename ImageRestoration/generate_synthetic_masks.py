import math
import random
import time
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import matplotlib

matplotlib.use('TkAgg')  # 或 'Qt5Agg', 'Agg'


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
    # thick_min = max(1, int(min_hw * thick_min_weight))
    # thick_max = max(2, int(min_hw * thick_max_weight))
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
    mask = torch.zeros((h, w), dtype=torch.bool)

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
        print("brush shape", brush.shape)
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
    h, w = real_mask.shape[-2:]

    real_damage_ratio = real_mask.mean().item()
    print(f"Real damage ratio: {real_damage_ratio:.4f}")

    # 人造mask占全图比例上限
    max_artificial_damage_ratio = 0.4

    # 人造mask比例随着真实mask比例的增加而减少
    target_artificial_ratio = max(0.0, max_artificial_damage_ratio * (1.0 - real_damage_ratio / 0.5))

    print(f"Target artificial damage ratio: {target_artificial_ratio:.4f}")
    # 如果实际人造mask比例为0，直接返回真实mask，不添加人造mask
    if target_artificial_ratio <= 0:
        return torch.zeros_like(real_mask)

    # if random.random() < 1.0:
    # M_a = generate_dilated_mask(real_mask, max_kernel=15)
    # else:
    M_a = generate_irregular_mask(h, w)

    # 确保人工掩码不会覆盖真实掩码区域
    print("M_a", M_a.shape)
    print("real_mask", real_mask.shape)
    M_a = M_a * (1 - real_mask)

    plt.imshow(M_a.numpy().squeeze(), cmap='gray')
    plt.axis('on')
    plt.show()

    # 计算当前人工掩码比例
    # current_ratio = M_a.mean().item()
    # print(f"Artificial mask initial ratio: {current_ratio:.4f}, target: {target_artificial_ratio:.4f}")
    # 当前人工掩码数量
    current_count = M_a.sum().item()
    # 全图总像素数量
    total_count = M_a.numel()

    print(f"Artificial mask initial count: {current_count}, total count: {total_count}")
    # 目标人工掩码数量
    target_count = int(total_count * target_artificial_ratio)

    i = 0
    if current_count > target_count:
        i += 1
        print("进入", i)

        # base_block_size = 64
        # base_hw = 512
        # min_hw = min(h, w)
        #
        # block_size = int(base_block_size * (min_hw / base_hw)**0.5)
        # print("block_size", block_size)
        # scale_h = max(1, h // block_size)
        # scale_w = max(1, w // block_size)
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

        plt.imshow(spatial_noise.cpu().numpy(), cmap='gray')
        plt.axis('on')
        plt.show()

        # 根据噪声值在掩码区域内进行筛选，按照噪声值大小去掉threshold比例的掩码
        # 将当前掩码区域转化为布尔值，目的是忽略掩码外的噪声值（忽略掩码外区域，在当前掩码区域进行裁切）
        valid_mask_bool = M_a.bool().squeeze()
        print("valid_mask_bool", valid_mask_bool.shape)
        print("spatial_noise", spatial_noise.shape)
        # 提取掩码区域对应的噪声值
        valid_noise_values = spatial_noise[valid_mask_bool]
        print(valid_noise_values.shape)
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

        plt.imshow(selector.squeeze().cpu().numpy(), cmap='gray')
        plt.axis('on')
        plt.show()

        print("before *************", M_a.shape)
        print("selector *************", selector.shape)
        M_a = M_a * selector

        plt.imshow(M_a.squeeze().cpu().numpy(), cmap='gray')
        plt.axis('on')
        plt.show()

        print("last *************", M_a.shape)

    return M_a


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    real_mask = torch.zeros((1, 512, 512))
    # real_mask = torch.zeros((1, 256, 256))
    # real_mask = torch.zeros((1, 128, 128))

    # 生成一个占比0.3的真实掩码用于测试
    real_mask[:, 0:128, 0:128] = 1.0

    plt.imshow(real_mask.squeeze().numpy(), cmap='gray')
    plt.axis('on')
    plt.show()

    print(real_mask.shape)

    masks = []
    for _ in range(10):
        mask = generate_artificial_mask(real_mask)
        mask = torch.clamp(real_mask + mask, 0, 1)
        masks.append(mask)

    print(len(masks))
    # 画图显示十张masks
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    for ax, img in zip(axs.flatten(), masks):
        img = img.squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.axis('on')

    plt.tight_layout()
    plt.show()
