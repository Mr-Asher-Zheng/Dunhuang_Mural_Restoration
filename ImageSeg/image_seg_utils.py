import os
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from prefetch_generator import BackgroundGenerator
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import math
from torch import nn

import warnings

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


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)

        preds = torch.sigmoid(preds)
        # 计算每个样本的交集和并集
        # 交集
        # (preds * targets) 表示 预测与真实的重叠区域
        # 然后对 [C,H,W] 这三个维度求和，每个 batch 样本得到一个总的“交集面积”
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        # 并集
        # preds.sum(...)：预测的“前景面积”（概率加总）。
        # targets.sum(...)：真实标签的前景面积（像素值 0/1 加总）。
        # 两者相加 ≈ 预测面积 + 真实面积。
        union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

        # 计算每个样本的 Dice 系数并取平均
        # 2 * intersection 是 2×|A ∩ B| + ε
        # union + epsilon 是 |A|+|B| + ε
        epsilon = 1e-5
        dice_loss = 1 - (2. * intersection + epsilon) / (union + epsilon)
        dice = dice_loss.mean()

        return bce_loss + dice


# 若更在意召回（减少漏检），把 beta 调大（例如 alpha=0.3, beta=0.7）。
# 若更在意精度（减少误报），把 alpha 调大。
# gamma 常在 [0.5, 2.0] 范围内试验：越大更专注难样本，但也可能导致训练不稳定。
# 常把 FocalTversky 与其他损失（比如 BCE）结合使用，特别是在早期训练阶段，这有助于稳定梯度。
class BoundaryLoss(nn.Module):
    pass


class FocalTversky_BCE_Loss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.BCE = nn.BCEWithLogitsLoss()
        # self.Boundary = BoundaryLoss()

    def forward(self, preds, targets):
        bce_loss = self.BCE(preds, targets)

        # boundary_loss = self.Boundary(preds, targets)

        preds = torch.sigmoid(preds)

        smooth = 1e-5
        # preds = preds.view(-1)
        # targets = targets.view(-1)

        # 如果 target 是 1，pred 是 1：
        # 这是 TP（预测正确）。
        TP = (preds * targets).sum(dim=(1, 2, 3))
        # 如果 target 为 0（背景），pred 为 1：
        # 这是 FP（误检）。
        FP = ((1 - targets) * preds).sum(dim=(1, 2, 3))
        # 如果 target 为 1，但 pred 为 0：
        # 这是 FN（漏检）。
        FN = (targets * (1 - preds)).sum(dim=(1, 2, 3))

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        focal_tversky = (1 - Tversky).pow(self.gamma).mean()
        return 0.3 * bce_loss + 0.7 * focal_tversky


class SegMuralDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=256):
        self.image_list = sorted([os.path.join(image_dir, f)
                                  for f in os.listdir(image_dir) if f.endswith(('png', 'jpg'))])
        self.mask_list = sorted([os.path.join(mask_dir, f)
                                 for f in os.listdir(mask_dir) if f.endswith(('png', 'jpg'))])

        assert len(self.image_list) == len(self.mask_list), "图像数量和mask数量不一致！"

        self.size = size

        self.transform = A.Compose([
            A.Resize(size, size, interpolation=cv2.INTER_LANCZOS4),

            # 翻转和旋转
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            # 模糊处理
            A.GaussianBlur(p=0.5),
            # 亮度对比度调整
            A.RandomBrightnessContrast(p=0.5),
            # 颜色抖动
            A.ColorJitter(p=0.5),

            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

        # self.transform = transforms.Compose([
        #     transforms.Resize((size, size)),
        # ])
        #
        # self.image_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ])
        #
        # self.mask_transform = transforms.Compose([
        #     transforms.ToTensor(),
        # ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 范围[0, 255]
        mask = cv2.imread(self.mask_list[idx], 0)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        # 范围[0, 255]
        mask = transformed['mask']
        # 范围[0, 1]
        mask = (mask > 0).float()  # 二值化掩码
        mask = mask.unsqueeze(0)  # 添加通道维度，变为[1,H,W]

        # image = Image.open(self.image_list[idx]).convert("RGB")
        # mask = Image.open(self.mask_list[idx]).convert("L")
        #
        # # 统一变换
        # image = self.transform(image)
        # mask = self.transform(mask)
        # # 分别进行图像和掩码的后续变换
        # image = self.image_transform(image)
        # mask = self.mask_transform(mask)

        # print(mask.min(), mask.max(), mask.dtype)
        # time.sleep(60)

        return {"image": image, "mask": mask}


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


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
