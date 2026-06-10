import argparse
import sys
from tqdm import tqdm
from datetime import datetime
import time

from image_seg_utils import *
from SAM_Adapter_Model import *


def train_and_val(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device,
                  total_steps, valid_steps, accum_steps):
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    writer = SummaryWriter(log_dir=f"./logs/Seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_iou = 0.0

    train_iterator = iter(train_loader)
    train_loader_tqdm = tqdm(range(total_steps), desc="Train", leave=False)

    for step in train_loader_tqdm:
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        # 获取一个batch的图像数据并移至指定设备
        # [16, 3, 256, 256]
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        with torch.amp.autocast(device_type='cuda'):
            # 使用模型预测噪声
            preds = model(images)

            # print(preds.shape, masks.shape)
            # time.sleep(60)

            # 计算预测噪声与真实噪声之间的损失
            loss = criterion(preds, masks)

            loss = loss / accum_steps  # 梯度累积，损失也要相应缩小

        scaler.scale(loss).backward()  # 反向传播

        # 每 accum_steps 个 mini-batch 更新一次参数（模拟更大的 batch size）
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)  # 将混合精度缩放后的梯度还原（unscale）回正常范围，否则梯度裁剪会基于错误的数值进行
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，限制梯度范数，防止梯度爆炸
            scaler.step(optimizer)  # 使用 scaler 执行优化器 step （内部会自动处理梯度缩放）
            scaler.update()  # 更新混合精度的缩放因子
            scheduler.step()  # 更新学习率
            optimizer.zero_grad(set_to_none=True)  # 清空梯度，set_to_none=True 可以减少显存占用

        # 累计损失
        train_losses = loss.item() * accum_steps  # 还原回正常的损失值
        # 显示当前轮次的累计损失
        train_loader_tqdm.set_postfix(loss=train_losses, step=step + 1)
        writer.add_scalar('Loss/Train', train_losses, step + 1)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], step + 1)

        if (step + 1) % valid_steps == 0:
            model.eval()
            val_losses = 0.0

            total_iou = 0.0
            total_images = 0

            val_loader_tqdm = tqdm(val_loader, desc="Valid", leave=False)
            with torch.no_grad():
                for i, batch in enumerate(val_loader_tqdm):
                    images = batch["image"].to(device)
                    masks = batch["mask"].to(device)

                    preds = model(images)
                    loss = criterion(preds, masks)
                    val_losses += loss.item()

                    preds = torch.sigmoid(preds)
                    preds = (preds > 0.5).float()

                    # IoU = |A∩B| / |A∪B|
                    #     = |A∩B| / (|A|+|B|−|A∩B|)
                    # ∣A∩B∣
                    intersection = (preds * masks).sum(dim=(1, 2, 3))
                    # |A|+|B|
                    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
                    # IoU
                    iou_per_image = (intersection + 1e-6) / (union - intersection + 1e-6)

                    total_iou += iou_per_image.sum().item()
                    total_images += images.size(0)

                    iou = total_iou / total_images

                    val_loader_tqdm.set_postfix(loss=val_losses / (i + 1), iou=iou, batch=i + 1)

            model.train()

            val_loss = val_losses / len(val_loader)

            writer.add_scalar('Loss/Val', val_loss, step + 1)
            writer.add_scalar('IOU/Val', iou, step + 1)

            if iou > best_iou:
                torch.save(model.state_dict(), f"./checkpoints/best_iou_model.pth")
                best_iou = iou
                print(f"\nBest Test Model Saved at Epoch {step + 1}")

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"./checkpoints/best_test_model.pth")
                best_val_loss = val_loss
                print(f"\nBest Test Loss Model Saved at Epoch {step + 1}")

        # 每500步保存一次模型
        if (step + 1) % 500 == 0:
            torch.save(model.state_dict(), f"./checkpoints/model_step_{step + 1}.pth")
            print(f"\nModel Saved at Step {step + 1}")

    writer.close()


def get_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_image_dir', type=str, default="../../MuralDH/MuralDH/Mural_seg/train/images",
                        help='Path to training images')
    parser.add_argument('--train_mask_dir', type=str, default="../../MuralDH/MuralDH/Mural_seg/train/labels",
                        help='Path to training masks')
    parser.add_argument('--val_image_dir', type=str, default="../../MuralDH/MuralDH/Mural_seg/test/images",
                        help='Path to validation images')
    parser.add_argument('--val_mask_dir', type=str, default="../../MuralDH/MuralDH/Mural_seg/test/labels",
                        help='Path to validation masks')

    parser.add_argument('--model_type', type=str, default='vit_b',
                        help='Type of SAM model to use (e.g., vit_b, vit_l, vit_h)')
    parser.add_argument('--sam_model_path', type=str, default='./checkpoints/sam_vit_b_01ec64.pth',
                        help='Path to SAM model checkpoint (e.g., sam_vit_b_01ec64.pth, sam_vit_l_0b3195.pth, sam_vit_h_4b8939.pth)')

    parser.add_argument('--num_steps', type=int, default=3000, help='Total training steps')
    parser.add_argument('--accum_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--valid_steps', type=int, default=25, help='Validation steps interval')

    parser.add_argument('--image_size', type=int, default=512, help='Input image size for training and validation')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')

    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')

    if len(sys.argv) == 1:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_train_args()

    seeds(args.seed)

    # 检查当前系统是否有可用的 GPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}, if want to use GPU, please ensure CUDA is properly installed and configured.")

    train_dataset = SegMuralDataset(args.train_image_dir, args.train_mask_dir, size=args.image_size)
    train_dataloader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataset = SegMuralDataset(args.val_image_dir, args.val_mask_dir, size=args.image_size)
    val_dataloader = DataLoaderX(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # image_sample(train_dataloader)
    # time.sleep(60)

    # 查看dataloader数量
    print(len(train_dataloader))
    print(len(val_dataloader))

    accum_steps = args.accum_steps
    total_steps = args.num_steps
    scheduler_steps = int(total_steps // accum_steps)
    warmup_steps = int(0.1 * scheduler_steps)

    model = SAMAdapterModel(model_type=args.model_type, checkpoint=args.sam_model_path)
    print(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-5)

    # criterion = BCEDiceLoss()
    criterion = FocalTversky_BCE_Loss()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, scheduler_steps)
    scaler = torch.amp.GradScaler()  # 混合精度

    train_and_val(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, scaler, device,
                  total_steps=total_steps, valid_steps=args.valid_steps, accum_steps=accum_steps)
