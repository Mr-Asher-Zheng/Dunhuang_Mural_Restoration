import sys
from tqdm import tqdm
from datetime import datetime
import time

from image_seg_utils import *
from SAM_Adapter_Model import *


def train_and_val(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device,
                  total_steps, valid_steps, accum_steps):
    os.makedirs("./model", exist_ok=True)
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

        scaler.scale(loss).backward()

        # 每 accum_steps 个 小batch 更新一次参数
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        # # 反向传播和优化
        # optimizer.zero_grad()  # 清空梯度
        # loss.backward()  # 反向传播计算梯度
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        # optimizer.step()  # 更新参数

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
                torch.save(model.state_dict(), f"./model/best_iou_model.pth")
                best_iou = iou
                print(f"\nBest Test Model Saved at Epoch {step + 1}")

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), f"./model/best_test_model.pth")
                best_val_loss = val_loss
                print(f"\nBest Test Loss Model Saved at Epoch {step + 1}")

        # 每500步保存一次模型
        if (step + 1) % 500 == 0:
            torch.save(model.state_dict(), f"./model/model_step_{step + 1}.pth")
            print(f"\nModel Saved at Step {step + 1}")

    writer.close()


def train_step(model, train_loader, criterion, optimizer, scaler, epoch, num_epochs, device, accum_steps=2):
    model.train()
    train_losses = 0.0
    optimizer.zero_grad(set_to_none=True)  # 只在外层先清空一次梯度

    train_loader_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", leave=True)
    for step, batch in enumerate(train_loader_tqdm):
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

        scaler.scale(loss).backward()

        # 每 accum_steps 个 小batch 更新一次参数
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # # 反向传播和优化
        # optimizer.zero_grad()  # 清空梯度
        # loss.backward()  # 反向传播计算梯度
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        # optimizer.step()  # 更新参数

        # 累计损失
        train_losses += loss.item() * accum_steps  # 还原回正常的损失值
        # 显示当前轮次的累计损失
        train_loader_tqdm.set_postfix(loss=train_losses / (step + 1))

    # # 如果最后一个小batch没凑够accum_steps，也要更新一次参数
    # if (step+1) % accum_steps != 0:
    #     scaler.unscale_(optimizer)
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #     scaler.step(optimizer)
    #     scaler.update()
    #     optimizer.zero_grad(set_to_none=True)

    return train_losses / len(train_loader)


def test_step(model, test_loader, criterion, epoch, num_epochs, device):
    model.eval()
    test_losses = 0.0
    total_iou = 0.0
    total_images = 0

    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc=f"Test Epoch {epoch + 1}/{num_epochs}", leave=True)
        for step, batch in enumerate(test_loader_tqdm):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            # masks = (masks > 0.5).float()

            preds = model(images)
            loss = criterion(preds, masks)
            test_losses += loss.item()

            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()

            # IoU = |A∩B| / |A∪B|
            #     = |A∩B| / (|A|+|B|−|A∩B|)
            # ∣A∩B∣
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            # |A|+|B|
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            iou_per_image = (intersection + 1e-6) / (union - intersection + 1e-6)

            total_iou += iou_per_image.sum().item()
            total_images += images.size(0)

            test_loader_tqdm.set_postfix(loss=test_losses / (step + 1),
                                         iou=total_iou / total_images)

    return total_iou / total_images, test_losses / len(test_loader)


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, scaler, device,
                num_epochs=50, total_steps=2000, accum_steps=4):
    os.makedirs("./model", exist_ok=True)
    writer = SummaryWriter(log_dir=f"./logs/Seg_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    best_train_loss = float('inf')
    best_iou = 0.0
    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, criterion, optimizer, scaler, epoch, num_epochs, device,
                                accum_steps)
        iou, test_loss = test_step(model, test_loader, criterion, epoch, num_epochs, device)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('IOU/Test', iou, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step()

        if iou > best_iou:
            torch.save(model.state_dict(), f"./model/best_iou_model.pth")
            best_iou = iou
            print(f"\nBest Test Model Saved at Epoch {epoch + 1}")

        if test_loss < best_test_loss:
            torch.save(model.state_dict(), f"./model/best_test_model.pth")
            best_test_loss = test_loss
            print(f"\nBest Test Loss Model Saved at Epoch {epoch + 1}")


if __name__ == "__main__":
    seeds(125)

    # 检查当前系统是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_image_dir = "../../MuralDH/MuralDH/Mural_seg/train/images"
    train_mask_dir = "../../MuralDH/MuralDH/Mural_seg/train/labels"
    val_image_dir = "../../MuralDH/MuralDH/Mural_seg/test/images"
    val_mask_dir = "../../MuralDH/MuralDH/Mural_seg/test/labels"

    train_dataset = SegMuralDataset(train_image_dir, train_mask_dir, size=512)
    train_dataloader = DataLoaderX(
        train_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataset = SegMuralDataset(val_image_dir, val_mask_dir, size=512)
    val_dataloader = DataLoaderX(
        val_dataset,
        batch_size=4,
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
    num_epochs = 50
    accum_steps = 2
    total_steps = 2000

    scheduler_steps = int(total_steps // accum_steps)
    warmup_steps = int(0.1 * scheduler_steps)

    valid_steps = 25

    # sam_model_path = r'./model/sam2.1_hiera_large.pt'
    # sam_model_cfg_path = r'./model/sam2.1_hiera_l.yaml'
    # model = SAMAdapterModel(model_cfg=sam_model_cfg_path, checkpoint=sam_model_path)
    model_type = "vit_b"  # 可选 "vit_b" 或 "vit_h"
    sam_model_path = r'./model/sam_vit_b_01ec64.pth'
    # sam_model_path = r'./model/sam_vit_h_4b8939.pth'

    model = SAMAdapterModel(model_type=model_type, checkpoint=sam_model_path)
    print(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, weight_decay=1e-5)
    # criterion = BCEDiceLoss()
    criterion = FocalTversky_BCE_Loss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, scheduler_steps)
    scaler = torch.amp.GradScaler()  # 混合精度

    # train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, scaler, device,
    #             num_epochs=num_epochs, total_steps=total_steps, accum_steps=1)
    train_and_val(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, scaler, device,
                  total_steps=total_steps, valid_steps=valid_steps, accum_steps=accum_steps)
