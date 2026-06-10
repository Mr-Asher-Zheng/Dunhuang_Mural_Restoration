import time

from Diffusion_model import *
from Utils import *


def train_step(model, train_loader, noise_scheduler, criterion, optimizer, epoch, num_epochs, device):
    model.train()
    train_losses = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", leave=True)
    for batch in train_loader_tqdm:
        # 获取一个batch的图像数据并移至指定设备
        # [16, 3, 256, 256]
        images = batch["image"].to(device)
        mask_label = batch["mask"].to(device)

        # 随机采样时间步t
        # [16]
        t = torch.randint(0, noise_scheduler.num_steps, (images.shape[0],), device=device)

        # 对图像添加噪声，获得带噪声的图像和噪声
        # t进入后会被扩展成[16, 1, 1, 1]，以便进行广播
        # noisy_images：[16, 3, 256, 256]
        # noise: [16, 3, 256, 256]
        noisy_images, noise = noise_scheduler.add_noise(images, t)

        # 使用模型预测噪声
        predicted_noise = model(noisy_images, t)

        # 计算预测噪声与真实噪声之间的损失
        loss = criterion(predicted_noise, noise)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()  # 更新参数

        # 累计损失
        train_losses += loss.item()
        # 显示当前轮次的累计损失
        train_loader_tqdm.set_postfix(loss=train_losses)

    return train_losses


def test_step(model, test_loader, noise_scheduler, criterion, epoch, num_epochs, device):
    model.eval()
    test_losses = 0.0
    with torch.no_grad():
        test_loader_tqdm = tqdm(test_loader, desc=f"Test Epoch {epoch + 1}/{num_epochs}", leave=True)
        for batch in test_loader_tqdm:
            images = batch["image"].to(device)
            # 创建形状为images.shape[0]（batch），所有元素值为noise_scheduler.num_steps - 1（最大噪声步）的张量
            t = torch.full((images.shape[0],), noise_scheduler.num_steps - 1, device=device)
            noisy_images, noise = noise_scheduler.add_noise(images, t)

            predicted_noise = model(noisy_images, t)
            loss = criterion(predicted_noise, noise)
            test_losses += loss.item()
            test_loader_tqdm.set_postfix(loss=test_losses)

    return test_losses


def train_model(model, train_loader, test_loader, noise_scheduler, criterion, optimizer, scheduler, device,
                num_epochs=100):
    os.makedirs("./model", exist_ok=True)
    writer = SummaryWriter(log_dir=f"./logs/Diffusion_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    best_train_loss = float('inf')
    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_step(model, train_loader, noise_scheduler, criterion, optimizer, epoch, num_epochs, device)
        test_loss = test_step(model, test_loader, noise_scheduler, criterion, epoch, num_epochs, device)

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f"../model/best_test_model.pth")
            print(f"\nBest Test Model Saved at Epoch {epoch + 1}")


if __name__ == '__main__':
    seeds(125)

    # 检查当前系统是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_image_dir = "../../../DH/image_mask/train/images"
    train_line_dir = "../../../DH/image_mask/train/lines"
    train_mask_dir = "../../../DH/image_mask/train/masks"

    val_image_dir = "../../../DH/image_mask/val/images"
    val_line_dir = "../../../DH/image_mask/val/lines"
    val_mask_dir = "../../../DH/image_mask/val/masks"

    train_dataset = RestorationMuralDataset(train_image_dir, train_line_dir, train_mask_dir, size=128)  # 128
    train_dataloader = DataLoaderX(
        train_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    test_dataset = RestorationMuralDataset(val_image_dir, val_line_dir, val_mask_dir, size=128)
    test_dataloader = DataLoaderX(
        test_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # ===================================== 查看数据集示例 =====================================
    # 查看数据集示例
    image_sample(train_dataloader)
    # time.sleep(60)

    # ===================================== 查看某一图像在某一时间步的加噪效果 =====================================
    # # 查看加噪过程示例
    # image = next(iter(train_dataloader))
    # # 取出一个batch的图像数据
    # image = image["image"]
    # # 创建噪声调度器
    # noise_scheduler = NoiseScheduler(num_steps=1000)
    # # 输入一个batch的图像和[0,1000)范围内的随机时间步，得到随机时间步的加噪图像
    # noise_image, noise = noise_scheduler.add_noise(
    #     image,
    #     torch.randint(0, noise_scheduler.num_steps, (image.shape[0],))
    # )
    # # 显示某一时间步的加噪图像
    # print(noise_image.shape)
    # plt.imshow(noise_image[0].permute(1, 2, 0).cpu().numpy())
    # plt.axis('off')
    # plt.show()

    # ==================================== 查看某一图像逐步加噪的过程 =====================================
    # 绘制某一图像逐步加噪的过程
    image = next(iter(train_dataloader))
    image = image["image"]
    noise_scheduler = NoiseScheduler(num_steps=1000)
    plot_diffusion_steps(image[0].permute(1, 2, 0), noise_scheduler, step_size=100)

    # ===================================== 训练 =====================================
    noise_scheduler = NoiseScheduler(num_steps=1000).to(device)
    model = Unet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=128).to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    criterion = nn.MSELoss()
    train_model(model, train_dataloader, test_dataloader, noise_scheduler, criterion, optimizer, scheduler, device,
                num_epochs=100)
