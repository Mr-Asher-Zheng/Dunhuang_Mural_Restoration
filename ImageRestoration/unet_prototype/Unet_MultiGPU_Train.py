from Unet_model import *
from Utils import *
from ImageRestoration.vqvae_prototype.VQVAE import VQVAE
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
from pytorch_msssim import ms_ssim


def psnr(img1, img2, max_value=1.0, eps=1e-5):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(max_value / torch.sqrt(mse + eps))


def predict_x0(x_t, t, noise_pred, noise_scheduler):
    sqrt_alpha_bar = noise_scheduler.get(noise_scheduler.sqrt_alpha_bar, t, x_t.shape)
    sqrt_one_minus_alpha_bar = noise_scheduler.get(noise_scheduler.sqrt_one_minus_alpha_bar, t, x_t.shape)

    x_0 = (x_t - sqrt_one_minus_alpha_bar * noise_pred) / sqrt_alpha_bar
    return x_0


def train_and_val(train_loader, val_loader, ddpm_loader,
                  model, vae,
                  criterion,
                  optimizer,
                  scheduler, noise_scheduler,
                  config,
                  total_steps, valid_steps, ddpm_steps,
                  device, accelerator):
    os.makedirs("./model", exist_ok=True)
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=f"./logs/Unet_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    best_val_PSNR = 0.0
    best_val_loss = float('inf')

    train_iterator = iter(train_loader)
    train_loader_tqdm = tqdm(range(total_steps), desc="Training", leave=False)

    for step in train_loader_tqdm:
        model.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        images = batch["image"].to(device)
        lines = batch["line"].to(device)
        masks = batch["mask"].to(device)
        anti_masks = batch["anti_mask"].to(device)

        # print("images, lines, masks, anti_masks", images.shape, lines.shape, masks.shape, anti_masks.shape)

        with torch.no_grad():
            latent_images, _, _ = vae.encode(images)
        # print(images.shape)

        t = torch.randint(0, noise_scheduler.num_steps, (latent_images.shape[0],), device=device)
        # print(t.shape)

        noises = torch.randn_like(latent_images).to(device)
        noisy_latent_images, noise = noise_scheduler.add_noise(latent_images, noises, t)
        # print("noisy_images, noise", noisy_images.shape, noise.shape)

        latent_noise_pred = model(noisy_latent_images, t, cond_input=lines)
        # print("noise_pred", noise_pred.shape)
        # time.sleep(60)

        loss = criterion(latent_noise_pred, noise)

        optimizer.zero_grad()
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        train_loader_tqdm.set_postfix(loss=loss.item())

        if (step + 1) % 25 == 0:
            global_train_loss = accelerator.reduce(loss, reduction='mean')
            if accelerator.is_main_process:
                writer.add_scalar("Loss/train", global_train_loss, step + 1)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step + 1)

        if (step + 1) % valid_steps == 0:
            model.eval()

            val_losses = 0.0
            val_PSNRs = 0.0
            val_ms_ssims = 0.0

            val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)

            with torch.no_grad():
                for i, batch in enumerate(val_loader_tqdm):
                    images = batch["image"].to(device)
                    lines = batch["line"].to(device)
                    masks = batch["mask"].to(device)
                    anti_masks = batch["anti_mask"].to(device)

                    # 编码到潜空间
                    latent_images, _, _ = vae.encode(images)
                    masks = torch.nn.functional.interpolate(masks, size=latent_images.shape[2:], mode='nearest')
                    anti_masks = torch.nn.functional.interpolate(anti_masks, size=latent_images.shape[2:],
                                                                 mode='nearest')
                    # print("images, masks, anti_masks", images.shape, masks.shape, anti_masks.shape)

                    # 获取随机时间步t和噪声
                    t = torch.randint(0, noise_scheduler.num_steps, (latent_images.shape[0],), device=device)
                    noises = torch.randn_like(latent_images).to(device)

                    # 潜空间图像添加随机时间步t噪声得到带噪声的潜空间图像
                    noisy_latent_images, noise = noise_scheduler.add_noise(latent_images, noises, t)

                    # 预测噪声
                    latent_noise_pred = model(noisy_latent_images, t, cond_input=lines)
                    # 根据带噪声的潜空间图像、时间步t和预测噪声计算x_0（也就是去噪后的潜空间图像）
                    latent_z0_pred = predict_x0(noisy_latent_images, t, latent_noise_pred, noise_scheduler)

                    # print("x_t final", x_t.shape)
                    x_t_decode = vae.decode(latent_z0_pred)
                    # print("x_t_decode", x_t_decode.shape)

                    # time.sleep(60)

                    # 将第一个batch的结果记录到tensorboard
                    if i == 0:
                        for index in range(x_t_decode.shape[0]):
                            img = x_t_decode[index].detach().cpu()
                            img = img * 0.5 + 0.5
                            img = torch.clip(img, 0, 1)
                            img = (img * 255).to(torch.uint8)
                            if accelerator.is_main_process:
                                writer.add_image(f'val_step_{index}', img, step + 1, dataformats='CHW')

                    # x_t_decode = x_t_decode * anti_masks
                    # images = images * anti_masks
                    val_PSNR = psnr(x_t_decode, images)
                    val_ms_ssim = ms_ssim(x_t_decode, images, data_range=2.0, size_average=True)
                    val_loss = criterion(x_t_decode, images)

                    val_PSNRs += val_PSNR.item()
                    val_ms_ssims += val_ms_ssim.item()
                    val_losses += val_loss.item()

                val_loader_tqdm.set_postfix(val_PSNR=val_PSNRs / (i + 1))
                val_loader_tqdm.set_postfix(val_ms_ssim=val_ms_ssims / (i + 1))
                val_loader_tqdm.set_postfix(val_loss=val_losses / (i + 1))

                avg_val_PSNR = torch.tensor(val_PSNRs / len(val_loader), device=device)
                avg_val_ms_ssim = torch.tensor(val_ms_ssims / len(val_loader), device=device)
                avg_val_loss = torch.tensor(val_losses / len(val_loader), device=device)

                global_val_PSNR = accelerator.reduce(avg_val_PSNR, reduction='mean')
                global_val_ms_ssim = accelerator.reduce(avg_val_ms_ssim, reduction='mean')
                global_val_loss = accelerator.reduce(avg_val_loss, reduction='mean')

                if accelerator.is_main_process:
                    writer.add_scalar("PSNR/val", global_val_PSNR, step + 1)
                    writer.add_scalar("MS_SSIM/val", global_val_ms_ssim, step + 1)
                    writer.add_scalar("MSE/val", global_val_loss, step + 1)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if global_val_PSNR > best_val_PSNR:
                    best_val_PSNR = global_val_PSNR

                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.save(unwrapped_model.state_dict(), f"./model/best_val_PSNR.pth")
                    print(f"\nBest Val PSNR Model Saved at Step {step + 1}, PSNR: {best_val_PSNR:.4f}")

        if (step + 1) % ddpm_steps == 0:
            model.eval()
            ddpm_loader_tqdm = tqdm(ddpm_loader, desc="ddpm", leave=False)

            with torch.no_grad():
                for i, batch in enumerate(ddpm_loader_tqdm):
                    images = batch["image"].to(device)
                    lines = batch["line"].to(device)
                    masks = batch["mask"].to(device)
                    anti_masks = batch["anti_mask"].to(device)

                    # 编码到潜空间
                    latent_images, _, _ = vae.encode(images)
                    masks = torch.nn.functional.interpolate(masks, size=latent_images.shape[2:], mode='nearest')
                    anti_masks = torch.nn.functional.interpolate(anti_masks, size=latent_images.shape[2:],
                                                                 mode='nearest')

                    # 逐步去噪，同时替换完好区域
                    # 从标准正态分布中采样初始噪声 x_T ~ N(0,I)
                    # [b, c, size, size]
                    x_t = torch.randn_like(latent_images).to(device)
                    # print("x_t", x_t.shape)

                    # 从标准正态分布中采样噪声 ε~N(0,I)
                    fixed_noises = x_t.clone()

                    # 逐步去噪，从 t=T(999) 到 t=0
                    for t in reversed(range(noise_scheduler.num_steps)):
                        t_batch = torch.tensor([t] * latent_images.shape[0]).to(device)

                        gt_noisy, _ = noise_scheduler.add_noise(latent_images, fixed_noises, t_batch)
                        # print("gt_noisy", gt_noisy.shape)

                        x_t = x_t * masks + gt_noisy * anti_masks
                        # print("x_t after anti_masks", x_t.shape)

                        # 获取采样需要的系数
                        sqrt_recip_alpha_bar = noise_scheduler.get(noise_scheduler.sqrt_recip_alphas_bar, t_batch,
                                                                   x_t.shape)
                        sqrt_recipm1_alpha_bar = noise_scheduler.get(noise_scheduler.sqrt_recipm1_alphas_bar, t_batch,
                                                                     x_t.shape)

                        posterior_mean_coef1 = noise_scheduler.get(noise_scheduler.posterior_mean_coef1, t_batch,
                                                                   x_t.shape)
                        posterior_mean_coef2 = noise_scheduler.get(noise_scheduler.posterior_mean_coef2, t_batch,
                                                                   x_t.shape)

                        # 预测噪声 ε_θ(x_t, t)
                        predicted_noise = model(x_t, t_batch, cond_input=lines)
                        # print("predicted_noise", predicted_noise.shape)

                        # time.sleep(60)

                        # 计算x_0的预测值： x_0 = 1/sqrt(α_bar_t) * x_t - sqrt(1/α_bar_t - 1) * ε_θ(x_t, t)
                        _x_0 = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * predicted_noise
                        # 计算后验分布均值 μ_θ(x_t, t)
                        model_mean = posterior_mean_coef1 * _x_0 + posterior_mean_coef2 * x_t
                        # 计算后验分布方差的对数值 log(σ_t^2)
                        model_log_var = noise_scheduler.get(
                            torch.log(torch.cat([noise_scheduler.posterior_var[1:2], noise_scheduler.betas[1:]])),
                            t_batch, x_t.shape)

                        if t > 0:
                            # t>0 时，从后验分布中采样：x_t-1 = μ_θ(x_t, t) + σ_t * z, z~N(0,I)
                            noise = torch.randn_like(x_t).to(device)
                            x_t = model_mean + torch.exp(0.5 * model_log_var) * noise
                        else:
                            # t=0 时，直接使用均值作为生成结果
                            x_t = model_mean

                    x_t_decode = vae.decode(x_t)

                    for index in range(x_t_decode.shape[0]):
                        img = x_t_decode[index].detach().cpu()
                        img = img * 0.5 + 0.5
                        img = torch.clip(img, 0, 1)
                        img = (img * 255).to(torch.uint8)
                        if accelerator.is_main_process:
                            writer.add_image(f'ddpm_step_{index}', img, step + 1, dataformats='CHW')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), f"./model/model_last.pth")
        print("Done Training.")


if __name__ == '__main__':
    seeds(42)

    print(torch.cuda.device_count())
    accelerator = Accelerator(
        mixed_precision="fp16",
    )

    print("Mixed precision mode:", accelerator.mixed_precision)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = accelerator.device
    print(f"Using device: {device}")
    print(f"Total GPUs detected: {accelerator.num_processes}")

    batch_size = 1

    # # 检查当前系统是否有可用的 GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f"Training on {device}")

    train_image_dir = "../../../DH/image_mask/train/images"
    train_line_dir = "../../../DH/image_mask/train/lines"
    train_mask_dir = "../../../DH/image_mask/train/masks"

    val_image_dir = "../../../DH/image_mask/val/images"
    val_line_dir = "../../../DH/image_mask/val/lines"
    val_mask_dir = "../../../DH/image_mask/val/masks"

    train_dataset = RestorationMuralDataset(train_image_dir, train_line_dir, train_mask_dir, size=512)
    train_dataloader = DataLoaderX(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataset = RestorationMuralDataset(val_image_dir, val_line_dir, val_mask_dir, size=512)
    val_dataloader = DataLoaderX(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # 从val_dataloader里随机挑选10张图片作为ddpm_dataloader
    indices = torch.randperm(len(val_dataset))[:batch_size]
    ddpm_dataset = Subset(val_dataset, indices)
    ddpm_dataloader = DataLoaderX(
        ddpm_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print("train:", len(train_dataloader))
    print("val:", len(val_dataloader))
    print("ddpm:", len(ddpm_dataloader))

    # for _ in range(50):
    #     image_sample(train_dataloader)
    #
    # time.sleep(60)

    # iter(train_dataloader)
    # print("DataLoader ready.")
    # time.sleep(60)

    z_channels = 4
    total_steps = 15000
    warmup_steps = 1000
    valid_steps = 500
    ddpm_steps = 1000
    learning_rate = 5e-5
    learning_rate = learning_rate * batch_size * accelerator.num_processes

    model = Unet(z_channels=z_channels, line_cond=True, image_cross_attn=False).to(device)

    vae = VQVAE().to(device)
    vae.eval()
    vae.load_state_dict(torch.load("../vqvae_prototype/vqvae_pred_new_3/vqvae_autoencoder_best.pth", weights_only=True))
    for param in vae.parameters():
        param.requires_grad = False

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    model, optimizer, train_dataloader, val_dataloader, ddpm_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, ddpm_dataloader
    )

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    noise_scheduler = NoiseScheduler(num_steps=1000).to(device)

    config = {
        "mse_weight": 1.0,
    }

    train_and_val(train_dataloader, val_dataloader, ddpm_dataloader,
                  model, vae,
                  criterion,
                  optimizer,
                  scheduler, noise_scheduler,
                  config,
                  total_steps, valid_steps, ddpm_steps,
                  device, accelerator)
