from ImageRestoration.unet_prototype.Diffusion_model import *
from Utils import *


# def sample(model, scheduler, num_samples, size, device):
#     """从噪声采样生成图像的函数
#     Args:
#         model: Unet模型，用于预测噪声
#         scheduler: 噪声调度器，包含采样所需的所有系数
#         num_samples: 要生成的样本数量
#         size: 生成图像的大小，如(3, 32, 32)
#         device: 运行设备
#     Returns:
#         生成的图像张量
#     """
#     model.eval()
#     with torch.no_grad():
#         # 从标准正态分布中采样初始噪声 x_T ~ N(0,I)
#         # [10, 3, 32, 32]
#         x_t = torch.randn(num_samples, *size).to(device)
#         print(x_t.shape)
#
#         # 逐步去噪，从 t=T(999) 到 t=0
#         for t in tqdm(reversed(range(scheduler.num_steps)), desc="Sampling"):
#             # 构造时间步batch
#             # [10]代表每次10个样本都在同一个时间步t
#             t_batch = torch.tensor([t] * num_samples).to(device)
#
#             # 获取采样需要的系数
#             sqrt_recip_alpha_bar = scheduler.get(scheduler.sqrt_recip_alphas_bar, t_batch, x_t.shape)
#             sqrt_recipm1_alpha_bar = scheduler.get(scheduler.sqrt_recipm1_alphas_bar, t_batch, x_t.shape)
#
#             posterior_mean_coef1 = scheduler.get(scheduler.posterior_mean_coef1, t_batch, x_t.shape)
#             posterior_mean_coef2 = scheduler.get(scheduler.posterior_mean_coef2, t_batch, x_t.shape)
#
#             # 预测噪声 ε_θ(x_t, t)
#             predicted_noise = model(x_t, t_batch)
#
#             # 计算x_0的预测值： x_0 = 1/sqrt(α_bar_t) * x_t - sqrt(1/α_bar_t - 1) * ε_θ(x_t, t)
#             _x_0 = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * predicted_noise
#             # 计算后验分布均值 μ_θ(x_t, t)
#             model_mean = posterior_mean_coef1 * _x_0 + posterior_mean_coef2 * x_t
#             # 计算后验分布方差的对数值 log(σ_t^2)
#             model_log_var = scheduler.get(torch.log(torch.cat([scheduler.posterior_var[1:2], scheduler.betas[1:]])), t_batch, x_t.shape)
#
#             if t > 0:
#                 # t>0 时，从后验分布中采样：x_t-1 = μ_θ(x_t, t) + σ_t * z, z~N(0,I)
#                 noise = torch.randn_like(x_t).to(device)
#                 x_t = model_mean + torch.exp(0.5 * model_log_var) * noise
#             else:
#                 # t=0 时，直接使用均值作为生成结果
#                 x_t = model_mean
#         # 将最终结果裁剪到[-1, 1]范围
#         x_0 = torch.clamp(x_t, -1.0, 1.0)
#     return x_0

def inpaint(model, scheduler, images, masks, device):
    """从噪声采样生成图像的函数
    Args:
        model: Unet模型，用于预测噪声
        scheduler: 噪声调度器，包含采样所需的所有系数
        images: 输入图像
        masks: 输入掩码
        device: 运行设备
    Returns:
        生成的图像张量
    """
    model.eval()
    with torch.no_grad():
        # 从标准正态分布中采样初始噪声 x_T ~ N(0,I)
        # [10, 3, size, size]
        x_t = torch.randn_like(images).to(device)
        print(x_t.shape)

        # 逐步去噪，从 t=T(999) 到 t=0
        for t in tqdm(reversed(range(scheduler.num_steps)), desc="Sampling"):
            # 构造时间步batch
            # [10]代表每次10个样本都在同一个时间步t
            t_batch = torch.tensor([t] * images.shape[0]).to(device)

            # 获取采样需要的系数
            sqrt_recip_alpha_bar = scheduler.get(scheduler.sqrt_recip_alphas_bar, t_batch, x_t.shape)
            sqrt_recipm1_alpha_bar = scheduler.get(scheduler.sqrt_recipm1_alphas_bar, t_batch, x_t.shape)

            posterior_mean_coef1 = scheduler.get(scheduler.posterior_mean_coef1, t_batch, x_t.shape)
            posterior_mean_coef2 = scheduler.get(scheduler.posterior_mean_coef2, t_batch, x_t.shape)

            # 预测噪声 ε_θ(x_t, t)
            predicted_noise = model(x_t, t_batch)

            # 计算x_0的预测值： x_0 = 1/sqrt(α_bar_t) * x_t - sqrt(1/α_bar_t - 1) * ε_θ(x_t, t)
            _x_0 = sqrt_recip_alpha_bar * x_t - sqrt_recipm1_alpha_bar * predicted_noise
            # 计算后验分布均值 μ_θ(x_t, t)
            model_mean = posterior_mean_coef1 * _x_0 + posterior_mean_coef2 * x_t
            # 计算后验分布方差的对数值 log(σ_t^2)
            model_log_var = scheduler.get(torch.log(torch.cat([scheduler.posterior_var[1:2], scheduler.betas[1:]])),
                                          t_batch, x_t.shape)

            if t > 0:
                # t>0 时，从后验分布中采样：x_t-1 = μ_θ(x_t, t) + σ_t * z, z~N(0,I)
                noise = torch.randn_like(x_t).to(device)
                x_t = model_mean + torch.exp(0.5 * model_log_var) * noise
            else:
                # t=0 时，直接使用均值作为生成结果
                x_t = model_mean

        # 只在掩码区域替换生成的像素，保留未掩码区域的原始像素（这样做会导致修复边缘和原图边缘割裂）
        x_t = x_t * masks + images * (1 - masks)

        x_0 = torch.clamp(x_t, -1.0, 1.0)
    return x_0


def plot(images):
    fig = plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.imshow(torchvision.utils.make_grid(images, nrow=5).permute(1, 2, 0))
    plt.tight_layout(pad=1)
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_image_dir = "../../../DH/image_mask/val/images"
    val_mask_dir = "../../../DH/image_mask/val/masks"
    val_line_dir = "../../../DH/image_mask/val/lines"

    val_dataset = RestorationMuralDataset(val_image_dir, val_line_dir, val_mask_dir, size=128)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=10,
    )
    image_batch = next(iter(val_dataloader))

    image_size = 128
    model = Unet(in_channels=3, out_channels=3, base_channels=64, time_emb_dim=128)
    model.load_state_dict(torch.load("../model/best_test_model.pth", weights_only=True))
    model = model.to(device)

    scheduler = NoiseScheduler(num_steps=1000).to(device)

    # images = sample(model, scheduler, 10, (3, image_size, image_size), device)
    images = inpaint(model, scheduler, image_batch["image"].to(device), image_batch["mask"].to(device), device)

    images = images.detach().cpu().clamp(0, 1)
    plot(images)
