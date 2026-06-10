import argparse
import os
import sys
import time
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler, \
    DDIMScheduler
from torch.utils.data import Subset
from accelerate import Accelerator
from pytorch_msssim import ms_ssim, SSIM

from utils import *


def plot(original_images_list, pred_images_list, input_images_list, lines_list, masks_list,
         gt_save_dir, pred_save_dir, input_save_dir, line_save_dir, mask_save_dir,
         names_list, save=False):
    original_images = torch.cat(original_images_list, dim=0)
    pred_images = torch.cat(pred_images_list, dim=0)
    input_images = torch.cat(input_images_list, dim=0)
    lines = torch.cat(lines_list, dim=0)
    masks = torch.cat(masks_list, dim=0)
    names_list = [item for sublist in names_list for item in sublist]
    names_list = [os.path.splitext(name)[0] for name in names_list]
    print(names_list)
    # ['DhMurals-inpainting-dataset_00125.png', 'Dunhuang_Faces_00303.png', 'Dunhuang_Grottoes_Painting_00025.png', 'MuralDH_00187.png', 'MuralDH_04297.png', 'MuralDH_05398.png']

    print("original_images.shape:", original_images.shape)
    print("pred_images.shape:", pred_images.shape)
    print("input_images.shape:", input_images.shape)
    print("lines.shape:", lines.shape)
    print("masks.shape:", masks.shape)
    # original_images.shape: torch.Size([4, 3, 512, 512])
    # pred_images.shape: torch.Size([4, 3, 512, 512])
    # input_images.shape: torch.Size([4, 3, 512, 512])
    # lines.shape: torch.Size([4, 3, 512, 512])
    # masks.shape: torch.Size([4, 1, 512, 512])

    original_images_np = original_images.permute(0, 2, 3, 1).numpy()
    pred_images_np = pred_images.permute(0, 2, 3, 1).numpy()
    input_images_np = input_images.permute(0, 2, 3, 1).numpy()
    lines_np = lines.permute(0, 2, 3, 1).numpy()
    masks_np = masks.permute(0, 2, 3, 1).numpy()

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    original_images_np = (original_images_np * std + mean) * 255
    pred_images_np = (pred_images_np * std + mean) * 255
    input_images_np = (input_images_np * std + mean) * 255
    lines_np = (lines_np * std + mean) * 255

    if save:

        for i in range(original_images_np.shape[0]):
            name = names_list[i]

            original = original_images_np[i].clip(0, 255).astype(np.uint8)
            pred = pred_images_np[i].clip(0, 255).astype(np.uint8)
            inp = input_images_np[i].clip(0, 255).astype(np.uint8)
            line = lines_np[i].clip(0, 255).astype(np.uint8)

            mask = masks_np[i][:, :, 0] * 255
            mask = mask.astype(np.uint8)

            Image.fromarray(original).save(os.path.join(gt_save_dir, f"{name}_gt.png"))
            Image.fromarray(pred).save(os.path.join(pred_save_dir, f"{name}_pred.png"))
            Image.fromarray(inp).save(os.path.join(input_save_dir, f"{name}_input.png"))
            Image.fromarray(line).save(os.path.join(line_save_dir, f"{name}_line.png"))
            Image.fromarray(mask).save(os.path.join(mask_save_dir, f"{name}_mask.png"))

    titles = ["Original Image", "Predicted Image", "Input Images", "Line Drawing", "Mask"]
    inputs = [original_images_np, pred_images_np, input_images_np, lines_np, masks_np]

    fig, axs = plt.subplots(pred_images.shape[0], 5, figsize=(16, 8))

    print(pred_images.shape[0])
    if pred_images.shape[0] == 1:
        # axs = axs.unsqueeze(0)
        axs = np.expand_dims(axs, axis=0)

    for i in range(pred_images.shape[0]):
        for j, (title, input) in enumerate(zip(titles, inputs)):
            ax = axs[i, j]
            img = input[i]
            img = img.astype("uint8")
            ax.imshow(img, cmap="gray" if input.shape[3] == 1 else None)
            ax.set_title(title if i == 0 else "")
            ax.axis("off")
    plt.tight_layout()
    plt.show()


def test1(val_loader, ddpm_dataloader, encoder_hidden_states_full,
          controlnet, vae, unet, tokenizer, text_encoder,
          criterion,
          noise_scheduler,
          weight_dtype,
          device, accelerator):
    controlnet.eval()

    val_losses = 0.0
    val_PSNRs = 0.0
    val_ms_ssims = 0.0

    val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)

    for i, batch in enumerate(val_loader_tqdm):
        with torch.no_grad():
            images = batch["image"].to(device, dtype=weight_dtype)
            lines = batch["line"].to(device, dtype=weight_dtype)
            masks = batch["mask"].to(device)
            anti_masks = batch["anti_mask"].to(device)
            encoder_hidden_states = encoder_hidden_states_full[:images.shape[0]]

            latent_images = vae.encode(images).latent_dist.sample()
            latent_images = latent_images * vae.config.scaling_factor

            masks = torch.nn.functional.interpolate(masks, size=latent_images.shape[2:], mode='nearest')
            anti_masks = torch.nn.functional.interpolate(anti_masks, size=latent_images.shape[2:],
                                                         mode='nearest')
            # print(noise_scheduler.config.num_train_timesteps)
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latent_images.shape[0],),
                              device=device).long()
            # print(t)

            noises = torch.randn_like(latent_images).to(device)
            noisy_latent_images = noise_scheduler.add_noise(latent_images, noises, t)

            controlnet_images = lines

            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latent_images,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_images,
                return_dict=False,
            )

            model_pred = unet(
                noisy_latent_images,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                return_dict=False,
            )[0]

            # 根据带噪声的潜空间图像、时间步t和预测噪声计算x_0（也就是去噪后的潜空间图像）
            latent_z0_pred = predict_x0(noisy_latent_images, t, model_pred, noise_scheduler)
            # print("latent_z0_pred:", latent_z0_pred.dtype)
            latent_z0_pred = latent_z0_pred.to(dtype=weight_dtype)
            # print("latent_z0_pred:", latent_z0_pred.dtype)

            x_t_decode = vae.decode(latent_z0_pred / vae.config.scaling_factor).sample

        # # 将第一个batch的结果记录到plt
        # if i == 0:
        #     imgs = []
        #     for index in range(x_t_decode.shape[0]):
        #         img = x_t_decode[index].detach().cpu()
        #         img = img * 0.5 + 0.5
        #         img = torch.clip(img, 0, 1)
        #         img = (img * 255).to(torch.uint8)
        #         img = img.permute(1, 2, 0).numpy()
        #         imgs.append(img)
        #     fig, axes = plt.subplots(1, len(imgs), figsize=(16, 4))
        #     for ax, img in zip(axes, imgs):
        #         ax.imshow(img)
        #         ax.axis("off")
        #     plt.tight_layout()
        #     plt.show()

        # x_t_decode = x_t_decode * anti_masks
        # images = images * anti_masks
        val_PSNR = psnr(x_t_decode, images)
        val_ms_ssim = ms_ssim(x_t_decode, images, data_range=2.0, size_average=True)
        val_loss = criterion(x_t_decode, images)

        val_PSNRs += val_PSNR.item()
        val_ms_ssims += val_ms_ssim.item()
        val_losses += val_loss.item()

        val_loader_tqdm.set_postfix({
            "val_PSNR": val_PSNRs / (i + 1),
            "val_ms_ssim": val_ms_ssims / (i + 1),
            "val_loss": val_losses / (i + 1),
        })

    avg_val_PSNR = torch.tensor(val_PSNRs / len(val_loader), device=device)
    avg_val_ms_ssim = torch.tensor(val_ms_ssims / len(val_loader), device=device)
    avg_val_loss = torch.tensor(val_losses / len(val_loader), device=device)

    global_val_PSNR = accelerator.reduce(avg_val_PSNR, reduction='mean')
    global_val_ms_ssim = accelerator.reduce(avg_val_ms_ssim, reduction='mean')
    global_val_loss = accelerator.reduce(avg_val_loss, reduction='mean')

    if accelerator.is_main_process:
        print("PSNR/val", global_val_PSNR)
        print("MS_SSIM/val", global_val_ms_ssim)
        print("MSE/val", global_val_loss)


def pred(ddpm_loader, encoder_hidden_states_full,
         controlnet, vae, unet, tokenizer, text_encoder,
         criterion,
         noise_scheduler,
         weight_dtype,
         device, accelerator):
    controlnet.eval()

    ddpm_PSNRs = 0.0
    ddpm_ms_ssims = 0.0
    ddpm_losses = 0.0

    original_images_list, pred_images_list, input_images_list, lines_list, masks_list = [], [], [], [], []
    names_list = []
    ddpm_loader_tqdm = tqdm(ddpm_loader, desc="ddpm", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(ddpm_loader_tqdm):
            original_images = batch["image"].to(device, dtype=weight_dtype)
            images = batch["image_mask"].to(device, dtype=weight_dtype)
            lines = batch["line"].to(device, dtype=weight_dtype)
            masks = batch["mask"].to(device)
            anti_masks = batch["anti_mask"].to(device)
            encoder_hidden_states = encoder_hidden_states_full[:images.shape[0]]

            # 编码到潜空间
            latent_images = vae.encode(images).latent_dist.sample()
            latent_images = latent_images * vae.config.scaling_factor

            masks = torch.nn.functional.interpolate(masks, size=latent_images.shape[2:], mode='nearest')
            anti_masks = torch.nn.functional.interpolate(anti_masks, size=latent_images.shape[2:],
                                                         mode='nearest')
            # 膨胀 mask
            kernel = torch.ones((1, 1, 3, 3), device=device)
            masks = torch.clamp(
                F.conv2d(masks, kernel, padding=1),
                0, 1
            )
            anti_masks = 1 - masks

            # 逐步去噪，同时替换完好区域
            # 从标准正态分布中采样初始噪声 x_T ~ N(0,I)
            # [b, c, size, size]
            x_t = torch.randn_like(latent_images).to(device)

            # 从标准正态分布中采样噪声 ε~N(0,I)
            fixed_noises = x_t.clone()
            # print(noise_scheduler.config.num_train_timesteps)
            # for t in reversed(range(2)):
            for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                t_batch = torch.tensor([t] * latent_images.shape[0]).to(device)

                gt_noisy = noise_scheduler.add_noise(latent_images, fixed_noises, t_batch)

                # 在每一步的去噪过程中，将完好区域替换为原图的潜空间表示（加噪声后的），残缺区域保持去噪过程中的结果
                x_t = x_t * masks + gt_noisy * anti_masks
                x_t = x_t.to(dtype=weight_dtype)

                controlnet_images = lines

                down_block_res_samples, mid_block_res_sample = controlnet(
                    x_t,
                    t_batch,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_images,
                    return_dict=False,
                )

                predicted_noise = unet(
                    x_t,
                    t_batch,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # predicted_noise = unet(
                #     x_t,
                #     t_batch,
                #     encoder_hidden_states=encoder_hidden_states,
                #     return_dict=False,
                # )[0]

                prev_samples = []
                for i in range(x_t.shape[0]):
                    step_output = noise_scheduler.step(
                        predicted_noise[i:i + 1],
                        t_batch[i],
                        x_t[i:i + 1],
                        return_dict=True,
                    )
                    prev_samples.append(step_output.prev_sample)

                # x_t = step_output.prev_sample
                x_t = torch.cat(prev_samples, dim=0)
            # x_t = x_t * masks + latent_images * anti_masks
            # x_t = x_t.to(dtype=weight_dtype)
            x_t_decode = vae.decode(x_t / vae.config.scaling_factor).sample

            x_t_decode = x_t_decode * batch["mask"].to(device) + original_images * batch["anti_mask"].to(device)
            x_t_decode = x_t_decode.to(dtype=weight_dtype)

            ddpm_PSNR = psnr(x_t_decode, original_images)
            ddpm_ms_ssim = ms_ssim(x_t_decode, original_images, data_range=2.0, size_average=True)
            ddpm_loss = criterion(x_t_decode, original_images)

            ddpm_PSNRs += ddpm_PSNR.item()
            ddpm_ms_ssims += ddpm_ms_ssim.item()
            ddpm_losses += ddpm_loss.item()

            ddpm_loader_tqdm.set_postfix({
                "ddpm_PSNR": ddpm_PSNRs / (batch_idx + 1),
                "ddpm_ms_ssim": ddpm_ms_ssims / (batch_idx + 1),
                "ddpm_loss": ddpm_losses / (batch_idx + 1),
            })

            pred_images_list.append(x_t_decode.detach().cpu())
            original_images_list.append(original_images.detach().cpu())
            input_images_list.append(images.detach().cpu())
            lines_list.append(lines.detach().cpu())
            masks_list.append(batch["mask"].detach().cpu())
            names_list.append(batch["name"])

        avg_ddpm_PSNR = torch.tensor(ddpm_PSNRs / len(ddpm_loader), device=device)
        avg_ddpm_ms_ssim = torch.tensor(ddpm_ms_ssims / len(ddpm_loader), device=device)
        avg_ddpm_loss = torch.tensor(ddpm_losses / len(ddpm_loader), device=device)

        global_ddpm_PSNR = accelerator.reduce(avg_ddpm_PSNR, reduction='mean')
        global_ddpm_ms_ssim = accelerator.reduce(avg_ddpm_ms_ssim, reduction='mean')
        global_ddpm_loss = accelerator.reduce(avg_ddpm_loss, reduction='mean')

        if accelerator.is_main_process:
            print("PSNR/val", global_ddpm_PSNR)
            print("MS_SSIM/val", global_ddpm_ms_ssim)
            print("MSE/val", global_ddpm_loss)

    return original_images_list, pred_images_list, input_images_list, lines_list, masks_list, names_list


def get_pred_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_image_dir", type=str, default="../../DH/image_mask/val/images",
                        help="Directory containing validation images")
    parser.add_argument("--val_line_dir", type=str, default="../../DH/image_mask/val/lines",
                        help="Directory containing validation line drawings")
    parser.add_argument("--val_mask_dir", type=str, default="../../DH/image_mask/val/masks",
                        help="Directory containing validation masks")

    # val_image_dir = "../../DH/image_mask/val/images"
    # val_line_dir = "../../DH/image_mask/val/lines"
    # val_mask_dir = "../../DH/image_mask/val/masks"

    parser.add_argument("--ddpm_image_dir", type=str, default="./samples/ddpm/images",
                        help="Directory containing DDPM images")
    parser.add_argument("--ddpm_line_dir", type=str, default="./samples/ddpm/lines",
                        help="Directory containing DDPM line drawings")
    parser.add_argument("--ddpm_mask_dir", type=str, default="./samples/ddpm/masks",
                        help="Directory containing DDPM masks")

    # ddpm_image_dir = "./samples/ddpm/images"
    # ddpm_line_dir = "./samples/ddpm/lines"
    # ddpm_mask_dir = "./samples/ddpm/masks"

    # ddpm_image_dir = "./samples/ddpm/images"
    # ddpm_line_dir = "./samples/ddpm/Finetune_ddpm/line"
    # ddpm_mask_dir = "./samples/ddpm/masks"

    # ddpm_image_dir = "./samples/Evaluation/LandscapeTest/gt"
    # ddpm_line_dir = "./samples/Evaluation/LandscapeTest/line"
    # ddpm_mask_dir = "./samples/Evaluation/LandscapeTest/mask"

    parser.add_argument("--gt_save_dir", type=str, default="./samples/ddpm/Finetune_ddpm/gt",
                        help="Directory to save ground truth images")
    parser.add_argument("--pred_save_dir", type=str, default="./samples/ddpm/Finetune_ddpm/pred",
                        help="Directory to save predicted images")
    parser.add_argument("--input_save_dir", type=str, default="./samples/ddpm/Finetune_ddpm/input",
                        help="Directory to save input images")
    parser.add_argument("--line_save_dir", type=str, default="./samples/ddpm/Finetune_ddpm/line",
                        help="Directory to save line drawings")
    parser.add_argument("--mask_save_dir", type=str, default="./samples/ddpm/Finetune_ddpm/mask",
                        help="Directory to save masks")

    # gt_save_dir = "./samples/ddpm/Finetune_ddpm/gt"
    # pred_save_dir = "./samples/ddpm/Finetune_ddpm/pred"
    # input_save_dir = "./samples/ddpm/Finetune_ddpm/input"
    # line_save_dir = "./samples/ddpm/Finetune_ddpm/line"
    # mask_save_dir = "./samples/ddpm/Finetune_ddpm/mask"

    # gt_save_dir = "./samples/Evaluation/result/gt"
    # pred_save_dir = "./samples/Evaluation/result/pred"
    # input_save_dir = "./samples/Evaluation/result/input"
    # line_save_dir = "./samples/Evaluation/result/line"
    # mask_save_dir = "./samples/Evaluation/result/mask"

    # gt_save_dir = "./samples/Evaluation/FinetuneControlNet/gt"
    # pred_save_dir = "./samples/Evaluation/FinetuneControlNet/pred"
    # input_save_dir = "./samples/Evaluation/FinetuneControlNet/input"
    # line_save_dir = "./samples/Evaluation/FinetuneControlNet/line"
    # mask_save_dir = "./samples/Evaluation/FinetuneControlNet/mask"

    # gt_save_dir = "./samples/Evaluation/ControlNet/gt"
    # pred_save_dir = "./samples/Evaluation/ControlNet/pred"
    # input_save_dir = "./samples/Evaluation/ControlNet/input"
    # line_save_dir = "./samples/Evaluation/ControlNet/line"
    # mask_save_dir = "./samples/Evaluation/ControlNet/mask"

    # gt_save_dir = "./samples/Evaluation/用mask不用轮廓/gt"
    # pred_save_dir = "./samples/Evaluation/用mask不用轮廓/pred"
    # input_save_dir = "./samples/Evaluation/用mask不用轮廓/input"
    # line_save_dir = "./samples/Evaluation/用mask不用轮廓/line"
    # mask_save_dir = "./samples/Evaluation/用mask不用轮廓/mask"

    # gt_save_dir = "./samples/Evaluation/用轮廓不用mask/gt"
    # pred_save_dir = "./samples/Evaluation/用轮廓不用mask/pred"
    # input_save_dir = "./samples/Evaluation/用轮廓不用mask/input"
    # line_save_dir = "./samples/Evaluation/用轮廓不用mask/line"
    # mask_save_dir = "./samples/Evaluation/用轮廓不用mask/mask"

    # gt_save_dir = "./samples/Evaluation/LandscapeTest/gt"
    # pred_save_dir = "./samples/Evaluation/LandscapeTest/pred"
    # input_save_dir = "./samples/Evaluation/LandscapeTest/input"
    # line_save_dir = "./samples/Evaluation/LandscapeTest/line"
    # mask_save_dir = "./samples/Evaluation/LandscapeTest/mask"
    # ====================================================================

    # gt_save_dir = "./samples/ddpm/扩散完后直接恢复完好区域/gt"
    # pred_save_dir = "./samples/ddpm/扩散完后直接恢复完好区域/pred"
    # input_save_dir = "./samples/ddpm/扩散完后直接恢复完好区域/input"
    # line_save_dir = "./samples/ddpm/扩散完后直接恢复完好区域/line"
    # mask_save_dir = "./samples/ddpm/扩散完后直接恢复完好区域/mask"

    # gt_save_dir = "./samples/ddpm/不膨胀/gt"
    # pred_save_dir = "./samples/ddpm/不膨胀/pred"
    # input_save_dir = "./samples/ddpm/不膨胀/input"
    # line_save_dir = "./samples/ddpm/不膨胀/line"
    # mask_save_dir = "./samples/ddpm/不膨胀/mask"

    # gt_save_dir = "./samples/ddpm/用mask不用轮廓/gt"
    # pred_save_dir = "./samples/ddpm/用mask不用轮廓/pred"
    # input_save_dir = "./samples/ddpm/用mask不用轮廓/input"
    # line_save_dir = "./samples/ddpm/用mask不用轮廓/line"
    # mask_save_dir = "./samples/ddpm/用mask不用轮廓/mask"

    # gt_save_dir = "./samples/ddpm/用轮廓不用mask/gt"
    # pred_save_dir = "./samples/ddpm/用轮廓不用mask/pred"
    # input_save_dir = "./samples/ddpm/用轮廓不用mask/input"
    # line_save_dir = "./samples/ddpm/用轮廓不用mask/line"
    # mask_save_dir = "./samples/ddpm/用轮廓不用mask/mask"

    # gt_save_dir = "./samples/ddpm/不用轮廓不用mask/gt"
    # pred_save_dir = "./samples/ddpm/不用轮廓不用mask/pred"
    # input_save_dir = "./samples/ddpm/不用轮廓不用mask/input"
    # line_save_dir = "./samples/ddpm/不用轮廓不用mask/line"
    # mask_save_dir = "./samples/ddpm/不用轮廓不用mask/mask"

    parser.add_argument("--sd_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Path to the Stable Diffusion model")
    parser.add_argument("--controlnet_path", type=str, default="./checkpoints/controlnet_best_PSNR",
                        help="Path to the ControlNet model")

    # sd_name = "runwayml/stable-diffusion-v1-5"
    # controlnet_name = "lllyasviel/control_v11p_sd15_lineart"
    # controlnet_name = "./checkpoints/controlnet_best_PSNR"

    parser.add_argument("--image_size", type=int, default=512, help="Size to which input images will be resized")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for evaluation")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help='Device to use for training (e.g., "cuda" or "cpu")')

    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        help='Mixed precision mode (e.g., "no", "fp16", "bf16")')

    # 互斥的参数组，用户只能选择保存结果或者不保存结果，不能同时选择两者
    group = parser.add_mutually_exclusive_group()
    # 保存
    group.add_argument("--save_results", action="store_true", dest="save_results",
                       help="Save results")
    # 不保存
    group.add_argument("--no_save_results", action="store_false", dest="save_results",
                       help="Do not save results")
    # 默认保存结果
    parser.set_defaults(save_results=True)

    if len(sys.argv) == 1:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_pred_args()

    seeds(args.seed)

    # 检查当前系统是否有可用的 GPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    accelerator = Accelerator(
        mixed_precision="fp16",
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    print("weight_dtype:", weight_dtype)
    device = accelerator.device
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Total GPUs detected: {accelerator.num_processes}")
    print(f"Training on {device}, if want to use CPU, please ensure CUDA is properly installed and configured.")

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path,
        # torch_dtype=weight_dtype,
        local_files_only=True,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.sd_path,
        controlnet=controlnet,
        # torch_dtype=weight_dtype,
        safety_checker=None,
        local_files_only=True,
    )

    # pipe.enable_xformers_memory_efficient_attention()

    vae = pipe.vae
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    val_dataset = RestorationMuralDataset(args.val_image_dir, args.val_line_dir, args.val_mask_dir, size=512)
    val_indices = torch.randperm(len(val_dataset))[:200]
    val_dataset = Subset(val_dataset, val_indices)
    val_dataloader = DataLoaderX(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # 从val_dataloader里随机挑选10张图片作为ddpm_dataloader
    indices = torch.randperm(len(val_dataset))[:100]

    # 在val_dataset抽取对应indices的子集作为ddpm_dataset
    # ddpm_dataset = Subset(val_dataset, indices)
    # 直接从原始数据目录构建ddpm_dataset
    ddpm_dataset = RestorationMuralDataset(args.ddpm_image_dir, args.ddpm_line_dir, args.ddpm_mask_dir, size=512)

    ddpm_dataloader = DataLoaderX(
        ddpm_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    print("val:", len(val_dataloader))
    print("ddpm:", len(ddpm_dataloader))

    criterion = nn.MSELoss()

    controlnet, val_dataloader, ddpm_dataloader = accelerator.prepare(
        controlnet, val_dataloader, ddpm_dataloader
    )

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae.eval()
    unet.eval()
    text_encoder.eval()

    controlnet.requires_grad_(True)

    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)

    controlnet.to(device)

    prompt = [""] * 64
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    with torch.no_grad():
        encoder_hidden_states_full = text_encoder(input_ids)[0]

    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    # noise_scheduler = DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")

    # test1(val_dataloader, ddpm_dataloader, encoder_hidden_states_full,
    #       controlnet, vae, unet, tokenizer, text_encoder,
    #       criterion,
    #       noise_scheduler,
    #       weight_dtype,
    #       device, accelerator)
    original_images_list, pred_images_list, input_images_list, lines_list, masks_list, names_list = pred(
        ddpm_dataloader,
        encoder_hidden_states_full,
        controlnet, vae, unet,
        tokenizer, text_encoder,
        criterion,
        noise_scheduler,
        weight_dtype,
        device, accelerator)

    if args.save_results:
        # 创建保存结果的目录
        os.makedirs(args.gt_save_dir, exist_ok=True)
        os.makedirs(args.pred_save_dir, exist_ok=True)
        os.makedirs(args.input_save_dir, exist_ok=True)
        os.makedirs(args.line_save_dir, exist_ok=True)
        os.makedirs(args.mask_save_dir, exist_ok=True)

        # 画图并保存结果
        plot(original_images_list, pred_images_list, input_images_list, lines_list, masks_list,
             args.gt_save_dir, args.pred_save_dir, args.input_save_dir, args.line_save_dir, args.mask_save_dir,
             names_list, args.save_results)
    else:
        # 画图
        plot(original_images_list, pred_images_list, input_images_list, lines_list, masks_list,
             args.gt_save_dir, args.pred_save_dir, args.input_save_dir, args.line_save_dir, args.mask_save_dir,
             names_list, args.save_results)
