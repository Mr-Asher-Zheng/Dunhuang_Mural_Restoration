import argparse
import sys
import time
from diffusers import StableDiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel, DDPMScheduler, \
    DDIMScheduler
from torch.utils.data import Subset
from accelerate import Accelerator
from pytorch_msssim import ms_ssim, SSIM

from utils import *


def train_and_val(train_loader, val_loader, ddpm_loader, encoder_hidden_states_full,
                  controlnet, vae, unet, tokenizer, text_encoder,
                  criterion,
                  optimizer,
                  scheduler, noise_scheduler,
                  config, weight_dtype,
                  total_steps, valid_steps, ddpm_steps,
                  device, accelerator):
    os.makedirs("./checkpoints", exist_ok=True)
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=f"./logs/ControlNet_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    best_val_PSNR = 0.0
    best_val_loss = float('inf')

    train_iterator = iter(train_loader)
    train_loader_tqdm = tqdm(range(total_steps), desc="Training", leave=False)

    for step in train_loader_tqdm:
        controlnet.train()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        images = batch["image"].to(device, dtype=weight_dtype)
        lines = batch["line"].to(device, dtype=weight_dtype)
        masks = batch["mask"].to(device)
        anti_masks = batch["anti_mask"].to(device)
        encoder_hidden_states = encoder_hidden_states_full[:images.shape[0]]

        # print("images:", images.dtype)

        with torch.no_grad():
            latent_images = vae.encode(images).latent_dist.sample()
            latent_images = latent_images * vae.config.scaling_factor

        # print("latent_images:", latent_images.dtype)

        # print("latent_images:", latent_images.shape)

        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latent_images.shape[0],),
                          device=device).long()

        # print("t:", t.shape)

        noises = torch.randn_like(latent_images).to(device)
        noisy_latent_images = noise_scheduler.add_noise(latent_images, noises, t)

        # print("noisy_latent_images:", noisy_latent_images.dtype)

        # print("noises:", noises.shape)
        # print("noisy_latent_images:", noisy_latent_images.shape)

        # print("encoder_hidden_states:", encoder_hidden_states.shape)

        controlnet_images = lines
        # print("controlnet_images:", controlnet_images.shape)

        down_block_res_samples, mid_block_res_sample = controlnet(
            noisy_latent_images,
            t,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_images,
            return_dict=False,
        )

        # print("down_block_res_samples:", [sample.shape for sample in down_block_res_samples])
        # print("mid_block_res_sample:", mid_block_res_sample.shape)

        # for sample in down_block_res_samples:
        #     print(sample.dtype)

        # for sample in down_block_res_samples:
        #     print(sample.to(dtype=weight_dtype).dtype)

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

        # print("model_pred:", model_pred.dtype)

        # print("model_pred:", model_pred.shape)
        # time.sleep(60)

        # 计算 loss
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noises
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latent_images, noises, t)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=False)

        train_loader_tqdm.set_postfix(loss=loss.item())

        if (step + 1) % 25 == 0:
            global_train_loss = accelerator.reduce(loss, reduction="mean")
            if accelerator.is_main_process:
                writer.add_scalar("Loss/train", global_train_loss, step + 1)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], step + 1)

        if (step + 1) % valid_steps == 0:
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

                    t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latent_images.shape[0],),
                                      device=device).long()

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
                writer.add_scalar("PSNR/val", global_val_PSNR, step + 1)
                writer.add_scalar("MS_SSIM/val", global_val_ms_ssim, step + 1)
                writer.add_scalar("MSE/val", global_val_loss, step + 1)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                if global_val_PSNR > best_val_PSNR:
                    best_val_PSNR = global_val_PSNR

                    unwrapped_controlnet = accelerator.unwrap_model(controlnet)

                    unwrapped_controlnet.save_pretrained(
                        f"./checkpoints/controlnet_best_PSNR",
                        safe_serialization=True,
                    )

        # ddpm全程扩散，在训练时打开会很费时间，不建议用此方法验证训练效果
        # if (step + 1) % ddpm_steps == 0:
        #     controlnet.eval()
        #     ddpm_loader_tqdm = tqdm(ddpm_loader, desc="ddpm", leave=False)
        #
        #     with torch.no_grad():
        #         for i, batch in enumerate(ddpm_loader_tqdm):
        #             images = batch["image"].to(device, dtype=weight_dtype)
        #             lines = batch["line"].to(device, dtype=weight_dtype)
        #             masks = batch["mask"].to(device)
        #             anti_masks = batch["anti_mask"].to(device)
        #             encoder_hidden_states = encoder_hidden_states_full[:images.shape[0]]
        #
        #             # 编码到潜空间
        #             latent_images = vae.encode(images).latent_dist.sample()
        #             latent_images = latent_images * vae.config.scaling_factor
        #
        #             # print("latent_images:", latent_images.dtype)
        #
        #             masks = torch.nn.functional.interpolate(masks, size=latent_images.shape[2:], mode='nearest')
        #             anti_masks = torch.nn.functional.interpolate(anti_masks, size=latent_images.shape[2:],
        #                                                          mode='nearest')
        #             # 逐步去噪，同时替换完好区域
        #             # 从标准正态分布中采样初始噪声 x_T ~ N(0,I)
        #             # [b, c, size, size]
        #             x_t = torch.randn_like(latent_images).to(device)
        #             # print("x_t", x_t.shape)
        #             # print("x_t:", x_t.dtype)
        #
        #             # 从标准正态分布中采样噪声 ε~N(0,I)
        #             fixed_noises = x_t.clone()
        #             print(noise_scheduler.config.num_train_timesteps)
        #             for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
        #                 t_batch = torch.tensor([t] * latent_images.shape[0]).to(device)
        #
        #                 gt_noisy = noise_scheduler.add_noise(latent_images, fixed_noises, t_batch)
        #                 # print("gt_noisy:", gt_noisy.dtype)
        #
        #                 x_t = x_t * masks + gt_noisy * anti_masks
        #                 x_t = x_t.to(dtype=weight_dtype)
        #                 # print("x_t:", x_t.dtype)
        #
        #                 controlnet_images = lines
        #
        #                 down_block_res_samples, mid_block_res_sample = controlnet(
        #                     x_t,
        #                     t_batch,
        #                     encoder_hidden_states=encoder_hidden_states,
        #                     controlnet_cond=controlnet_images,
        #                     return_dict=False,
        #                 )
        #
        #                 predicted_noise = unet(
        #                     x_t,
        #                     t_batch,
        #                     encoder_hidden_states=encoder_hidden_states,
        #                     down_block_additional_residuals=[
        #                         sample.to(dtype=weight_dtype) for sample in down_block_res_samples
        #                     ],
        #                     mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
        #                     return_dict=False,
        #                 )[0]
        #
        #                 # print("predicted_noise:", predicted_noise.shape)
        #                 # print("t_batch:", t_batch.shape)
        #                 # print("x_t:", x_t.shape)
        #
        #                 prev_samples = []
        #                 for i in range(x_t.shape[0]):
        #                     step_output = noise_scheduler.step(
        #                         predicted_noise[i:i+1],
        #                         t_batch[i],
        #                         x_t[i:i+1],
        #                         return_dict=True,
        #                     )
        #                     prev_samples.append(step_output.prev_sample)
        #
        #                 # x_t = step_output.prev_sample
        #                 x_t = torch.cat(prev_samples, dim=0)
        #                 x_t_decode = vae.decode(x_t / vae.config.scaling_factor).sample
        #
        #                 for index in range(x_t_decode.shape[0]):
        #                     img = x_t_decode[index].detach().cpu()
        #                     img = img * 0.5 + 0.5
        #                     img = torch.clip(img, 0, 1)
        #                     img = (img * 255).to(torch.uint8)
        #                     if accelerator.is_main_process:
        #                         writer.add_image(f'ddpm_step_{index}', img, step + 1, dataformats='CHW')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_controlnet = accelerator.unwrap_model(controlnet)
        unwrapped_controlnet.save_pretrained(
            f"./checkpoints/controlnet_final",
            safe_serialization=True,
        )
        print("Done Training!")


def get_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("train_image_dir", type=str, default="../../DH/image_mask/train/images",
                        help="Path to the training images directory")
    parser.add_argument("train_line_dir", type=str, default="../../DH/image_mask/train/lines",
                        help="Path to the training line directory")
    parser.add_argument("train_mask_dir", type=str, default="../../DH/image_mask/train/masks",
                        help="Path to the training mask directory")

    parser.add_argument("val_image_dir", type=str, default="../../DH/image_mask/val/images",
                        help="Path to the validation images directory")
    parser.add_argument("val_line_dir", type=str, default="../../DH/image_mask/val/lines",
                        help="Path to the validation line directory")
    parser.add_argument("val_mask_dir", type=str, default="../../DH/image_mask/val/masks",
                        help="Path to the validation mask directory")

    parser.add_argument("--sd_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Path to the Stable Diffusion model")
    parser.add_argument("--controlnet_path", type=str, default="lllyasviel/control_v11p_sd15_lineart",
                        help="Path to the ControlNet model")

    parser.add_argument("--total_steps", type=int, default=15000, help="Total number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--valid_steps", type=int, default=500, help="Number of steps between validations")
    parser.add_argument("--ddpm_steps", type=int, default=1000, help="Number of steps between ddpm evaluations")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")

    parser.add_argument("--image_size", type=int, default=512, help="Size of the input images")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and validation")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help='Device to use for training (e.g., "cuda" or "cpu")')

    parser.add_argument("--mixed_precision", type=str, default="fp16",
                        help='Mixed precision mode (e.g., "no", "fp16", "bf16")')
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

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
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
        # controlnet_name,
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

    train_dataset = RestorationMuralDataset(args.train_image_dir, args.train_line_dir, args.train_mask_dir,
                                            size=args.image_size)
    train_dataloader = DataLoaderX(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = RestorationMuralDataset(args.val_image_dir, args.val_line_dir, args.val_mask_dir,
                                          size=args.image_size)
    val_indices = torch.randperm(len(val_dataset))[:2]
    val_dataset = Subset(val_dataset, val_indices)
    val_dataloader = DataLoaderX(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # 从val_dataloader里随机挑选10张图片作为ddpm_dataloader
    indices = torch.randperm(len(val_dataset))[:4]
    ddpm_dataset = Subset(val_dataset, indices)
    ddpm_dataloader = DataLoaderX(
        ddpm_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print("train:", len(train_dataloader))
    print("val:", len(val_dataloader))
    print("ddpm:", len(ddpm_dataloader))

    # total_steps = 15000
    # warmup_steps = 1000
    # valid_steps = 500
    # ddpm_steps = 1000
    # learning_rate = 5e-5
    learning_rate = args.learning_rate * args.batch_size * accelerator.num_processes

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8
    )

    controlnet, optimizer, train_dataloader, val_dataloader, ddpm_dataloader = accelerator.prepare(
        controlnet, optimizer, train_dataloader, val_dataloader, ddpm_dataloader
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

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.total_steps)
    noise_scheduler = DDPMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")
    # noise_scheduler = DDIMScheduler.from_pretrained(args.sd_path, subfolder="scheduler")

    config = {
        "mse_weight": 1.0,
    }

    train_and_val(train_dataloader, val_dataloader, ddpm_dataloader, encoder_hidden_states_full,
                  controlnet, vae, unet, tokenizer, text_encoder,
                  criterion,
                  optimizer,
                  scheduler, noise_scheduler,
                  config, weight_dtype,
                  args.total_steps, args.valid_steps, args.ddpm_steps,
                  device, accelerator)
