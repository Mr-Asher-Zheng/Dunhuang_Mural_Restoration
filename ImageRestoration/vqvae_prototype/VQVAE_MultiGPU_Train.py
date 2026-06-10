import math
import os
import time
from torchvision import transforms
import lpips
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from prefetch_generator import BackgroundGenerator
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ms_ssim, SSIM
import torch.nn.functional as F

from VQVAE import *
from discriminator import *


def seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


class VQVAEMuralDataset(Dataset):
    def __init__(self, image_dir, size=256):
        self.image_list = sorted([os.path.join(image_dir, f)
                                  for f in os.listdir(image_dir) if f.endswith(('png', 'jpg'))])
        self.size = size

        self.base_transform = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])

        self.image_transform = transforms.Compose([
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        # self.lpips_image_transform = transforms.Compose([
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        # ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # 分布范围[0, 255]
        image = Image.open(self.image_list[idx]).convert("RGB")

        # 分布范围[0, 1]
        base = self.base_transform(image)

        # 分布范围[-1, 1]
        image = self.image_transform(base)

        # lpips_image = self.lpips_image_transform(base)

        return {"image": image}


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def d_hinge_loss(real_logit, fake_logit):
    real_loss = torch.mean(F.relu(1.0 - real_logit))
    fake_loss = torch.mean(F.relu(1.0 + fake_logit))
    return real_loss + fake_loss


def g_hinge_loss(fake_logit):
    return -torch.mean(fake_logit)


def train_and_val(train_loader, val_loader,
                  model, lpips_model, discriminator,
                  recon_criterion, disc_criterion,
                  optimizer_d, optimizer_g,
                  scheduler, config,
                  total_steps, valid_steps,
                  disc_step_start,
                  device, accelerator):
    os.makedirs("../model", exist_ok=True)
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=f"./logs/VQVAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    step_count = 0

    train_g_losses = 0.0
    train_disc_losses = 0.0

    train_g_loss = torch.tensor(0.0, device=device)
    train_disc_loss = torch.tensor(0.0, device=device)
    train_disc_fake_loss = torch.tensor(0.0, device=device)

    best_g_loss = float('inf')
    best_disc_loss = float('inf')

    val_loader_len = len(val_loader)

    train_iterator = iter(train_loader)
    train_loader_tqdm = tqdm(range(total_steps), desc="Train", leave=False)

    ssim = SSIM(data_range=2.0, size_average=True, channel=3)

    for step in train_loader_tqdm:
        # 判别器开始训练的计数器
        step_count += 1

        model.train()
        discriminator.eval()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        images = batch['image']
        # lpips_images = batch['lpips_image']

        ######## Generator #########
        model_output = model(images)
        # output, z, quantize_losses = model_output
        output, z, quantize_loss, perplexity = model_output

        # 重建损失
        train_recon_loss = config["mse_weight"] * recon_criterion(output, images)
        # train_codebook_loss = config["codebook_weight"] * quantize_losses['codebook_loss']
        # train_commitment_loss = config["commitment_weight"] * quantize_losses['commitment_loss']
        train_commitment_loss = config["commitment_weight"] * quantize_loss

        # train_ms_ssim_loss = (1 - ms_ssim(output, images, data_range=2.0, size_average=True))
        train_ms_ssim_loss = 1 - ssim(output, images)

        train_ms_ssim_loss = config["ms_ssim_weight"] * train_ms_ssim_loss
        # 感知损失
        train_lpips_loss = torch.mean(lpips_model(output, images))
        train_lpips_loss = config["lpips_weight"] * train_lpips_loss

        train_g_loss = train_recon_loss + train_ms_ssim_loss + train_commitment_loss + train_lpips_loss

        # train_g_loss = train_recon_loss + train_ms_ssim_loss + train_codebook_loss + train_commitment_loss

        # train_g_loss = train_recon_loss + train_codebook_loss + train_commitment_loss

        # 当计数器达到某值时开启
        # 对抗损失
        if step_count > disc_step_start:
            disc_fake_pred = discriminator(output)

            # # bce：
            # train_disc_fake_loss = disc_criterion(
            #     disc_fake_pred,
            #     torch.ones_like(disc_fake_pred)
            # )
            # train_disc_fake_loss = 0.5 * train_disc_fake_loss
            # train_g_loss += train_disc_fake_loss

            # hinge：
            train_disc_fake_loss = g_hinge_loss(disc_fake_pred)
            train_disc_fake_loss = config["disc_weight"] * train_disc_fake_loss
            train_g_loss += train_disc_fake_loss

        optimizer_g.zero_grad()
        # train_g_loss.backward(retain_graph=True)
        accelerator.backward(train_g_loss)

        if accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer_g.step()
        scheduler.step()

        ###########################

        ######## Discriminator #########
        if step_count > disc_step_start:
            model.eval()
            discriminator.train()

            fake = output.detach()
            disc_fake_pred = discriminator(fake)
            disc_real_pred = discriminator(images)

            # # bce：
            # disc_fake_loss = disc_criterion(
            #     disc_fake_pred,
            #     torch.zeros_like(disc_fake_pred)
            # )
            # disc_real_loss = disc_criterion(
            #     disc_real_pred,
            #     torch.ones_like(disc_real_pred)
            # )
            # train_disc_loss = (disc_fake_loss + disc_real_loss) / 2

            # hinge：
            train_disc_loss = d_hinge_loss(disc_real_pred, disc_fake_pred)

            optimizer_d.zero_grad()
            # train_disc_loss.backward()
            # accelerator.backward的参数必须是局部的loss
            accelerator.backward(train_disc_loss)
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()

            # # 累计判别器损失
            # disc_losses = accelerator.reduce(train_disc_loss, reduction='sum')
        #############################

        # # 累计生成器损失
        # g_losses = accelerator.reduce(train_g_loss, reduction='sum')

        # train_loader_tqdm.set_postfix(g_loss=train_g_losses.item(), disc_loss=train_disc_losses.item())

        train_loader_tqdm.set_postfix(train_g_loss=train_g_loss.item(), disc_loss=train_disc_loss.item())

        # 每25步主进程进行日志记录
        if step_count % 25 == 0:
            global_train_g_loss = accelerator.reduce(train_g_loss, reduction='mean')
            global_train_recon_loss = accelerator.reduce(train_recon_loss, reduction='mean')
            global_train_lpips_loss = accelerator.reduce(train_lpips_loss, reduction='mean')
            # global_train_codebook_loss = accelerator.reduce(train_codebook_loss, reduction='mean')
            global_train_commitment_loss = accelerator.reduce(train_commitment_loss, reduction='mean')
            global_perplexity = accelerator.reduce(perplexity, reduction='mean')

            global_train_ms_ssim_loss = accelerator.reduce(train_ms_ssim_loss, reduction='mean')
            if step_count > disc_step_start:
                global_train_disc_loss = accelerator.reduce(train_disc_loss, reduction='mean')
                global_train_disc_fake_loss = accelerator.reduce(train_disc_fake_loss, reduction='mean')

            if accelerator.is_main_process:
                writer.add_scalar('g_loss/train', global_train_g_loss, step_count)
                writer.add_scalar('recon_loss/train', global_train_recon_loss, step_count)
                writer.add_scalar('lpips_loss/train', global_train_lpips_loss, step_count)
                # writer.add_scalar('codebook_loss/train', global_train_codebook_loss, step_count)
                writer.add_scalar('commitment_loss/train', global_train_commitment_loss, step_count)
                writer.add_scalar('perplexity/train', global_perplexity, step_count)

                writer.add_scalar('ms_ssim_loss/train', global_train_ms_ssim_loss, step_count)
                if step_count > disc_step_start:
                    writer.add_scalar('disc_loss/train', global_train_disc_loss, step_count)
                    writer.add_scalar('disc_fake_loss/train', global_train_disc_fake_loss, step_count)

        if (step + 1) % valid_steps == 0:
            model.eval()
            discriminator.eval()

            val_g_losses = 0.0
            val_disc_losses = 0.0

            val_recon_losses = 0.0
            val_codebook_losses = 0.0
            val_commitment_losses = 0.0
            val_perplexity = 0.0
            val_lpips_losses = 0.0
            val_ms_ssim_losses = 0.0
            val_disc_fake_losses = 0.0

            val_loader_tqdm = tqdm(val_loader, desc="Valid", leave=False)
            for i, batch in enumerate(val_loader_tqdm):
                with torch.no_grad():
                    images = batch['image']
                    # lpips_images = batch['lpips_image']

                    ######## Generator #########
                    model_output = model(images)
                    # output, z, quantize_losses = model_output
                    output, z, quantize_loss, perplexity = model_output

                    if i == 0:
                        for index in range(output.shape[0]):
                            img = output[index].detach().cpu()
                            img = img * 0.5 + 0.5
                            img = torch.clip(img, 0, 1)
                            img = (img * 255).to(torch.uint8)
                            if accelerator.is_main_process:
                                writer.add_image(f'val_{index}', img, step_count, dataformats='CHW')

                    # 重建损失
                    val_recon_loss = config["mse_weight"] * recon_criterion(output, images)
                    # val_codebook_loss = config["codebook_weight"] * quantize_losses['codebook_loss']
                    # val_commitment_loss = config["commitment_weight"] * quantize_losses['commitment_loss']
                    val_commitment_loss = config["commitment_weight"] * quantize_loss

                    # 感知损失
                    val_lpips_loss = torch.mean(lpips_model(output, images))
                    val_lpips_loss = config["lpips_weight"] * val_lpips_loss

                    # val_ms_ssim_loss = (1 - ms_ssim(output, images, data_range=2.0, size_average=True))
                    val_ms_ssim_loss = 1 - ssim(output, images)
                    val_ms_ssim_loss = config["ms_ssim_weight"] * val_ms_ssim_loss

                    val_recon_losses += val_recon_loss.item()
                    # val_codebook_losses += val_codebook_loss.item()
                    val_commitment_losses += val_commitment_loss.item()
                    val_perplexity += perplexity.item()
                    val_lpips_losses += val_lpips_loss.item()

                    val_ms_ssim_losses += val_ms_ssim_loss.item()
                    # val_g_loss = val_recon_loss + val_ms_ssim_loss + val_codebook_loss + val_commitment_loss + val_lpips_loss
                    val_g_loss = val_recon_loss + val_ms_ssim_loss + val_commitment_loss + val_lpips_loss

                    # val_g_loss = val_recon_loss + val_codebook_loss + val_commitment_loss + val_lpips_loss

                    # 对抗损失
                    if step_count > disc_step_start:
                        disc_fake_pred = discriminator(output)

                        # # bce：
                        # val_disc_fake_loss = disc_criterion(
                        #     disc_fake_pred,
                        #     torch.ones_like(disc_fake_pred)
                        # )
                        # val_disc_fake_loss = 0.5 * val_disc_fake_loss
                        # val_disc_fake_losses += val_disc_fake_loss.item()

                        # hinge：
                        val_disc_fake_loss = g_hinge_loss(disc_fake_pred)
                        val_disc_fake_loss = config["disc_weight"] * val_disc_fake_loss
                        val_disc_fake_losses += val_disc_fake_loss.item()

                        val_g_loss += val_disc_fake_loss

                    val_g_losses += val_g_loss.item()
                    ################################

                    ######## Discriminator #########
                    if step_count > disc_step_start:
                        fake = output.detach()
                        disc_fake_pred = discriminator(fake)
                        disc_real_pred = discriminator(images)

                        # # bce：
                        # disc_fake_loss = disc_criterion(
                        #     disc_fake_pred,
                        #     torch.zeros_like(disc_fake_pred)
                        # )
                        # disc_real_loss = disc_criterion(
                        #     disc_real_pred,
                        #     torch.ones_like(disc_real_pred)
                        # )
                        # val_disc_loss = (disc_fake_loss + disc_real_loss) / 2

                        # hinge：
                        val_disc_loss = d_hinge_loss(disc_real_pred, disc_fake_pred)

                        val_disc_losses += val_disc_loss.item()
                    #############################

                val_loader_tqdm.set_postfix(val_g_loss=val_g_losses / (i + 1),
                                            val_disc_loss=val_disc_losses / (i + 1))

            model.train()
            discriminator.eval()

            avg_val_g_loss = torch.tensor(val_g_losses / val_loader_len, device=device)
            avg_val_recon_loss = torch.tensor(val_recon_losses / val_loader_len, device=device)
            # avg_val_codebook_loss = torch.tensor(val_codebook_losses / val_loader_len, device=device)
            avg_val_commitment_loss = torch.tensor(val_commitment_losses / val_loader_len, device=device)
            avg_val_lpips_loss = torch.tensor(val_lpips_losses / val_loader_len, device=device)
            avg_val_perplexity = torch.tensor(val_perplexity / val_loader_len, device=device)

            avg_val_ms_ssim_loss = torch.tensor(val_ms_ssim_losses / val_loader_len, device=device)
            if step_count > disc_step_start:
                avg_val_disc_loss = torch.tensor(val_disc_losses / val_loader_len, device=device)
                avg_val_disc_fake_loss = torch.tensor(val_disc_fake_losses / val_loader_len, device=device)

            global_val_g_loss = accelerator.reduce(avg_val_g_loss, reduction='mean')
            global_val_recon_loss = accelerator.reduce(avg_val_recon_loss, reduction='mean')
            # global_val_codebook_loss = accelerator.reduce(avg_val_codebook_loss, reduction='mean')
            global_val_commitment_loss = accelerator.reduce(avg_val_commitment_loss, reduction='mean')
            global_val_perplexity = accelerator.reduce(avg_val_perplexity, reduction='mean')
            global_val_lpips_loss = accelerator.reduce(avg_val_lpips_loss, reduction='mean')
            global_val_ms_ssim_loss = accelerator.reduce(avg_val_ms_ssim_loss, reduction='mean')

            if step_count > disc_step_start:
                global_val_disc_loss = accelerator.reduce(avg_val_disc_loss, reduction='mean')
                global_val_disc_fake_loss = accelerator.reduce(avg_val_disc_fake_loss, reduction='mean')

            if accelerator.is_main_process:
                writer.add_scalar('g_loss/val', global_val_g_loss, step_count)
                writer.add_scalar('recon_loss/val', global_val_recon_loss, step_count)
                # writer.add_scalar('codebook_loss/val', global_val_codebook_loss, step_count)
                writer.add_scalar('commitment_loss/val', global_val_commitment_loss, step_count)
                writer.add_scalar('perplexity/val', global_val_perplexity, step_count)
                writer.add_scalar('lpips_loss/val', global_val_lpips_loss, step_count)
                writer.add_scalar('ms_ssim_loss/val', global_val_ms_ssim_loss, step_count)

                if step_count > disc_step_start:
                    writer.add_scalar('disc_loss/val', global_val_disc_loss, step_count)
                    writer.add_scalar('disc_fake_loss/val', global_val_disc_fake_loss, step_count)

                writer.add_scalar('lr_model', optimizer_g.param_groups[0]['lr'], step_count)
                writer.add_scalar('lr_discriminator', optimizer_d.param_groups[0]['lr'], step_count)

            if step_count > disc_step_start:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if global_val_g_loss < best_g_loss:
                        best_g_loss = global_val_g_loss

                        unwrapped_model = accelerator.unwrap_model(model)
                        torch.save(unwrapped_model.state_dict(), f"../model/vqvae_autoencoder_best.pth")
                        unwrapped_discriminator = accelerator.unwrap_model(discriminator)
                        torch.save(unwrapped_discriminator.state_dict(), f"../model/vqvae_discriminator_best.pth")

                        print(
                            f"\nBest vqvae autoencoder and discriminator model saved at step {step_count} with g_loss: {best_g_loss:.4f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_discriminator = accelerator.unwrap_model(discriminator)
        torch.save(unwrapped_model.state_dict(), f"../model/vqvae_autoencoder_last.pth")
        torch.save(unwrapped_discriminator.state_dict(), f"../model/vqvae_discriminator_last.pth")
        print("Done Training.")


if __name__ == "__main__":
    seeds(42)

    # 调试代码
    # torch.autograd.set_detect_anomaly(True)

    # 检测GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    accelerator = Accelerator(
        mixed_precision="fp16",
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    device = accelerator.device
    print(f"Using device: {device}")
    print(f"Total GPUs detected: {accelerator.num_processes}")

    # train_image_dir = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural512"
    # train_image_dir = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\train\images"
    # val_image_dir = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\test\images"

    # train_image_dir = r"/kaggle/input/muraldh/MuralDH/MuralDH/Mural_seg/train/images"

    train_image_dir = r"../../../DH/image_mask/train/images"
    val_image_dir = r"../../../DH/image_mask/val/images"

    train_dataset = VQVAEMuralDataset(train_image_dir, size=128)
    val_dataset = VQVAEMuralDataset(val_image_dir, size=128)

    train_data_loader = DataLoaderX(
        train_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    val_data_loader = DataLoaderX(
        val_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = VQVAE()
    lpips_model = lpips.LPIPS(net='vgg').eval()
    discriminator = Discriminator(im_channels=3)

    total_steps = 15000
    warmup_steps = 1000
    valid_steps = 500
    disc_step_start = int(0.4 * total_steps)

    recon_criterion = nn.MSELoss()
    disc_criterion = nn.BCEWithLogitsLoss()

    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)

    model, discriminator, optimizer_g, optimizer_d, train_data_loader, val_data_loader = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, train_data_loader, val_data_loader
    )

    scheduler = get_cosine_schedule_with_warmup(optimizer_g, warmup_steps, total_steps)

    lpips_model = lpips_model.to(accelerator.device)

    config = {
        "mse_weight": 1.0,
        "lpips_weight": 1.0,
        # "codebook_weight": 1.0,
        "commitment_weight": 0.25,

        "disc_weight": 0.1,
        "ms_ssim_weight": 0.5,
    }

    train_and_val(train_data_loader, val_data_loader,
                  model, lpips_model, discriminator,
                  recon_criterion, disc_criterion,
                  optimizer_d, optimizer_g,
                  scheduler, config,
                  total_steps, valid_steps,
                  disc_step_start,
                  device, accelerator)
