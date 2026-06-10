import os
from torchvision import transforms, models
import lpips
import torch
from VQVAE import *
from discriminator import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from prefetch_generator import BackgroundGenerator
from PIL import Image
import numpy as np
from tqdm import tqdm
from datetime import datetime


def seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class VQVAEMuralDataset(Dataset):
    def __init__(self, image_dir, size=256):
        self.image_list = sorted([os.path.join(image_dir, f)
                                  for f in os.listdir(image_dir) if f.endswith(('png', 'jpg'))])
        self.size = size

        self.image_transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")

        image = self.image_transform(image)

        return {"image": image}


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


# def train_and_val(train_loader, val_loader,
#                   model, lpips_model, discriminator,
#                   recon_criterion, disc_criterion,
#                   optimizer_d, optimizer_g,
#                   scheduler,
#                   total_steps, valid_steps,
#                   step_count, disc_step_start,
#                   device):
#     os.makedirs("./model", exist_ok=True)
#     writer = SummaryWriter(log_dir=f"./logs/VQVAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
#     step_count = 0
#
#     best_g_loss = float('inf')
#     best_disc_loss = float('inf')
#
#     train_iterator = iter(train_loader)
#     train_loader_tqdm = tqdm(range(total_steps), desc="Train", leave=False)
#
#     for step in train_loader_tqdm:
#         model.train()
#
#         try:
#             batch = next(train_iterator)
#         except StopIteration:
#             train_iterator = iter(train_loader)
#             batch = next(train_iterator)





def train_step(train_loader,
               model, lpips_model, discriminator,
               recon_criterion, disc_criterion,
               optimizer_d, optimizer_g,
               epoch_idx, num_epochs,
               step_count, disc_step_start,
               writer, device):
    model.train()
    g_losses = 0.0
    disc_losses = 0.0

    g_loss = 0.0
    disc_loss = 0.0
    recon_loss = 0.0
    codebook_loss = 0.0
    commitment_loss = 0.0
    lpips_loss = 0.0

    train_loader_tqdm = tqdm(train_loader, desc=f"Train Epoch {epoch_idx + 1}/{num_epochs}", leave=True)
    for im in train_loader_tqdm:
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        step_count += 1
        # print(step_count)
        im = im['image'].float().to(device)

        ######## Generator #########
        model_output = model(im)
        output, z, quantize_losses = model_output

        # 重建损失
        recon_loss = recon_criterion(output, im)
        codebook_loss = 1 * quantize_losses['codebook_loss']
        commitment_loss = 0.2 * quantize_losses['commitment_loss']

        g_loss = recon_loss + codebook_loss + commitment_loss

        # 对抗损失
        if step_count > disc_step_start:
            disc_fake_pred = discriminator(output)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device)
            )
            g_loss += 0.5 * disc_fake_loss

        # 感知损失
        lpips_loss = torch.mean(lpips_model(output, im))

        g_loss += lpips_loss
        g_loss.backward(retain_graph=True)
        optimizer_g.step()

        ###########################

        ######## Discriminator #########
        if step_count > disc_step_start:
            fake = output
            disc_fake_pred = discriminator(fake.detach())
            disc_real_pred = discriminator(im)
            disc_fake_loss = disc_criterion(
                disc_fake_pred,
                torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device)
            )
            disc_real_loss = disc_criterion(
                disc_real_pred,
                torch.ones(disc_real_pred.shape, device=disc_real_pred.device)
            )
            disc_loss = (disc_fake_loss + disc_real_loss) / 2
            disc_loss.backward()
            optimizer_d.step()

            disc_losses += disc_loss.item()
        #############################

        # 累计损失
        g_losses += g_loss.item()
        train_loader_tqdm.set_postfix(g_loss=g_losses, disc_loss=disc_losses)

        writer.add_scalar('Loss/g_loss', g_loss, step_count)
        writer.add_scalar('Loss/disc_loss', disc_loss, step_count)
        writer.add_scalar('Loss/recon_loss', recon_loss, step_count)
        writer.add_scalar('Loss/codebook_loss', codebook_loss, step_count)
        writer.add_scalar('Loss/commitment_loss', commitment_loss, step_count)
        writer.add_scalar('Loss/lpips_loss', lpips_loss, step_count)

    return g_losses, disc_losses, step_count, writer


# g_loss, disc_loss, recon_loss, codebook_loss, commitment_loss, lpips_loss

def train_model(train_loader,
                model, lpips_model, discriminator,
                recon_criterion, disc_criterion,
                optimizer_d, optimizer_g,
                num_epochs, disc_step_start, device):
    os.makedirs("../model", exist_ok=True)
    writer = SummaryWriter(log_dir=f"./logs/VQVAE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    step_count = 0

    best_g_loss = float('inf')
    best_disc_loss = float('inf')

    for epoch_idx in range(num_epochs):
        g_losses, disc_losses, step_count, writer = train_step(train_loader,
                                                               model, lpips_model, discriminator,
                                                               recon_criterion, disc_criterion,
                                                               optimizer_d, optimizer_g,
                                                               epoch_idx, num_epochs,
                                                               step_count, disc_step_start,
                                                               writer, device)

        # writer.add_scalar('Loss/g_loss', g_loss, epoch_idx)
        # writer.add_scalar('Loss/disc_loss', disc_loss, epoch_idx)
        # writer.add_scalar('Loss/recon_loss', recon_loss, epoch_idx)
        # writer.add_scalar('Loss/codebook_loss', codebook_loss, epoch_idx)
        # writer.add_scalar('Loss/commitment_loss', commitment_loss, epoch_idx)
        # writer.add_scalar('Loss/lpips_loss', lpips_loss, epoch_idx)

        if g_losses < best_g_loss:
            best_g_loss = g_losses
            torch.save(model.state_dict(), f"../model/vqvae_autoencoder_best.pth")
            print(f"\nBest vqvae autoencoder model saved at epoch {epoch_idx + 1} with g_loss: {best_g_loss:.4f}")

        if step_count > disc_step_start and disc_losses < best_disc_loss:
            if disc_losses < best_disc_loss:
                best_disc_loss = disc_losses
                torch.save(discriminator.state_dict(), f"../model/vqvae_discriminator_best.pth")
                print(
                    f"\nBest vqvae discriminator model saved at epoch {epoch_idx + 1} with disc_loss: {best_disc_loss:.4f}")

    # 保存最后的模型
    torch.save(model.state_dict(), f"../model/vqvae_autoencoder_last.pth")
    torch.save(discriminator.state_dict(), f"../model/vqvae_discriminator_last.pth")
    print("Done Training...")


if __name__ == "__main__":
    seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train_image_dir = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural512"
    train_image_dir = r"D:\作业\毕业设计\MuralDH\MuralDH\Mural_seg\train\images"

    im_dataset = VQVAEMuralDataset(train_image_dir, size=256)
    data_loader = DataLoaderX(
        im_dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = VQVAE().to(device)
    lpips_model = lpips.LPIPS(net='vgg').eval().to(device)
    discriminator = Discriminator(im_channels=3).to(device)

    num_epochs = 50

    recon_criterion = nn.MSELoss()
    disc_criterion = nn.BCEWithLogitsLoss()

    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
    optimizer_g = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.5, 0.999))

    disc_step_start = 100

    train_model(data_loader, model, lpips_model, discriminator,
                recon_criterion, disc_criterion,
                optimizer_d, optimizer_g,
                num_epochs, disc_step_start, device)

    # step_count = 0
    #
    # for epoch_idx in range(num_epochs):
    #     for im in tqdm(data_loader):
    #         optimizer_g.zero_grad()
    #         optimizer_d.zero_grad()
    #
    #         step_count += 1
    #         im = im['image'].float().to(device)
    #         # im = im.float().to(device)
    #
    #         ######## Generator #########
    #         model_output = model(im)
    #         output, z, quantize_losses = model_output
    #
    #         recon_loss = recon_criterion(output, im)
    #         g_loss = (
    #                 recon_loss +
    #                 (1 * quantize_losses['codebook_loss']) +
    #                 (0.2 * quantize_losses['commitment_loss'])
    #         )
    #
    #         if step_count > disc_step_start:
    #             disc_fake_pred = discriminator(output)
    #             disc_fake_loss = disc_criterion(
    #                 disc_fake_pred,
    #                 torch.ones(disc_fake_pred.shape, device=disc_fake_pred.device)
    #             )
    #             g_loss += 0.5 * disc_fake_loss
    #
    #         lpips_loss = torch.mean(lpips_model(output, im))
    #         g_loss += lpips_loss
    #         g_loss.backward(retain_graph=True)
    #         optimizer_g.step()
    #         ###########################
    #
    #         ######## Discriminator #########
    #         if step_count > disc_step_start:
    #             fake = output
    #             disc_fake_pred = discriminator(fake.detach())
    #             disc_real_pred = discriminator(im)
    #             disc_fake_loss = disc_criterion(
    #                 disc_fake_pred,
    #                 torch.zeros(disc_fake_pred.shape, device=disc_fake_pred.device)
    #             )
    #             disc_real_loss = disc_criterion(
    #                 disc_real_pred,
    #                 torch.ones(disc_real_pred.shape, device=disc_real_pred.device)
    #             )
    #             disc_loss = 0.5 * (disc_fake_loss + disc_real_loss) / 2
    #             disc_loss.backward()
    #             optimizer_d.step()
    #         #############################
    #
    #     torch.save(model.state_dict(), f"vqvqe_autoencoder.pth")
    #     torch.save(discriminator.state_dict(), f"vqvqe_discriminator.pth")
    # print("Done Training...")
