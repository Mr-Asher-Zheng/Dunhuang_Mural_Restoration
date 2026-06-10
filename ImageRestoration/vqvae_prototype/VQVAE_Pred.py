import math
import os
import time

from matplotlib import pyplot as plt
from torchvision import transforms
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
from accelerate import Accelerator


def seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class VQVAEPred(Dataset):
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

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx]).convert("RGB")
        base = self.base_transform(image)
        image = self.image_transform(base)

        path = self.image_list[idx]

        return {"image": image, "filename": os.path.basename(path)}


def VQVAE_pred(model, test_dataloader, save_dir, device):
    model.eval()

    with torch.no_grad():
        test_loader_tqdm = tqdm(test_dataloader, desc="VQVAE Prediction")
        for step, batch in enumerate(test_loader_tqdm):
            images = batch["image"].to(device)
            filenames = batch["filename"]

            model_output = model(images)
            # output：[8, 3, 256, 256]
            # output, z, quantize_losses = model_output
            output, z, quantize_loss, perplexity = model_output

            # (8, 256, 256, 3)
            output = output.detach().cpu().permute(0, 2, 3, 1).numpy()

            # 反归一化
            mean = np.array([0.5, 0.5, 0.5])
            std = np.array([0.5, 0.5, 0.5])

            # 保存图片
            for i in range(output.shape[0]):
                img = output[i]
                img = (img * std) + mean

                img = np.clip(img, 0, 1)

                # 转成 [0,255] uint8
                img = (img * 255).astype(np.uint8)

                img_pil = Image.fromarray(img)
                save_path = os.path.join(save_dir, filenames[i])
                img_pil.save(save_path)

    print("VQVAE Prediction Done!")


if __name__ == "__main__":
    seeds(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = r"vqvae_pred_new_3"
    test_image_dir = r"../../../DH/image_mask/val/images"

    os.makedirs(save_dir, exist_ok=True)

    test_dataset = VQVAEPred(test_image_dir, size=512)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    model = VQVAE()
    # model.load_state_dict(torch.load("./model/vqvae_autoencoder_best.pth"))
    model.load_state_dict(torch.load("vqvae_pred_new_3/vqvae_autoencoder_best.pth"))

    model = model.to(device)

    VQVAE_pred(model, test_dataloader, save_dir, device)
