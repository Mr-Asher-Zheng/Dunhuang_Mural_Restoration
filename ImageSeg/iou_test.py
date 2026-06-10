import argparse
import sys
from tqdm import tqdm
import time
# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from image_seg_utils import *
from SAM_Adapter_Model import *


def test_model(model, val_loader, criterion, device):
    model.eval()
    val_losses = 0.0
    total_iou = 0.0
    total_images = 0

    val_loader_tqdm = tqdm(val_loader, desc="Valid", leave=False)
    with torch.no_grad():
        for i, batch in enumerate(val_loader_tqdm):
            # images = batch["image"].squeeze(0).to(device)
            # masks = batch["mask"].squeeze(0).to(device)

            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # print(images.shape, masks.shape)
            # torch.Size([1, 3, 512, 512])
            # torch.Size([1, 1, 512, 512])
            # images = images.unsqueeze(0)
            preds = model(images)
            # print(preds)
            # time.sleep(60)
            #
            # preds = model.generate(images)
            # print(preds)
            # print("=========")
            # mask = preds[0]["segmentation"]
            # plt.imshow(mask)
            # plt.show()
            # print("=========")
            # time.sleep(60)
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

    val_loss = val_losses / len(val_loader)

    print('Loss/Val', val_loss)
    print('IOU/Val', iou)


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_dir", type=str, default=r"./test_images",
                        help="path of test images for segmentation prediction")
    parser.add_argument("--test_mask_dir", type=str, default=r"./test_masks",
                        help="path of test masks for segmentation evaluation")

    parser.add_argument('--model_type', type=str, default='vit_h',
                        help='Type of SAM model to use (e.g., vit_b, vit_l, vit_h)')
    parser.add_argument('--sam_model_path', type=str, default='./checkpoints/sam_vit_h_4b8939.pth',
                        help='Path to SAM model checkpoint (e.g., sam_vit_b_01ec64.pth, sam_vit_l_0b3195.pth, sam_vit_h_4b8939.pth)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_iou_model.pth',
                        help='Path to the trained model checkpoint for segmentation prediction')

    # "./checkpoints/best_iou_model.pth"
    # "./checkpoints/k7_SAM_Focal_Tversky_Loss.pth"

    parser.add_argument('--image_size', type=int, default=512, help='Input image size for training and validation')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training and validation')

    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')

    if len(sys.argv) == 1:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    # 获取参数
    args = get_test_args()

    # 设置随机种子以确保结果可复现
    seeds(args.seed)

    # 检查当前系统是否有可用的 GPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}, if want to use GPU, please ensure CUDA is properly installed and configured.")

    os.makedirs(args.test_image_dir, exist_ok=True)
    os.makedirs(args.test_mask_dir, exist_ok=True)

    test_dataset = SegMuralDataset(args.test_image_dir, args.test_mask_dir, size=args.image_size)
    test_dataloader = DataLoaderX(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # 查看dataloader数量
    print(len(test_dataloader))

    model = SAMAdapterModel(model_type=args.model_type, checkpoint=args.sam_model_path)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model = model.to(device)
    print(model)
    print("model loaded successfully!")

    # =====================================================================
    # model = sam_model_registry[args.model_type](checkpoint=args.sam_model_path)
    # print(model)
    # model = model.to(device)
    # mask_generator = SamAutomaticMaskGenerator(model)
    # =====================================================================

    criterion = FocalTversky_BCE_Loss()
    test_model(model, test_dataloader, criterion, device)
