import argparse
import os
import sys
import torch
from tqdm import tqdm
from PIL import Image

from image_seg_utils import *
from SAM_Adapter_Model import *


class SegPred(Dataset):
    def __init__(self, image_dir, size=512):
        self.image_list = sorted([os.path.join(image_dir, f)
                                  for f in os.listdir(image_dir) if f.endswith(('png', 'jpg'))])
        self.size = size
        # self.image_transform = transforms.Compose([
        #     transforms.Resize((size, size)),
        #     transforms.ToTensor(),
        # ])

        self.transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # image = Image.open(self.image_list[idx]).convert("RGB")
        # image = self.image_transform(image)

        image = cv2.imread(self.image_list[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image)
        image = transformed['image']

        path = self.image_list[idx]

        return {"image": image, "filename": os.path.basename(path)}


def Image_seg_detect(model, test_dataloader, save_dir, device):
    model.eval()

    with torch.no_grad():
        test_loader_tqdm = tqdm(test_dataloader, desc="Segmentation Prediction")
        for step, batch in enumerate(test_loader_tqdm):
            images = batch["image"].to(device)
            filenames = batch["filename"]

            # print(images.shape)

            # [8, 1, 512, 512]
            pred_masks = model(images)

            # [8, 512, 512, 1]
            pred_masks = torch.clamp(pred_masks, -1, 1).detach().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()

            # [8, 512, 512]
            if pred_masks.shape[-1] == 1:
                pred_masks = pred_masks.squeeze(-1)

            for i in range(pred_masks.shape[0]):
                pred_mask = pred_masks[i]
                pred_mask = (pred_mask * 255).astype(np.uint8)

                pred_mask = Image.fromarray(pred_mask, mode='L')

                save_path = os.path.join(save_dir, filenames[i])
                pred_mask.save(save_path)

            # plt.figure(figsize=(15, 5))
            # for i in range(pred_masks.shape[0]):
            #     # 保存预测的掩码图像
            #     save_path = os.path.join(save_dir, filenames[i])
            #     plt.imsave(save_path, pred_masks[i], cmap='gray')
            #     # 显示预测的掩码图像
            #     plt.subplot(1, pred_masks.shape[0], i + 1)
            #     plt.imshow(pred_masks[i], cmap='gray')
            #     plt.axis('off')
            # plt.tight_layout()
            # plt.show()

    print("Segmentation prediction Done!")


def get_pred_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_dir", type=str, default=r"./test_images",
                        help="path of test images for segmentation prediction")
    parser.add_argument("--save_image_dir", type=str, default=r"./test_seg_pred",
                        help="path to save predicted segmentation masks")

    # save_image_dir = r"../../MuralDH/MuralDH/seg_pred"
    # test_image_dir = r"../../MuralDH/MuralDH/Mural512"

    # save_image_dir = r"../../Dunhuang_Grottoes_Painting/train/seg_pred"
    # test_image_dir = r"../../Dunhuang_Grottoes_Painting/train/train_GT_cropped"

    # save_image_dir = r"../../Dunhuang_Grottoes_Painting/test/seg_pred"
    # test_image_dir = r"../../Dunhuang_Grottoes_Painting/test/test_GT_cropped"

    # save_image_dir = r"../../DhMurals-inpainting-dataset/train/seg_pred"
    # test_image_dir = r"../../DhMurals-inpainting-dataset/train/images_cropped"

    # save_image_dir = r"../../DhMurals-inpainting-dataset/val/seg_pred"
    # test_image_dir = r"../../DhMurals-inpainting-dataset/val/images_cropped"

    # save_image_dir = r"../../DhMurals-inpainting-dataset/test/seg_pred"
    # test_image_dir = r"../../DhMurals-inpainting-dataset/test/images_cropped"

    # save_image_dir = r"../../Dunhuang_Faces/Dunhuang_Faces-20K/seg_pred"
    # test_image_dir = r"../../Dunhuang_Faces/Dunhuang_Faces-20K/face_imgs_cropped"

    # save_image_dir = r"../../DH/samples/seg/finetune_SAM-Adapter"
    # test_image_dir = r"../../DH/samples/seg/images"

    # save_image_dir = r"../ImageRestoration/samples/Evaluation/LandscapeTest/mask"
    # test_image_dir = r"../ImageRestoration/samples/Evaluation/LandscapeTest/gt"

    # save_image_dir = r"../../DH/kaggle/mask_origin"
    # test_image_dir = r"../../DH/kaggle/gt"

    parser.add_argument('--model_type', type=str, default='vit_h',
                        help='Type of SAM model to use (e.g., vit_b, vit_l, vit_h)')
    parser.add_argument('--sam_model_path', type=str, default='./checkpoints/sam_vit_h_4b8939.pth',
                        help='Path to SAM model checkpoint (e.g., sam_vit_b_01ec64.pth, sam_vit_l_0b3195.pth, sam_vit_h_4b8939.pth)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_iou_model.pth',
                        help='Path to the trained model checkpoint for segmentation prediction')

    # "./checkpoints/best_iou_model.pth"
    # "./checkpoints/k7_SAM_Focal_Tversky_Loss.pth"

    parser.add_argument('--image_size', type=int, default=512, help='Input image size for training and validation')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training and validation')

    parser.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training (e.g., "cuda" or "cpu")')

    if len(sys.argv) == 1:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    # 获取参数
    args = get_pred_args()

    # 设置随机种子以确保结果可复现
    seeds(args.seed)

    # 检查当前系统是否有可用的 GPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}, if want to use GPU, please ensure CUDA is properly installed and configured.")

    os.makedirs(args.test_image_dir, exist_ok=True)
    os.makedirs(args.save_image_dir, exist_ok=True)

    test_dataset = SegPred(args.test_image_dir, size=args.image_size)
    test_dataloader = DataLoaderX(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = SAMAdapterModel(model_type=args.model_type, checkpoint=args.sam_model_path)
    model.load_state_dict(torch.load(args.checkpoint, weights_only=True))

    model = model.to(device)
    print("model loaded successfully!")

    Image_seg_detect(model, test_dataloader, args.save_image_dir, device)
