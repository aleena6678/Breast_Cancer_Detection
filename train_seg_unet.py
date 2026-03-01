import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# ====== PATHS & CONFIG ======
DATA_ROOT = "data"
IMAGE_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR  = os.path.join(DATA_ROOT, "masks")

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_OUT = "breast_seg_unet.pth"

# ====== DATASET ======

class BUSSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=256):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.img_size = img_size

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),  # [0,1], CxHxW
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))

        # Read mask; if missing, treat as all background
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path)
            # lesion: any non-black pixel in mask
            lesion = (mask[:, :, 0] > 10) | (mask[:, :, 1] > 10) | (mask[:, :, 2] > 10)
            mask = lesion.astype(np.uint8) * 255
        else:
            mask = np.zeros_like(img, dtype=np.uint8)

        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        img = self.img_transform(img)          # [1, H, W]
        mask = torch.from_numpy(mask).float() / 255.0  # [H, W] in [0,1]
        mask = mask.unsqueeze(0)              # [1, H, W]

        return img, mask


# ====== SIMPLE U-NET ======

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        bn = self.bottleneck(p3)

        u3 = self.up3(bn)
        u3 = torch.cat([u3, c3], dim=1)
        c3 = self.conv3(u3)

        u2 = self.up2(c3)
        u2 = torch.cat([u2, c2], dim=1)
        c2 = self.conv2(u2)

        u1 = self.up1(c2)
        u1 = torch.cat([u1, c1], dim=1)
        c1 = self.conv1(u1)

        out = self.out_conv(c1)   # [B, 1, H, W]
        return out


# ====== TRAINING LOOP ======

def main():
    dataset = BUSSegDataset(IMAGE_DIR, MASK_DIR, img_size=IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = UNetSmall(in_channels=1, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Using {DEVICE}, training samples: {len(dataset)}")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs = imgs.to(DEVICE)   # [B,1,H,W]
            masks = masks.to(DEVICE) # [B,1,H,W]

            optimizer.zero_grad()
            logits = model(imgs)     # [B,1,H,W]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Segmentation model saved to {MODEL_OUT}")


if __name__ == "__main__":
    main()
