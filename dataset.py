import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class GeoDataset(Dataset):
    def __init__(self, img_dir, csv_path):
        self.img_dir = img_dir
        # Load CSV containing latitude and longitude for each image
        self.coords = pd.read_csv(csv_path)

        # Standard ImageNet normalization — required since DINOv2 was pretrained on ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # Images are named by index (e.g. 0.png, 1.png, ...)
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        lat = self.coords.loc[idx, "latitude"]
        lon = self.coords.loc[idx, "longitude"]

        # Return image tensor and (lat, lon) as the regression target
        return img, torch.tensor([lat, lon], dtype=torch.float32)