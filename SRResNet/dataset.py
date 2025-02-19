import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

class SuperResolutionDataset(Dataset):
    def __init__(self, img100_dir, img30_dir, transform=None):
        self.img100_dir = img100_dir
        self.img30_dir = img30_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(img100_dir))  # 確保名稱一致

        # 你可以將以下部分直接放在這裡來確保大小正確
        self.transform_hr = transforms.Compose([
            transforms.Resize((90, 90)),  # 確保 HR 影像為 100x100
            transforms.ToTensor()
        ])
        
        self.transform_lr = transforms.Compose([
            transforms.Resize((30, 30)),  # 確保 LR 影像為 30x30
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        
        img100_path = os.path.join(self.img100_dir, img_name)
        img30_path = os.path.join(self.img30_dir, img_name)

        img100 = Image.open(img100_path).convert("RGB")
        img30 = Image.open(img30_path).convert("RGB")

        if self.transform:
            img100 = self.transform(img100)
            img30 = self.transform(img30)

        return img30, img100  # LR (輸入), HR (標籤)
