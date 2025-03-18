## 데이터 증강을 위한 transform 정의
import torch
from torchvision import transforms


def get_transforms():
    return transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=torch.float)
    ])