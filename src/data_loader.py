import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import torch
from torch.utils.data import DataLoader, random_split
from src.transforms import get_transforms
import argparse 
import sys

class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.img_files = sorted(os.listdir(root))

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.root, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image file: {img_path}, {e}")
            return None  # 오류 발생 시 None 반환

        transform = transforms.ToTensor()
        img = transform(img)

        return img, img_file  # 이미지와 파일 이름 반환

    def __len__(self):
        return len(self.img_files)

    def collate_fn(self, batch):
        batch = [data for data in batch if data is not None]  # None 데이터 제거
        images, file_names = zip(*batch)
        images = torch.stack(images, 0)
        return images, file_names
    
    
class COCODataset(Dataset):
    def __init__(self, root, ann_dir, transform=None):
        self.root = root
        self.ann_dir = ann_dir
        self.transform = transform
        self.ann_files = sorted(os.listdir(ann_dir))  ##확인

    def __getitem__(self, idx):
        ann_file = self.ann_files[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)

        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading annotation file: {ann_path}, {e}")
            return None  # 오류 발생 시 None 반환

        img_file = ann['images'][0]['file_name']
        img_path = os.path.join(self.root, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image file: {img_path}, {e}")
            return None  # 오류 발생 시 None 반환

        if self.transform:
            img = self.transform(img)

        target = {}
        target['bbox'] = torch.tensor([ann['annotations'][0]['bbox']], dtype=torch.float32)
        target['labels'] = torch.tensor([ann['categories'][0]['id']], dtype=torch.int64)
        target['image_id'] = torch.tensor([ann['annotations'][0]['image_id']], dtype=torch.int64)

        return img, target

    def __len__(self):
        return len(self.ann_files)

    def get_img_info(self, idx):
        ann_file = self.ann_files[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading annotation file: {ann_path}, {e}")
            return None
        return {"file_name": ann['images'][0]['file_name'], "height": ann['images'][0]['height'], "width": ann['images'][0]['width']}

    def get_ann_info(self, idx):
        ann_file = self.ann_files[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading annotation file: {ann_path}, {e}")
            return None

TRAIN_ROOT = "data/train/train_images"
TRAIN_ANN_DIR = "data/train/train_annotations"
TEST_ROOT = "data/test/test_images"

def get_train_val_loader(batch_size, val_ratio=0.2, debug=False):
    transform = get_transforms()
    full_dataset = COCODataset(TRAIN_ROOT, TRAIN_ANN_DIR, transform=transform)
    train_size = int((1 - val_ratio) * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
        
    def collator(batch):
        batch = [data for data in batch if data is not None]
        images, targets = zip(*batch)
        images = torch.stack(images, 0)
        return images, targets

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collator
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator
    )

    if debug:
        # 데이터 로더 사용 예시
        for images, targets in train_loader:
            print(f"Batch of images shape: {images.shape}")
            print(f"Batch of targets: {targets}")
            break

    return train_loader, val_loader

def get_test_loader(batch_size=1, debug=False):
    test_dataset = TestDataset(TEST_ROOT, transform=None)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=test_dataset.collate_fn
    )

    if debug:
        # 데이터 로더 사용 예시
        for images, filename in test_loader:
            print(f"Batch of images shape: {images.shape}")
            print(f"Batch of filenames: {filename}")
            break

    return test_loader



# # 데이터 로더 생성
# train_loader, val_loader = get_train_val_loader(batch_size, val_ratio)
# test_loader = get_test_loader(batch_size)