import os
import torch
import torchvision
import json
from glob import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision import transforms
from PIL import Image

# 데이터셋 클래스 정의
class PillDataset(Dataset):
    def __init__(self, json_dir, img_dir, transforms=None):
        self.json_files = glob(os.path.join(json_dir, "*.json"))  
        self.img_dir = img_dir
        self.transforms = transforms
        self.data = self._load_json_files()

    def _load_json_files(self):
        """JSON 폴더 내 모든 파일을 읽어 통합"""
        all_data = {"images": [], "annotations": [], "categories": []}
        for json_file in self.json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            all_data["images"].extend(data["images"])
            all_data["annotations"].extend(data["annotations"])
            all_data["categories"].extend(data["categories"])
        return all_data

    def __len__(self):
        return len(self.data["images"])

    def __getitem__(self, idx):
        img_info = self.data["images"][idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # 해당 이미지에 대한 어노테이션 찾기
        image_id = img_info["id"]
        boxes, labels, areas = [], [], []

        for ann in self.data["annotations"]:
            if ann["image_id"] == image_id:
                x_min, y_min, width, height = ann["bbox"]
                boxes.append([x_min, y_min, x_min + width, y_min + height])
                labels.append(ann["category_id"])
                areas.append(ann["area"])

        # 변환 적용 (PyTorch 모델 입력 형태에 맞추기)
        if self.transforms:
            image = self.transforms(image)

        # Faster R-CNN에 맞는 타겟 데이터 구성
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),  # iscrowd는 0으로 설정
        }

        return image, target


# DataLoader를 위한 collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))


# DataLoader 생성 함수
def get_dataloader(json_dir, img_dir, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = PillDataset(json_dir, img_dir, transforms=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def split_dataloader(dataloader, val_split=0.2):
    dataset_size = len(dataloader.dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Dataset을 학습용과 검증용으로 분할
    train_dataset, val_dataset = random_split(dataloader.dataset, [train_size, val_size])

    # DataLoader로 변환
    train_loader = DataLoader(train_dataset, batch_size=dataloader.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=dataloader.batch_size, shuffle=True, collate_fn=collate_fn)

    return train_loader, val_loader