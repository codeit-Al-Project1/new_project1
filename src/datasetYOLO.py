import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes
from utils import get_transforms

class YOLODataset(Dataset):
    def __init__(self, img_dir, ann_dir, transforms=None, mode='train'):
        """
        YOLO 데이터셋 클래스 (학습 및 검증에 맞게 처리)
        :param img_dir: 이미지 경로
        :param ann_dir: 어노테이션 경로
        :param transforms: 데이터 증강 함수
        :param mode: 'train', 'val', 'test' 중 하나
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.mode = mode

        # 이미지와 어노테이션 파일 로드
        self.images = sorted(os.listdir(img_dir))
        self.annotations = sorted(os.listdir(ann_dir))

        # 파일 매칭
        img_basename = set(os.path.splitext(f)[0] for f in self.images)
        ann_basename = set(os.path.splitext(f)[0] for f in self.annotations)

        # 공통 이름만 선택
        common_name = img_basename & ann_basename
        self.images = [f"{name}.png" for name in common_name]
        self.annotations = [f"{name}.json" for name in common_name]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 이미지와 어노테이션을 반환합니다.
        """
        img_path = os.path.join(self.img_dir, self.images[idx])
        ann_path = os.path.join(self.ann_dir, self.annotations[idx])

        # 이미지 로드
        img = Image.open(img_path).convert("RGB")

        # 어노테이션 로드
        with open(ann_path, 'r') as f:
            ann = json.load(f)
        
        bboxes = [obj["bbox"] for obj in ann["annotations"]]
        labels = [obj["category_id"] for obj in ann["annotations"]]

        bboxes_tensor = BoundingBoxes(
            torch.tensor(bboxes, dtype=torch.float32),
            format="XYWH",
            canvas_size=(img.height, img.width)
        )

        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        # 데이터 증강 (if exists)
        if self.transforms:
            img, bboxes_tensor = self.transforms(img, bboxes_tensor)

        targets = {
            "boxes": bboxes_tensor,
            "labels": labels_tensor
        }

        return img, targets
