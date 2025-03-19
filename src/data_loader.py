import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
import argparse 
import sys


# 데이터 증강을 위한 transform 정의
def get_transforms(mode='train'):
    """
    데이터 증강 및 전처리 함수를 반환합니다.

    Args:
        mode (str): 'train', 'val', 'test' 중 하나

    Returns:
        torchvision.transforms.v2.Compose: 변환 함수
    """
    ################################################################
    # 리사이즈 크기 설정해야함
    ####################################
    if mode == 'train':
        return T.Compose([
            T.ToImage(), # PIL → TVImage 자동 변환
            T.RandomHorizontalFlip(),   # 수평 뒤집기
            T.RandomVerticalFlip(),     # 수직 뒤집기
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),   # 밝기 조절
            T.ToDtype(torch.float32, scale=True)  # 0 ~ 1 스케일링
        ])
    elif mode == "val" or mode == "test":
        return T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose either 'train', 'val', or 'test'.")

    
    
# 
class PillDataset(Dataset):
    def __init__(self, image_dir, ann_dir=None, mode='train', transform=None):
        """
        알약 이미지 데이터셋 클래스

        Args:
            image_dir (str): 이미지 파일들이 저장된 경로
            ann_dir (str, optional): 어노테이션 파일이 저장된 경로 (train/val 모드에서 필요)
            mode (str): 'train', 'val', 'test' 중 하나
            transform (callable, optional): 이미지 변환 함수
        
        Raises:
            AssertionError: Train/Val 모드에서 `ann_dir`이 필요함
        """
        self.img_dir = image_dir
        self.ann_dir = ann_dir
        self.mode = mode
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

        # train, val/ test 분기
        if self.mode in ['train', 'val']:
            assert ann_dir is not None, "Train/Val 모드에서는 ann_dir가 필요합니다."
            self.annots = sorted(os.listdir(ann_dir))
        else:
            self.annots = None

    def __getitem__(self, idx):
        """
        Returns:
            train/val 모드: (img(TVImage), bboxes_tensor(BoundingBoxes), labels_tensor(torch.Tensor))
            test 모드: (img(TVImage), img_file(str))
        """

        # 이미지 인덱싱
        img_file = self.images[idx]
        img_path = os.path.join(self.img_dir, img_file)

        # 이미지 파일 확인
        try:
            img = Image.open(img_path).convert("RGB")
        except FileNotFoundError as e:
            print(f"Error loading image file: {img_path}, {e}")
            return None  # 오류 발생 시 None 반환

        # 학습과 검증 분기
        if self.mode in ['train', 'val']:
            # 어노테이션 파일 인덱싱
            ann_file = self.annots[idx]
            ann_path = os.path.join(self.ann_dir, ann_file)

            # 어노테이션 파일 확인
            try:
                with open(ann_path, 'r', encoding='utf-8') as f:
                    ann = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading annotation file: {ann_path}, {e}")
                return None  # 오류 발생 시 None 반환
            
            # bbox와 labels 추출
            bboxes = [obj["bbox"] for obj in ann["annotations"]]
            labels = [obj["category_id"] for obj in ann["annotations"]]

            # 텐서로 변환 tv_tensor
            bboxes_tensor = BoundingBoxes(
                torch.tensor(bboxes, dtype=torch.float32),
                format="XYWH",
                canvas_size=(img.height, img.width)
            )
            labels_tensor = torch.tensor(labels, dtype=torch.int64)

            if self.transform:
                img, bboxes_tensor, labels_tensor = self.transform(img, bboxes_tensor, labels_tensor)

            return img, bboxes_tensor, labels_tensor

        # 시험 분기 
        else:
            if self.transform:
                img = self.transform(img)
            return img, img_file


    def __len__(self):
        return len(self.annots)

    def get_img_info(self, idx):
        ann_file = self.annots[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                ann = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading annotation file: {ann_path}, {e}")
            return None
        return {"file_name": ann['images'][0]['file_name'], "height": ann['images'][0]['height'], "width": ann['images'][0]['width']}

    def get_ann_info(self, idx):
        ann_file = self.annots[idx]
        ann_path = os.path.join(self.ann_dir, ann_file)
        try:
            with open(ann_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading annotation file: {ann_path}, {e}")
            return None


TRAIN_ROOT = "data/train_images"
TRAIN_ANN_DIR = "data/train_annots_modify"
TEST_ROOT = "data/test_images"

def get_loader(img_dir, ann_dir, batch_size=16, mode="train", val_ratio=0.2, debug=False, seed=42):
    
    # 트랜스폼
    transforms = get_transforms(mode=mode)

    # 데이터셋
    dataset = PillDataset(image_dir=img_dir, ann_dir=ann_dir, mode=mode, transform=transforms)

    # collator 정의
    def collator(batch):
        return tuple(zip(*batch))

    # 랜덤시드 설정
    generator = torch.Generator().manual_seed(seed)   # 시드 고정

    # 훈련/검증의 경우
    if mode == 'train' or mode == 'val':
        # 훈련/ 검증 분리하기
        train_size = int((1 - val_ratio) * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    

        if mode == 'train':
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collator
            )

            # 배치 사이즈 예시
            if debug:
                for images, targets in train_loader:
                    print(f"Batch of images shape: {images.shape}")
                    print(f"Batch of targets: {targets}")
                    break

            return train_loader
        
        elif mode == 'val':
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator
            )

            # 배치 사이즈 예시
            if debug:
                for images, targets in val_loader:
                    print(f"Batch of images shape: {images.shape}")
                    print(f"Batch of targets: {targets}")
                    break

            return val_loader
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose either 'train', 'val', or 'test'.")

    # 시험의 경우
    elif mode == 'test':
        test_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator
        )

        # 배치 사이즈 예시
        if debug:
            for images, targets in test_loader:
                print(f"Batch of images shape: {images.shape}")
                print(f"Batch of targets: {targets}")
                break

        return test_loader
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose either 'train', 'val', or 'test'.")