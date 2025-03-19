import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
import torch
from torch.utils.data import DataLoader, random_split
import argparse 
import sys


# 데이터 증강을 위한 transform 정의
def get_transforms(mode='train'):
    """
    
    """
    if mode == 'train':
        return T.Compose([
            T.ToImage(), # PIL → TVImage 자동 변환
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.ToDtype(torch.float32, scale=True)  # 0 ~ 1 스케일링
        ])
    elif mode == "val" or mode == "test":
        return T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose either 'train' or 'test'.")

    
    
# 
class PillDataset(Dataset):
    def __init__(self, image_dir, ann_dir=None, mode='train', transform=None):
        """
        Args:
            image_dir (str): 루트 경로 (ex: /data/train_images)
            ann_dir (str): 어노테이션 경로 (ex: /data/train_annots_modify)
            mode (str): 'train' / 'val' / 'test'
            transforms (callable, optional): 이미지 변환 함수
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
        if self.mode in ['train', 'val']:
            return len(self.annots)  # ✅ self.annots 사용 (수정)
        else:
            return len(self.images)  # ✅ 테스트 모드에서는 이미지 개수 기준

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


TRAIN_ROOT = "data/train/train_images"
TRAIN_ANN_DIR = "data/train/train_annotations"
TEST_ROOT = "data/test_images"

def get_loader(img_dir, ann_dir, batch_size=16, mode="train", val_ratio=0.2, debug=False, seed=42):
    transforms = get_transforms(mode=mode)
    dataset = PillDataset(image_dir=img_dir, ann_dir=ann_dir, mode=mode, transform=transforms)

    # 랜덤시드
    generator = torch.Generator().manual_seed(seed)   # 시드 고정

###########################################################################################
# 이후 수정 250318(화) 마무리

    train_size = int((1 - val_ratio) * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
        
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

# def get_test_loader(batch_size=1, debug=False):
#     test_dataset = TestDataset(TEST_ROOT, transform=None)

#     test_loader = DataLoader(
#         test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=test_dataset.collate_fn
#     )

#     if debug:
#         # 데이터 로더 사용 예시
#         for images, filename in test_loader:
#             print(f"Batch of images shape: {images.shape}")
#             print(f"Batch of filenames: {filename}")
#             break

#     return test_loader



# # 데이터 로더 생성
# train_loader, val_loader = get_train_val_loader(batch_size, val_ratio)
# test_loader = get_test_loader(batch_size)

if __name__ == "__main__":
    # 기본 설정
    img_dir = TRAIN_ROOT  # 훈련용 이미지 경로
    ann_dir = TRAIN_ANN_DIR  # 훈련용 어노테이션 경로
    
    batch_size = 4  # 작은 배치 크기로 테스트
    mode = "train"
    
    # 데이터 로더 생성
    train_loader, val_loader = get_loader(img_dir, ann_dir, batch_size=batch_size, mode=mode, debug=True)

    # 데이터 로딩 테스트 (한 번만 실행)
    for images, targets in train_loader:
        print(f"첫 번째 배치 - 이미지 개수: {len(images)}")
        print(f"첫 번째 배치 - 첫 번째 이미지 크기: {images[0].shape}")
        print(f"첫 번째 배치 - 타겟 개수: {len(targets)}")
        break  # 한 번만 실행 후 종료