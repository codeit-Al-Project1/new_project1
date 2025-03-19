#################
# 실행방법?
# python src/data_loader.py --mode train --batch_size 4 --debug
# 
# 카테고리 매핑, 리사이즈(나중에), 데이터셋 리턴값

import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes, Image as TVImage
import argparse 
import sys

####################################################################################################
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

####################################################################################################
# json파일에서 카테고리 매핑을 만드는 함수
def get_category_mapping(ann_dir):
    id_to_name = {}
    #  id -> name 매핑 수집
    for file in os.listdir(ann_dir):
        with open(os.path.join(ann_dir, file), 'r', encoding='utf-8') as f:
            ann = json.load(f)
            for cat in ann['categories']:
                id_to_name[cat['id']] = cat['name']

    # 이름 기준 정렬
    sorted_names = sorted(set(id_to_name.values())) # 중복 제거 후 정렬

    # 1부터 인덱싱, 0은 배경, 마지막 숫자는 No Class
    name_to_idx = {'Background': 0}
    for idx, name in enumerate(sorted_names, start=1):
        name_to_idx[name] = idx
    name_to_idx['No Class'] =  len(name_to_idx)

    # 역 매핑
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}
    return name_to_idx, idx_to_name

####################################################################################################
# 데이터셋
class PillDataset(Dataset):
    def __init__(self, image_dir, ann_dir=None, mode='train', category_mapping=None, transform=None, debug=False):
        """
        알약 이미지 데이터셋 클래스

        Args:
            image_dir (str): 이미지 파일들이 저장된 경로
            ann_dir (str, optional): 어노테이션 파일이 저장된 경로 (train/val 모드에서 필요)
            mode (str): 'train', 'val', 'test' 중 하나
            category_mapping ()
            transform (callable, optional): 이미지 변환 함수
            debug (bool): 디버깅 모드 (이미지와 어노테이션 중 하나가 없는 경우 출력)
        
        Raises:
            AssertionError: Train/Val 모드에서 `ann_dir`이 필요함
        """
        self.img_dir = image_dir
        self.ann_dir = ann_dir
        self.mode = mode
        self.transform = transform

        # 카테고리 이름 <-> 아이디 매핑
        self.category_mapping = category_mapping

        self.images = sorted(os.listdir(image_dir))

        # train, val/ test 분기
        if self.mode in ['train', 'val']:
            assert ann_dir is not None, "Train/Val 모드에서는 ann_dir가 필요합니다."
            self.annots = sorted(os.listdir(ann_dir))

            # 이미지-어노테이션 불일치 필터링
            # 예시: K-001900-010224-016551-031705_0_2_0_2_70_000_200.png
            img_basename = set(os.path.splitext(f)[0] for f in self.images)
            # 예시: K-001900-010224-016551-031705_0_2_0_2_70_000_200.png.json
            ann_basename = set(os.path.splitext(os.path.splitext(f)[0])[0] for f in self.annots)    # 두 번 적용
            
            # 공통이름 및 차이점
            common_name = img_basename & ann_basename 
            missing_img = ann_basename - img_basename
            missing_ann = img_basename - ann_basename

            # 디버깅(차이점 출력)
            if debug:
                print(f"총 {len(common_name)}개의 이미지-어노테이션 파일 처리")
                if missing_img:
                    print(f"[WARNING] 어노테이션은 있지만 이미지가 없는 파일 목록(총 {len(missing_img)}개개):")
                    for name in sorted(missing_img):
                        print(f" - {name}")
                if missing_ann:
                    print(f"[WARNING] 이미지는 있지만 어노테이션이 없는 파일 목록(총 {len(missing_ann)}개):")
                    for name in sorted(missing_ann):
                        print(f" - {name}")
            
            # 공통 파일만 필터링
            self.images = [f"{name}.png" for name in common_name]
            self.annots = [f"{name}.png.json" for name in common_name]

        else:
            self.annots = None

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 이미지 및 어노테이션 데이터를 가져옵니다.

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
            raise FileNotFoundError(f"Error loading image file: {img_path}, {e}")

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
                raise RuntimeError(f"Error loading annotation file: {ann_path}, {e}")
            
            # bbox와 labels 추출
            bboxes = [obj["bbox"] for obj in ann["annotations"]]
            pill_names = [obj["name"] for obj in ann["categories"]]
            labels = [self.category_mapping[cat_name] for cat_name in pill_names]  # 카테고리 매핑으로 변환
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas = [obj["area"] for obj in ann["annotations"]]
            image_id = ann["images"][0]["id"]

            # 텐서로 변환 tv_tensor
            bboxes_tensor = BoundingBoxes(
                torch.tensor(bboxes, dtype=torch.float32),
                format="XYWH",
                canvas_size=(img.height, img.width)
            )
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
            image_id_tensor = torch.tensor(image_id, dtype=torch.int64)
            iscrowd_tensor = torch.zeros((len(bboxes_tensor),), dtype=torch.int64)
            orig_size_tensor = torch.tensor([img.height, img.width], dtype=torch.int64)

            # 트랜스폼 적용
            if self.transform:
                img, bboxes_tensor = self.transform(img, bboxes_tensor)

            targets = {
                'boxes': bboxes_tensor,
                'labels': labels_tensor,
                'image_id': image_id_tensor,
                'area': areas_tensor,
                'is_crowd': iscrowd_tensor,
                'orig_size': orig_size_tensor,
                'pill_names': pill_names
            }
            

            # 이미지와 타겟
            return img, targets

        # 시험 분기 
        else:
            if self.transform:
                img = self.transform(img)
                
            # 이미지, _ for in batch
            return img, img_file


    def __len__(self):
        return len(self.images)

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
        

####################################################################################################
# 데이터 로더 함수
def get_loader(img_dir, ann_dir, batch_size=16, mode="train", val_ratio=0.2, debug=False, seed=42):
    """
    데이터 로더를 반환하는 함수

    Args:
        img_dir (str): 이미지 폴더 경로
        ann_dir (str): 어노테이션 폴더 경로
        batch_size (int): 배치 크기
        mode (str): 'train', 'val', 'test' 중 하나
        val_ratio (float): 검증 데이터셋 비율
        debug (bool): 디버깅 모드 (True일 경우 배치 데이터 출력)
        seed (int): 랜덤 시드

    Returns:
        torch.utils.data.DataLoader: 해당 모드의 데이터 로더
    """    
    # 트랜스폼
    transforms = get_transforms(mode=mode)

    # 카테고리 매핑
    name_to_idx, idx_to_name = get_category_mapping(ann_dir=ann_dir)
    # print(name_to_idx)
    # print('')
    # print(idx_to_name)

    # 데이터셋
    dataset = PillDataset(image_dir=img_dir, ann_dir=ann_dir, mode=mode, category_mapping=name_to_idx, transform=transforms, debug=debug)

    # collator 정의
    def collator(batch):
        batch = [b for b in batch if b is not None] # None 제거
        return tuple(zip(*batch)) if batch else None

    # 랜덤시드 설정
    generator = torch.Generator().manual_seed(seed) # 시드 고정

    # 훈련/검증의 경우
    if mode == 'train' or mode == 'val':
        # 훈련/ 검증 분리하기
        train_size = int((1 - val_ratio) * len(dataset))
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        loader = DataLoader(
            train_dataset if mode == 'train' else val_dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            drop_last=True,
            collate_fn=collator
        )

        if debug:
            print(f"\n{mode}_loader의 데이터 크기: {len(loader)}")
            for batch in loader:
                if batch is not None:
                    images, targets = batch
                    print(f"Batch size: {len(images)}")

                    # 첫 번째 샘플의 image 정보 출력
                    print(f"\n[첫 번째 샘플 이미지 정보]")
                    print(f" - Image size: {images[0].shape}")

                    # 첫 번째 샘플의 target 정보 출력
                    sample_target = targets[0]
                    print(f"[첫 번째 샘플 타겟 정보]")
                    print(f" - image_id: {sample_target['image_id'].item()}")
                    print(f" - boxes shape: {sample_target['boxes'].shape}")
                    print(f" - labels: {sample_target['labels'].tolist()}")
                    print(f" - areas: {sample_target['area'].tolist()}")
                    print(f" - orig_size: {sample_target['orig_size'].tolist()}")
                    print(f" - pill_names: {sample_target['pill_names']}")                                    
                break

        return loader
    
    # 시험의 경우
    elif mode == 'test':
        test_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator
        )

        # 배치 사이즈 예시
        if debug:
            print(f"\n{mode}_loader의 데이터 크기: {len(test_loader)}")
            for batch in test_loader:
                if batch is not None:
                    # (img, img_file)
                    images, img_name = batch
                    print(f"Batch size: {len(images)}")

                    # # 첫 번째 샘플의 image 정보 출력
                    print(f"Image shape: {images[0].shape}")
                    print(f"File  name : {img_name[0]}")
                break  

        return test_loader
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose either 'train', 'val', or 'test'.")

####################################################################################################
# 메인 시작    
if __name__ == "__main__":
    # argparse 시작
    parser = argparse.ArgumentParser(description="PillDataset DataLoader Debug Runner")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val', 'test'], help="운영 모드")
    parser.add_argument('--batch_size', type=int, default=4, help="배치 크기")
    parser.add_argument('--debug', action='store_true', help="디버깅 모드 여부")
    args = parser.parse_args()
    # 변경 사항 끝

    TRAIN_ROOT = "data/train_images"
    TRAIN_ANN_DIR = "data/train_annots_modify"
    TEST_ROOT = "data/test_images"

    # 선택한 모드에 맞춰 로더 실행 및 디버깅 테스트
    if args.mode in ['train', 'val']:
        loader = get_loader(TRAIN_ROOT, TRAIN_ANN_DIR, batch_size=args.batch_size, mode=args.mode, debug=args.debug)
        print(f"{args.mode} loader 생성 완료.")
    elif args.mode == 'test':
        loader = get_loader(TEST_ROOT, None, batch_size=args.batch_size, mode=args.mode, debug=args.debug)
        print("test loader 생성 완료.")
    else:
        raise ValueError("잘못된 mode 값입니다. 'train', 'val', 'test' 중 하나를 입력하세요.")