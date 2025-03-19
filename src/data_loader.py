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
    """
    어노테이션 디렉토리 내 JSON 파일들을 탐색하여  
    카테고리 ID와 이름 간의 매핑을 생성하고,  
    이름 기준 정렬 후 인덱스를 재부여하는 함수입니다.

    Args:
        ann_dir (str): 어노테이션 JSON 파일들이 저장된 디렉토리 경로

    Returns:
        name_to_idx (dict):  
            - Key: 카테고리 이름 (str)  
            - Value: 인덱스 (int)  
            - 0은 'Background'로 설정하며, 마지막 인덱스는 'No Class'로 지정합니다.

        idx_to_name (dict):  
            - Key: 인덱스 (int)  
            - Value: 카테고리 이름 (str)  
            - 역방향으로 인덱스를 통해 이름을 찾을 수 있는 매핑을 제공합니다.

    Note:
        - 중복된 카테고리 이름은 제거 후 정렬합니다.
        - 이름 기준으로 정렬하여 일관된 인덱스를 제공합니다.
        - 추후 모델 학습 시 카테고리 ID → 카테고리 이름 매핑 및 시각화에 활용됩니다.
    """
    # 디버깅 메시지 출력
    if not isinstance(ann_dir, str):
        raise TypeError(f"ann_dir는 문자열(str)이어야 합니다. 현재 타입: {type(ann_dir)}")
    if not os.path.exists(ann_dir):
        raise FileNotFoundError(f"ann_dir 경로가 존재하지 않습니다: {ann_dir}")
    
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
        PillDataset 클래스

        알약 이미지와 어노테이션(json) 파일을 로드하여  
        모델 학습 및 검증을 위한 데이터셋 형태로 제공하는 PyTorch Dataset 클래스입니다.  

        Attributes:
            image_dir (str): 이미지 파일이 저장된 디렉토리 경로  
            ann_dir (str, optional): 어노테이션 JSON 파일들이 저장된 디렉토리 경로 (train/val 모드에서 필요)  
            mode (str): 'train', 'val', 'test' 중 하나로 데이터셋의 동작 모드를 결정  
            category_mapping (dict): 카테고리 이름과 인덱스 매핑 정보  
            transform (callable, optional): 이미지 및 bounding box에 적용할 변환 함수  
            debug (bool): 이미지/어노테이션 불일치 시 경고 출력 여부  

        Notes:
            - train/val 모드에서는 이미지-어노테이션 쌍을 필터링 후 로드합니다.  
            - test 모드에서는 어노테이션 없이 이미지 파일 경로만 반환합니다.  
        """
        # 디버깅 메시지 출력 
        if not isinstance(image_dir, str):
            raise TypeError(f"image_dir는 문자열(str)이어야 합니다. 현재 타입: {type(image_dir)}")
        if ann_dir is not None and not isinstance(ann_dir, str):
            raise TypeError(f"ann_dir는 문자열(str)이거나 None이어야 합니다. 현재 타입: {type(ann_dir)}")
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"mode는 'train', 'val', 'test' 중 하나여야 합니다. 현재 입력: {mode}")
        if category_mapping is not None and not isinstance(category_mapping, dict):
            raise TypeError(f"category_mapping은 dict 타입이어야 합니다. 현재 타입: {type(category_mapping)}")
        if transform is not None and not callable(transform):
            raise TypeError(f"transform은 호출 가능 객체이어야 합니다. 현재 타입: {type(transform)}")
        if not isinstance(debug, bool):
            raise TypeError(f"debug는 bool 타입이어야 합니다. 현재 타입: {type(debug)}")
        
        # 인자 받기기
        self.img_dir = image_dir
        self.ann_dir = ann_dir
        self.mode = mode
        self.transform = transform
        self.category_mapping = category_mapping    # 카테고리 이름 <-> 아이디 매핑

        # 이미지
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
                print(f"\n[DEBUG] 총 {len(common_name)}개의 이미지-어노테이션 파일 처리")
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
        __getitem__ 함수

        인덱스를 입력 받아 해당 인덱스의 이미지 및 어노테이션 데이터를 반환합니다.

        Args:
            idx (int): 호출할 데이터의 인덱스  

        Returns:
            - train/val 모드:  
                img (TVImage): 이미지 텐서  
                targets (dict):  
                    - boxes (BoundingBoxes): 바운딩 박스 좌표  
                    - labels (torch.Tensor): 클래스 인덱스 라벨  
                    - image_id (torch.Tensor): 이미지 고유 식별자  
                    - area (torch.Tensor): 객체 영역  
                    - is_crowd (torch.Tensor): crowd 플래그 (0으로 고정)  
                    - orig_size (torch.Tensor): 원본 이미지 크기  
                    - pill_names (list): 바운딩 박스에 해당하는 알약 이름  

            - test 모드:  
                img (TVImage): 이미지 텐서  
                img_file (str): 이미지 파일명 (추론 시 사용)  
        """
        # 디버깅 메시지 출력
        if not isinstance(idx, int):
            raise TypeError(f"인덱스는 int 타입이어야 합니다. 현재 타입: {type(idx)}")
        if idx >= len(self.images):
            raise IndexError(f"인덱스 {idx}가 데이터셋 크기 {len(self.images)}보다 큽니다.")

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

            # COCODateset 기준 + 알약 이름 추가
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

########################################################################################################
# test.py때에 수정이 필요해 보임
            # 이미지, _ for in batch
            return img, img_file


    def __len__(self):
        return len(self.images)

    def get_img_info(self, idx):
        """
        get_img_info 함수

        특정 인덱스에 해당하는 이미지의 파일명 및 크기 정보를 반환합니다.

        Args:
            idx (int): 가져올 이미지의 인덱스  

        Returns:
            dict:  
                - file_name (str): 이미지 파일 이름  
                - height (int): 이미지 세로 크기  
                - width (int): 이미지 가로 크기  

        Note:
            어노테이션 파일을 로드하여 이미지 메타데이터 정보를 가져옵니다.  
            사용 예시: info = dataset.get_img_info(0)
                       print(info)
        """
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
        """
        get_ann_info 함수

        특정 인덱스에 해당하는 어노테이션 JSON 파일의 내용을 반환합니다.

        Args:
            idx (int): 가져올 어노테이션 인덱스  

        Returns:
            dict: 해당 어노테이션 JSON의 전체 내용 (파싱된 JSON)  

        Note:
            어노테이션 파일이 잘못되었거나 JSON 디코딩 실패 시 None 반환  
            사용예시: ann = dataset.get_ann_info(0)
                      print(json.dumps(ann, indent=2, ensure_ascii=False))
        """
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
    # 디버깅 메시지 출력
    if not isinstance(img_dir, str):
        raise TypeError(f"img_dir는 문자열(str)이어야 합니다. 현재 타입: {type(img_dir)}")
    if ann_dir is not None and not isinstance(ann_dir, str):
        raise TypeError(f"ann_dir는 문자열(str)이거나 None이어야 합니다. 현재 타입: {type(ann_dir)}")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size는 양의 정수여야 합니다. 현재 입력: {batch_size}")
    if mode not in ['train', 'val', 'test']:
        raise ValueError(f"mode는 'train', 'val', 'test' 중 하나여야 합니다. 현재 입력: {mode}")
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio는 0과 1 사이의 실수여야 합니다. 현재 입력: {val_ratio}")
    if not isinstance(debug, bool):
        raise TypeError(f"debug는 bool 타입이어야 합니다. 현재 타입: {type(debug)}")
    
    # 트랜스폼
    transforms = get_transforms(mode=mode)

    # 카테고리 매핑 (train/val에서만 필요)
    if mode in ['train', 'val']:
        name_to_idx, idx_to_name = get_category_mapping(ann_dir=ann_dir)
        # 매핑 보여주기
        if debug:
            print("\n[DEBUG] 카테고리 매핑 정보:")
            print(f"- 총 클래스 수: {len(name_to_idx)}")
            print(f"- No Class 인덱스: {name_to_idx['No Class']}")
            print("\n[DEBUG] idx_to_name 매핑 (정렬 출력):")
            max_idx_len = len(str(max(idx_to_name.keys())))  # 인덱스 최대 길이 구하기
            for idx in sorted(idx_to_name.keys()):
                print(f"  {idx:>{max_idx_len}}: {idx_to_name[idx]}")

    else:
        name_to_idx, idx_to_name = None, None

    # 데이터셋
    dataset = PillDataset(image_dir=img_dir, ann_dir=ann_dir, mode=mode, category_mapping=name_to_idx, transform=transforms, debug=debug)

    # [DEBUG 추가]
    if debug and mode in ['train', 'val']:
        print(f"\n[DEBUG] 전체 데이터셋 크기: {len(dataset)}개")

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
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size], generator=generator)

        # [DEBUG 추가]
        if debug:
            print(f"[DEBUG] 랜덤시드 고정: {seed}")
            print(f"[DEBUG] Train/Val Split: Train = {train_size}, Val = {len(dataset) - train_size}")

        loader = DataLoader(
            train_dataset if mode == 'train' else val_dataset,
            batch_size=batch_size,
            shuffle=(mode == 'train'),
            drop_last=True,
            collate_fn=collator
        )

        if debug:
            print(f"\n[DEBUG] {mode} loader 배치 수: {len(loader)}")
            for batch in loader:
                if batch is not None:
                    images, targets = batch
                    print(f"[DEBUG] Batch size: {len(images)}")
                    print(f"[DEBUG] 첫 이미지 크기: {images[0].shape}")

                    sample_target = targets[0]
                    boxes = sample_target['boxes']
                    areas = sample_target['area']
                    print(f"[DEBUG] 첫 샘플 image_id: {sample_target['image_id'].item()}")
                    print(f"[DEBUG] 박스 개수: {boxes.shape[0]}")
                    print(f"[DEBUG] 박스 크기 (W,H) 최대/최소: {boxes[:, 2:].max().item()}, {boxes[:, 2:].min().item()}")
                    print("[DEBUG] 라벨별 출현 횟수:")
                    label_counts = torch.bincount(sample_target['labels'])
                    for idx, count in enumerate(label_counts):
                        if count > 0:
                            label_name = name_to_idx and idx_to_name.get(idx, "Unknown")
                            print(f"  - {idx}: {label_name} → {count}회")
                        
                    print(f"[DEBUG] 이미지 텐서 메모리: {images[0].element_size() * images[0].nelement() / 1024 ** 2:.2f}MB")
                    print(f"[DEBUG] Pill names (샘플): {sample_target['pill_names']}")                                 
                break

        return loader
    
    # 시험의 경우
    elif mode == 'test':
        test_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collator
        )

        # 배치 사이즈 예시
        if debug:
            print(f"\n[DEBUG] 테스트 데이터셋 배치 수: {len(test_loader)}") 
            for batch in test_loader:
                if batch is not None:
                    images, img_name = batch
                    print(f"[DEBUG] Batch size: {len(images)}")
                    print(f"[DEBUG] 첫 이미지 shape: {images[0].shape}")
                    print(f"[DEBUG] 배치 이미지 파일명 목록: {img_name}")
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