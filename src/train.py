import torch
import torch.nn as nn
from tqdm import tqdm
from src.dataset import get_dataloader
from src.model import get_fast_rcnn_model, save_model
from src.dataset import split_dataloader
from src.utils import get_optimizer, get_scheduler  # utils.py에서 가져오기

"""
**Fast R-CNN에서 SGD를 쓰는 이유**

일반화 성능이 더 좋음 (과적합 방지)
메모리 효율적 (대규모 데이터에 적합)
Momentum을 추가하면 안정적 (빠른 수렴)
즉, Fast R-CNN에서는 빠르게 최적화하는 것보다, 일반화가 잘 되면서도 안정적인 학습이 더 중요하므로 SGD를 선택
"""
def train(json_dir, img_dir, batch_size=5, num_classes=74, num_epochs=5, lr=0.001, optimizer_name = "sgd", scheduler_name = "step", device="cuda"):
    # 학습용 데이터 로더, 분할
    dataloader = get_dataloader(json_dir, img_dir, batch_size)
    train_loader, val_loader = split_dataloader(dataloader, val_split=0.2)

    # 모델 정의
    model = get_fast_rcnn_model(num_classes).to(device)
    
    # utils.py에서 옵티마이저 가져오기
    optimizer = get_optimizer(optimizer_name, model, lr=lr, weight_decay=0.0005)
    
    # utils.py에서 스케줄러 가져오기
    scheduler = get_scheduler(scheduler_name, optimizer)

    # 검증 손실 초기화
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        epoch_loss_details = {}

        # 1. 학습 단계
        for images, targets in tqdm(train_loader, total=len(train_loader), desc="Processing training"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 모델 학습
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()

            # Gradient Clipping 추가
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += losses.item()

            # 개별 손실 요소 누적
            for k, v in loss_dict.items():
                if k not in epoch_loss_details:
                    epoch_loss_details[k] = 0
                epoch_loss_details[k] += v.item()

        avg_loss_details = ", ".join([f"{k}: {v / len(train_loader):.4f}" for k, v in epoch_loss_details.items()])
        print(f"Epoch {epoch+1} Complete - Total Loss: {total_loss:.4f}, Avg Loss Per Component: {avg_loss_details}")
        
        # 학습 종료 후 스케줄러 업데이트
        if scheduler_name.lower() == "plateau":
            scheduler.step(total_loss)  # ReduceLROnPlateau는 loss 값을 전달해야 함
        else:
            scheduler.step()
        
        # 2. 검증 단계
        model.eval()
        val_loss = 0
        val_loss_details = {}

        with torch.no_grad():
            for images, targets in tqdm(enumerate(val_loader), total=len(val_loader), desc="Processing validation"):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                val_loss += losses.item()

                for k, v in loss_dict.items():
                    if k not in val_loss_details:
                        val_loss_details[k] = 0
                    val_loss_details[k] += v.item()

        avg_val_loss_details = ", ".join([f"{k}: {v / len(val_loader):.4f}" for k, v in val_loss_details.items()])
        print(f"Validation Loss for Epoch {epoch+1} - Total Loss: {val_loss:.4f}, Avg Loss Per Component: {avg_val_loss_details}")

        # 검증 손실이 개선되었으면 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Validation Loss decreased to {best_val_loss:.4f}. Saving model...")
            save_model(model, save_dir="../models", base_name="best_trained_model", ext=".pth")

if __name__ == "__main__":
    train(json_dir="data/mapped_annotations.json", img_dir="data/train_images")