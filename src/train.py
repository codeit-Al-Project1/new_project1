import torch
import torch.optim as optim
from tqdm import tqdm
import os
import sys
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .data_loader import get_train_val_loader
from .model import get_model, save_model
from src.utils import evaluate_mAP

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
learning_rate = 0.005
weight_decay = 1e-4
num_classes = 74  # 카테고리 개수 + 배경
batch_size = 4 # ** 추가 **

# 데이터 로드
train_loader, val_loader = get_train_val_loader(batch_size, val_ratio = 0.2, debug = False ) # 데이터 디렉토리 추가하기

# 모델 로드
model = get_model(num_classes).to(device)

# 옵티마이저
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay) # SGD or AdamW

# 스케쥴러
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

# 학습 함수
def train(model, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")

    for images, targets in loop:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return running_loss / len(train_loader)

# 검증 함수
def validate(model, val_loader):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        loop = tqdm(val_loader, leave=False, desc="Validating")

        for images, targets in loop: # DataLoader에서 가져옴
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            val_loss += loss.item()

            # mAP 평가를 위한 예측 저장
            outputs = model(images)
            all_preds.extend(outputs)
            all_targets.extend(targets)

    mAP = evaluate_mAP(all_preds, all_targets)
    return val_loss / len(val_loader), mAP

# 학습 루프
best_mAP = 0.0
os.makedirs("checkpoints", exist_ok=True)

for epoch in range(epochs):
    print(f"\n🔹 Epoch {epoch+1}/{epochs} 시작")

    train_loss = train(model, train_loader, optimizer, epoch)
    val_loss, mAP = validate(model, val_loader)

    print(f"📉 Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mAP: {mAP:.4f}")

    scheduler.step(val_loss)

    if mAP > best_mAP:
        best_mAP = mAP
        save_model(model)
        print("✅ Best model saved!")

print("Training Complete!")
