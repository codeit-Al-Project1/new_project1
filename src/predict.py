import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
from PIL import Image

# 테스트 이미지 시각화
def visualize_predictions(model, image_path, device, threshold=0.5):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
    # 모델 예측
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    # 예측 결과 시각화
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(image)

    # 예측된 bounding box와 confidence score 가져오기
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    # confidence score가 threshold 이상인 것만 필터링
    for i in range(len(scores)):
        if scores[i] > threshold:
            box = boxes[i].cpu().numpy()
            label = labels[i].cpu().numpy()
            score = scores[i].cpu().numpy()

            # bounding box 그리기
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            # 레이블과 score 출력
            ax.text(
                box[0], box[1] - 10, 
                f"Class: {label}, Score: {score:.4f}", 
                fontsize=10, color='red', 
                bbox=dict(facecolor='white', alpha=0.7)
            )

    plt.show()
