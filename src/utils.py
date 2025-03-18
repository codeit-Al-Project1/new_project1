import torch
from torchvision.ops import box_iou

def evaluate_mAP(preds, targets):
    iou_threshold = 0.5
    correct = 0
    total = 0

    for pred, target in zip(preds, targets):
        if len(pred["boxes"]) == 0 or len(target["boxes"]) == 0:
            continue
        
        ious = box_iou(pred["boxes"], target["boxes"])
        correct += (ious > iou_threshold).sum().item()
        total += len(target["boxes"])

    return correct / total if total > 0 else 0
