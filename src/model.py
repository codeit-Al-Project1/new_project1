import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def save_model(model, path="checkpoints/best_model.pth"):
    torch.save(model.state_dict(), path)

def load_model(path, num_classes):
    model = get_model(num_classes)
    model.load_state_dict(torch.load(path))
    return model
