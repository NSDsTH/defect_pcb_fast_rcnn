# src/models/fasterrcnn.py
from typing import Optional
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_fasterrcnn_resnet50_fpn(num_classes: int,
                                  pretrained: bool = True) -> torchvision.models.detection.FasterRCNN:
    """
    สร้าง Faster R-CNN ResNet50 + FPN
    - num_classes: รวม background (เช่น defects=6 → num_classes=7)
    - pretrained: ใช้ weight COCO เป็นฐานเพื่อ convergence ไวขึ้น
    """
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    head = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.box_predictor = head
    return model
