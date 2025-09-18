# src/models/fasterrcnn.py
from __future__ import annotations
import torch
from typing import Optional

# torchvision รุ่นใหม่จะมีพวก Weights เหล่านี้
try:
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models import ResNet50_Weights

    _HAS_NEW_API = True
except Exception:
    # รองรับรุ่นเก่าที่ยังใช้ pretrained=
    from torchvision.models.detection import fasterrcnn_resnet50_fpn

    FasterRCNN_ResNet50_FPN_Weights = None  # type: ignore
    ResNet50_Weights = None  # type: ignore
    _HAS_NEW_API = False

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def _set_num_classes(model, num_classes: int) -> None:
    """เปลี่ยนหัวทำนายให้ตรงกับจำนวนคลาส (รวม background แล้ว)"""
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


def build_fasterrcnn_resnet50_fpn(
    num_classes: int,
    *,
    pretrained: bool = True,  # อยากได้ weight COCO ไหม (หัวจะถูกแทนที่อยู่ดี)
    pretrained_backbone: bool = True,  # เอา weight backbone มั้ย (ถ้าเริ่มจาก random ตั้ง False)
    trainable_backbone_layers: int = 3,  # 0–5 (3 คือ block หลัง ๆ train ได้)
    image_mean: Optional[list[float]] = None,
    image_std: Optional[list[float]] = None,
    device: Optional[torch.device | str] = None,
    freeze_backbone: bool = False,  # ถ้าต้องการ freeze ฟีเจอร์ extractor
) -> torch.nn.Module:
    """
    สร้าง Faster R-CNN (ResNet50 + FPN) แบบรองรับทั้ง torchvision รุ่นใหม่/เก่า
    - num_classes รวม background แล้ว (เช่น 6 defect + 1 = 7)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    kwargs = {
        "trainable_backbone_layers": trainable_backbone_layers,
    }

    if _HAS_NEW_API:
        # รุ่นใหม่: ใช้ weights / weights_backbone
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=weights_backbone,
            **kwargs,
        )
    else:
        # รุ่นเก่า: ยังใช้ pretrained / pretrained_backbone ได้
        try:
            model = fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                pretrained_backbone=pretrained_backbone,
                **kwargs,
            )
        except TypeError:
            # เผื่อบาง build รับได้แค่ pretrained อย่างเดียว
            model = fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                **kwargs,
            )

    # ตั้งค่ามาตรฐานภาพ (ถ้ามี)
    if image_mean is not None:
        model.transform.image_mean = image_mean
    if image_std is not None:
        model.transform.image_std = image_std

    # เปลี่ยนหัวให้ตรงจำนวนคลาส
    _set_num_classes(model, num_classes)

    # freeze backbone ถ้าต้องการ
    if freeze_backbone:
        for name, p in model.backbone.named_parameters():
            p.requires_grad = False

    model.to(device)
    return model


# ===== ตัวอย่างการใช้งาน =====
# from models.fasterrcnn import build_fasterrcnn_resnet50_fpn
# model = build_fasterrcnn_resnet50_fpn(
#     num_classes=7,
#     pretrained=True,
#     pretrained_backbone=True,
#     trainable_backbone_layers=3,
#     device="cuda",
# )
