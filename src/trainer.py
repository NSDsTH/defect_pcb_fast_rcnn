# src/trainer.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import torch
from torch.utils.data import DataLoader

# from torch.cuda.amp import autocast, GradScaler

from torch.amp import autocast, GradScaler

from dataset_deeppcb import DeepPCBDataset, collate_fn
from models.fasterrcnn import build_fasterrcnn_resnet50_fpn


# -----------------------------
# Config
# -----------------------------
@dataclass
class TrainConfig:
    datasets_root: str = "./datasets"
    out_dir: str = "./models"

    # data
    imgsz: int = 640
    batch_size: int = 2
    workers: int = 2

    # train
    epochs: int = 20
    lr: float = 5e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    steps_per_epoch: Optional[int] = None  # จำกัดจำนวน step ต่อ epoch (ถ้า None ใช้ทั้งชุด)

    # model
    num_defect_classes: int = 6
    pretrained_backbone: bool = True  # ใช้ weight COCO เป็นฐาน

    # perf
    amp: bool = True  # mixed precision เฉพาะเมื่อใช้ GPU


# -----------------------------
# Trainer
# -----------------------------
class PCBTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Device = {self.device}")

        # datasets
        self.train_set = DeepPCBDataset(
            cfg.datasets_root, split="train", imgsz=cfg.imgsz
        )
        self.val_set = DeepPCBDataset(cfg.datasets_root, split="val", imgsz=cfg.imgsz)

        # loaders
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.workers,
            collate_fn=collate_fn,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            collate_fn=collate_fn,
        )

        # model
        num_classes = 1 + cfg.num_defect_classes  # background + defects
        self.model = build_fasterrcnn_resnet50_fpn(
            num_classes=num_classes, pretrained=cfg.pretrained_backbone
        ).to(self.device)

        # optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )

        # amp
        # self.use_amp = (self.device.type == "cuda") and cfg.amp
        # self.scaler = GradScaler(enabled=self.use_amp)
        self.use_amp = (self.device.type == "cuda") and cfg.amp
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)

        # ckpt
        self.best_val = float("inf")
        self.best_path = os.path.join(cfg.out_dir, "best.pth")
        self.last_path = os.path.join(cfg.out_dir, "last.pth")

    # --------- public API ----------
    def fit(self) -> str:
        """เทรนเต็มรอบ epochs; คืน path ของ best checkpoint"""
        for epoch in range(1, self.cfg.epochs + 1):
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._evaluate_loss()
            print(
                f"[Epoch {epoch}/{self.cfg.epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
            )

            if val_loss < self.best_val:
                self.best_val = val_loss
                torch.save(self.model.state_dict(), self.best_path)
                print(f"✅ Saved BEST -> {self.best_path} (val_loss={val_loss:.4f})")

        torch.save(self.model.state_dict(), self.last_path)
        print(f"Saved LAST -> {self.last_path}")
        return self.best_path

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str, strict: bool = True):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state, strict=strict)

    # --------- train/eval internals ----------
    def _train_one_epoch(self, epoch: int, print_freq: int = 20) -> float:
        self.model.train()
        running = 0.0
        steps_limit = self.cfg.steps_per_epoch or len(self.train_loader)

        for i, (images, targets) in enumerate(self.train_loader, start=1):
            if i > steps_limit:
                break

            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: (v.to(self.device) if hasattr(v, "to") else v)
                    for k, v in t.items()
                }
                for t in targets
            ]

            if self.use_amp:
                # with autocast():
                #     loss_dict = self.model(images, targets)
                #     loss = sum(loss_dict.values())
                with autocast(device_type="cuda", enabled=self.use_amp):
                    loss_dict = self.model(images, targets)
                    loss = sum(loss_dict.values())
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            running += loss.item()
            if i % print_freq == 0:
                print(f"[epoch {epoch} | {i}/{steps_limit}] loss={loss.item():.4f}")

        return running / max(1, min(steps_limit, len(self.train_loader)))

    @torch.no_grad()
    def _evaluate_loss(self) -> float:
        # NOTE: torchvision detection จะคืน loss dict เฉพาะใน train mode
        was_training = self.model.training
        self.model.train()

        total = 0.0
        count = 0
        for images, targets in self.val_loader:
            images = [img.to(self.device) for img in images]
            targets = [
                {
                    k: (v.to(self.device) if hasattr(v, "to") else v)
                    for k, v in t.items()
                }
                for t in targets
            ]
            loss_dict = self.model(images, targets)
            total += sum(loss_dict.values()).item()
            count += 1

        if not was_training:
            self.model.eval()
        return total / max(1, count)
