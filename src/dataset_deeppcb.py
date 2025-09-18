# src/dataset_deeppcb.py
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# -----------------------------
# Public constants
# -----------------------------
CLASS_ID_TO_NAME: Dict[int, str] = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "pinhole",
    5: "spurious_copper",
}


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class DeepPCBConfig:
    datasets_root: str = "./datasets"  # มี trainval.txt / test.txt ที่ราก
    split: str = "train"  # "train" | "val"
    imgsz: int = 640  # 0 = ไม่ resize
    groups: Optional[List[str]] = None  # ตัวอย่าง ["group00041", "group12000"] ถ้าอยากจำกัด
    # ถ้าอยาก custom transforms ให้ส่งใน Dataset ctor


# -----------------------------
# Helpers
# -----------------------------
_ID_PATTERN = re.compile(r"\d{5,}")


def _read_ids(split_path: str) -> List[str]:
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ split: {split_path}")
    with open(split_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = _ID_PATTERN.findall(text)
    if not ids:
        raise RuntimeError(
            f"{split_path} ว่างหรือไม่พบ ID (ต้องเป็นกลุ่มตัวเลขยาว เช่น 00041000)"
        )
    return ids


def _iter_groups(root: str, allow: Optional[List[str]]) -> List[str]:
    """คืนลิสต์พาธของโฟลเดอร์ group* ที่ใช้ได้"""
    groups = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        if not name.startswith("group"):
            continue
        if allow and name not in allow:
            continue
        groups.append(p)
    if not groups:
        raise RuntimeError(f"ไม่พบโฟลเดอร์ 'group*' ภายใต้ {root}")
    return sorted(groups)


def _pair_img_label_dirs(group_dir: str) -> List[Tuple[str, str]]:
    """
    จับคู่ (img_dir, label_dir) จากชื่อโฟลเดอร์ 'ID' และ 'ID_not'
    เช่น group00041/00041 ↔ group00041/00041_not
    """
    subs = [
        d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))
    ]
    pairs: List[Tuple[str, str]] = []
    for s in subs:
        if s.endswith("_not"):
            continue
        lbl = f"{s}_not"
        if lbl in subs:
            pairs.append((os.path.join(group_dir, s), os.path.join(group_dir, lbl)))
    return sorted(pairs)


def _safe_boxes(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)  # x1,x2
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)  # y1,y2
    # ensure x2>x1, y2>y1
    boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0] + 1)
    boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1] + 1)
    return boxes


def _resize_boxes(boxes: np.ndarray, ow: int, oh: int, nw: int, nh: int) -> np.ndarray:
    if boxes.size == 0 or (ow == nw and oh == nh):
        return boxes
    sx, sy = nw / float(ow), nh / float(oh)
    boxes = boxes.copy()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return boxes


def default_transforms() -> T.Compose:
    return T.Compose([T.ToTensor()])


# -----------------------------
# Dataset
# -----------------------------
class DeepPCBDataset(Dataset):
    """
    รองรับโครงสร้าง:
        datasets/
          trainval.txt
          test.txt
          group00041/
            00041/        # รูป .jpg (ชื่อมีเลข id อยู่ในชื่อ)
            00041_not/    # label .txt (x1 y1 x2 y2 class[1..6]) ชื่อไฟล์ = <id>.txt
          group12000/
            12000/
            12000_not/
          ...

    หมายเหตุ:
    - split="train" ใช้ trainval.txt ที่ราก, split="val" ใช้ test.txt ที่ราก
    - ดึงเลข id จากชื่อไฟล์รูปด้วย regex (รองรับ ..._test.jpg หรือชื่ออื่นที่มีตัวเลข id)
    - map class 1..6 → 0..5 อัตโนมัติ
    """

    def __init__(
        self,
        datasets_root: str,
        split: str = "train",
        imgsz: int = 640,
        transforms: Optional[T.Compose] = None,
        groups: Optional[List[str]] = None,
    ) -> None:
        assert split in ("train", "val"), "split ต้องเป็น 'train' หรือ 'val'"
        self.cfg = DeepPCBConfig(datasets_root, split, imgsz, groups)
        self.transforms = transforms or default_transforms()

        # 1) load split ids
        split_file = os.path.join(
            self.cfg.datasets_root, "trainval.txt" if split == "train" else "test.txt"
        )
        self._id_set = set(_read_ids(split_file))

        # 2) index (image_path, label_path)
        self._items: List[Tuple[str, str]] = []
        for g in _iter_groups(self.cfg.datasets_root, self.cfg.groups):
            for img_dir, lbl_dir in _pair_img_label_dirs(g):
                for fname in os.listdir(img_dir):
                    if not fname.lower().endswith(".jpg"):
                        continue
                    m = _ID_PATTERN.search(fname)
                    if not m:
                        continue
                    fid = m.group(0)
                    if fid not in self._id_set:
                        continue
                    img_path = os.path.join(img_dir, fname)
                    lbl_path = os.path.join(lbl_dir, f"{fid}.txt")
                    self._items.append((img_path, lbl_path))

        if not self._items:
            raise RuntimeError(
                f"ไม่พบข้อมูลสำหรับ split='{split}' - ตรวจ trainval/test.txt, "
                f"ชื่อไฟล์รูปต้องมีเลข id, และมี <id>.txt ใน *_not/"
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path, txt_path = self._items[index]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"อ่านรูปไม่ได้: {img_path}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        oh, ow = img.shape[:2]

        # อ่านกล่องจาก .txt: x1 y1 x2 y2 cls(1..6)
        boxes: List[List[float]] = []
        labels: List[int] = []
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip().split()
                    if len(s) < 5:
                        continue
                    x1, y1, x2, y2, cls = map(int, s[:5])
                    cls = max(1, min(6, cls)) - 1  # map 1..6 → 0..5
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)
        # รูปไม่มี .txt → ถือว่า “ไม่มี defect” (กล่องว่าง) ก็ยังเทรนได้

        boxes_np = (
            np.asarray(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )
        boxes_np = _safe_boxes(boxes_np, ow, oh)

        # resize หากตั้ง imgsz > 0
        if self.cfg.imgsz and self.cfg.imgsz > 0:
            img = cv2.resize(
                img, (self.cfg.imgsz, self.cfg.imgsz), interpolation=cv2.INTER_LINEAR
            )
            boxes_np = _resize_boxes(boxes_np, ow, oh, self.cfg.imgsz, self.cfg.imgsz)

        # to tensor
        image = self.transforms(img)
        target: Dict[str, Any] = {
            "boxes": torch.as_tensor(boxes_np, dtype=torch.float32),
            "labels": torch.as_tensor(
                np.asarray(labels, dtype=np.int64), dtype=torch.int64
            ),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (
                (
                    torch.as_tensor(boxes_np[:, 2] - boxes_np[:, 0])
                    * torch.as_tensor(boxes_np[:, 3] - boxes_np[:, 1])
                )
                if boxes_np.size
                else torch.tensor([])
            ),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "orig_size": torch.as_tensor([oh, ow], dtype=torch.int64),
            "path": img_path,
        }
        return image, target


# -----------------------------
# DataLoader helper
# -----------------------------
def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)
