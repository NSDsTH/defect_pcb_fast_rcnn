# src/evaluator.py
from __future__ import annotations
import os
from typing import List, Tuple, Dict, Any, Optional
import time
import numpy as np
import torch

from infer_engine import PCBInferEngine


# --------- helper: อ่าน GT ของ DeepPCB (.txt: x1 y1 x2 y2 class[1..6]) ----------
def load_gt_txt(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    boxes, labels = [], []
    if not os.path.isfile(txt_path):
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            xs = s.split()
            if len(xs) < 5:
                continue
            x1, y1, x2, y2 = map(float, xs[:4])
            cls = int(xs[4]) - 1  # 1..6 -> 0..5
            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    return np.asarray(boxes, np.float32), np.asarray(labels, np.int64)


# --------- helper: IoU ----------
def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)


# --------- helper: match greedy (F1 @ IoU>thr) ----------
def match_one_image(
    pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thr=0.33
):
    P, G = pred_boxes.shape[0], gt_boxes.shape[0]
    if P == 0 and G == 0:
        return 0, 0, 0
    if P == 0:
        return 0, 0, G
    if G == 0:
        return 0, P, 0

    ious = iou_xyxy(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)  # high->low
    used_gt = set()
    TP = FP = 0
    for p in order:
        g = int(np.argmax(ious[p]))
        ok_iou = ious[p, g] >= iou_thr
        ok_cls = (pred_labels[p] == gt_labels[g]) if g not in used_gt else False
        if ok_iou and ok_cls:
            TP += 1
            used_gt.add(g)
        else:
            FP += 1
    FN = G - len(used_gt)
    return TP, FP, FN


def prf(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return precision, recall, f1


def _with_suffix(path_rel: str, suffix: str) -> str:
    import os

    root, ext = os.path.splitext(path_rel)
    if not ext:  # กันเคสไม่มีนามสกุล
        return path_rel
    return root + suffix + ext


class PCBEvaluator:
    """
    ประเมินโมเดลแบบง่าย:
      - F1 / Precision / Recall @ IoU>thr (ดีสำหรับ DeepPCB: thr=0.33)
      - AP@IoU=0.33 และ mAP COCO [.5:.95] ผ่าน torchmetrics

    วิธีใช้ดูข้างล่างสุด (example)
    """

    def __init__(
        self,
        weights: str,
        conf_thresh: float = 0.5,
        iou_thr_f1: float = 0.33,
        device: Optional[str] = None,
        use_coco_map: bool = True,
        use_ap_033: bool = True,
        image_suffix: Optional[str] = None,
    ):
        self.engine = PCBInferEngine(weights=weights, device=device)
        self.conf_thresh = float(conf_thresh)
        self.iou_thr_f1 = float(iou_thr_f1)

        self.use_coco_map = use_coco_map
        self.use_ap_033 = use_ap_033
        self.image_suffix = image_suffix

        # เตรียม metric ของ torchmetrics
        self.map_coco = None
        self.map_033 = None
        if self.use_coco_map or self.use_ap_033:
            from torchmetrics.detection.mean_ap import MeanAveragePrecision

            if self.use_coco_map:
                self.map_coco = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
            if self.use_ap_033:
                # AP ที่ IoU เดียว = 0.33
                # self.map_033 = MeanAveragePrecision(
                #     box_format="xyxy",
                #     iou_type="bbox",
                #     iou_thresholds=torch.tensor([self.iou_thr_f1]),
                # )
                self.map_033 = MeanAveragePrecision(
                    box_format="xyxy",
                    iou_type="bbox",
                    iou_thresholds=[float(self.iou_thr_f1)],  # ต้องเป็น list[float]
                    # class_metrics=True  # ถ้าอยากได้รายคลาส เปิดอันนี้ได้
                )

        # ตัวนับภาพและ PRF
        self.reset_counters()

    def reset_counters(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.num_images = 0
        if self.map_coco is not None:
            self.map_coco.reset()
        if self.map_033 is not None:
            self.map_033.reset()

    def add_sample(self, image_path: str, gt_txt_path: str):
        """
        เพิ่มหนึ่งภาพ (path รูป + path label .txt) เข้า evaluator
        จะรันทำนายด้วย engine แล้วอัปเดต metric ทั้งหมด
        """
        # GT
        gt_boxes, gt_labels = load_gt_txt(gt_txt_path)

        # Predict (ดึงทั้งหมดแล้วค่อยกรองด้วย conf)
        preds = self.engine.predict_path(image_path, conf=0.0)
        if len(preds) == 0:
            p_boxes = np.zeros((0, 4), np.float32)
            p_labels = np.zeros((0,), np.int64)
            p_scores = np.zeros((0,), np.float32)
        else:
            preds = [p for p in preds if p["score"] >= self.conf_thresh]
            if len(preds) == 0:
                p_boxes = np.zeros((0, 4), np.float32)
                p_labels = np.zeros((0,), np.int64)
                p_scores = np.zeros((0,), np.float32)
            else:
                p_boxes = np.asarray([p["box"] for p in preds], np.float32)
                p_labels = np.asarray([p["label"] for p in preds], np.int64)
                p_scores = np.asarray([p["score"] for p in preds], np.float32)

        # --- F1 (greedy match @ IoU>iour_thr_f1) ---
        tp, fp, fn = match_one_image(
            p_boxes, p_labels, p_scores, gt_boxes, gt_labels, iou_thr=self.iou_thr_f1
        )
        self.total_tp += tp
        self.total_fp += fp
        self.total_fn += fn

        # --- mAP (COCO) + AP@0.33 ---
        preds_tm = [
            {
                "boxes": torch.from_numpy(p_boxes),
                "scores": torch.from_numpy(p_scores),
                "labels": torch.from_numpy(p_labels),
            }
        ]
        target_tm = [
            {
                "boxes": torch.from_numpy(gt_boxes),
                "labels": torch.from_numpy(gt_labels),
            }
        ]
        if self.map_coco is not None:
            self.map_coco.update(preds_tm, target_tm)
        if self.map_033 is not None:
            self.map_033.update(preds_tm, target_tm)

        self.num_images += 1

    def add_from_split_file(self, datasets_root: str, split_file: str, limit: int = 0):
        pairs = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                img_rel, txt_rel = s.split()
                pairs.append((img_rel, txt_rel))
                if limit > 0 and len(pairs) >= limit:
                    break

        missing = 0
        for img_rel, txt_rel in pairs:
            # normalize slash
            img_rel = img_rel.replace("\\", "/")
            txt_rel = txt_rel.replace("\\", "/")

            # เตรียม “รายการพาธรูปที่เป็นไปได้”
            img_rel_candidates = [img_rel]
            if self.image_suffix:
                img_rel_candidates.append(_with_suffix(img_rel, self.image_suffix))

            chosen_img = None
            for rel in img_rel_candidates:
                cand = os.path.normpath(os.path.join(datasets_root, rel))
                if os.path.isfile(cand):
                    chosen_img = cand
                    break

            gt_path = os.path.normpath(os.path.join(datasets_root, txt_rel))

            if chosen_img is None or not os.path.isfile(gt_path):
                print(f"[warn] not found -> img:{img_rel_candidates}  gt:{txt_rel}")
                missing += 1
                continue

            self.add_sample(chosen_img, gt_path)

        if missing:
            print(f"[info] skipped {missing} samples (file missing)")

        def compute(self) -> Dict[str, Any]:
            """
            คืนผลสรุปเป็น dict:
            - precision, recall, f1 (ที่ IoU>iou_thr_f1 และใช้ conf_thresh)
            - ap033_map (ค่าเฉลี่ย AP@0.33)  [ถ้าเปิด]
            - coco_map, coco_ap50, coco_ap75   [ถ้าเปิด]
            - images_count
            """
            precision, recall, f1 = prf(self.total_tp, self.total_fp, self.total_fn)
            out = {
                "images_count": self.num_images,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "conf_thresh": float(self.conf_thresh),
                "iou_thr_f1": float(self.iou_thr_f1),
            }
            if self.map_033 is not None:
                r033 = self.map_033.compute()
                out["ap033_map"] = float(r033["map"].item()) if "map" in r033 else None
            if self.map_coco is not None:
                rcoco = self.map_coco.compute()
                out.update(
                    {
                        "coco_map": float(rcoco["map"].item()),
                        "coco_ap50": float(rcoco["map_50"].item()),
                        "coco_ap75": float(rcoco["map_75"].item()),
                    }
                )
            return out

    def summary_str(self, stats: Dict[str, Any]) -> str:
        lines = []
        lines.append("========== SUMMARY ==========")
        lines.append(f"Images           : {stats['images_count']}")
        lines.append(
            f"Conf / IoU(F1)   : {stats['conf_thresh']:.2f} / {stats['iou_thr_f1']:.2f}"
        )
        lines.append(
            f"Precision / Recall / F1 : {stats['precision']:.4f} / {stats['recall']:.4f} / {stats['f1']:.4f}"
        )
        if "ap033_map" in stats:
            lines.append(f"AP@0.33 (mean)   : {stats['ap033_map']:.4f}")
        if "coco_map" in stats:
            lines.append(
                f"mAP@[.5:.95]     : {stats['coco_map']:.4f} | AP50={stats['coco_ap50']:.4f} | AP75={stats['coco_ap75']:.4f}"
            )
        return "\n".join(lines)
