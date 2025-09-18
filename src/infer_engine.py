# src/infer_engine.py
import os
import cv2
import torch
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# from dataset_deeppcb import CLASS_ID_TO_NAME  # ชื่อคลาส 6 คลาส

CLASS_ID_TO_NAME = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "pinhole",
    5: "spurious_copper",
}


class PCBInferEngine:
    def __init__(
        self, weights: str, device: str | None = None, num_classes: int = 1 + 6
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model(num_classes)
        state = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @staticmethod
    def _build_model(num_classes: int):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        return model

    @torch.no_grad()
    def predict_image(self, image_bgr: np.ndarray, conf: float = 0.5):
        """รับภาพ BGR (np.uint8) → คืน list[dict]: {box, label, score, name}"""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = torchvision.transforms.functional.to_tensor(rgb).to(self.device)
        out = self.model([tensor])[0]
        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()  # 1..N (0 คือ background)
        scores = out["scores"].cpu().numpy()

        preds = []
        for (x1, y1, x2, y2), lab, sc in zip(boxes, labels, scores):
            if sc < conf:
                continue
            cls_idx = int(lab - 1)  # map กลับ 0..5
            preds.append(
                {
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "label": cls_idx,
                    "name": CLASS_ID_TO_NAME.get(cls_idx, str(cls_idx)),
                    "score": float(sc),
                }
            )
        return preds

    @staticmethod
    def draw_predictions(image_bgr: np.ndarray, preds, color=(0, 0, 255), thickness=2):
        """วาดกล่องบนภาพ (in-place)"""
        img = image_bgr.copy()
        for p in preds:
            x1, y1, x2, y2 = map(int, p["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                img,
                f'{p["name"]}:{p["score"]:.2f}',
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return img

    def predict_path(
        self, image_path: str, conf: float = 0.5, save_to: str | None = None
    ):
        """อ่านรูปจากพาธ → preds (และบันทึกถ้ากำหนด save_to)"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        preds = self.predict_image(img, conf=conf)
        if save_to:
            os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
            out_img = self.draw_predictions(img, preds)
            cv2.imwrite(save_to, out_img)
        return preds

    def predict_folder(
        self, folder: str, out_dir: str | None = None, conf: float = 0.5
    ):
        """วิ่งทั้งโฟลเดอร์ .jpg → คืน dict[filename] = preds (และบันทึกรูปถ้ากำหนด out_dir)"""
        results = {}
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        for f in sorted(os.listdir(folder)):
            if not f.lower().endswith(".jpg"):
                continue
            in_path = os.path.join(folder, f)
            preds = self.predict_path(
                in_path,
                conf=conf,
                save_to=(os.path.join(out_dir, f"pred_{f}") if out_dir else None),
            )
            results[f] = preds
        return results
