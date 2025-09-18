import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluator import PCBEvaluator

if __name__ == "__main__":
    eva = PCBEvaluator(
        weights="models/best.pth",
        conf_thresh=0.75,
        iou_thr_f1=0.33,
        use_coco_map=True,
        use_ap_033=True,
        image_suffix="_test",
    )
    eva.add_from_split_file(
        datasets_root="./datasets",  # root ที่มีโฟลเดอร์ groupxxxxx อยู่
        split_file="./datasets/test.txt",
        limit=0,
    )
    stats = eva.compute()
    print(eva.summary_str(stats))
