from src.infer_engine import PCBInferEngine

engine = PCBInferEngine(weights="./models/best.pth")  # auto: cuda ถ้ามี
# ทำนายรูปเดียว + เซฟรูปผลลัพธ์
preds = engine.predict_path(
    "./datasets/group20085/20085/20085001_test.jpg",
    conf=0.75,
    save_to="./outputs/pred_20085001.jpg"
)
print(preds[:2])  # ดู 2 กล่องแรก

# # ทำนายทั้งโฟลเดอร์
# results = engine.predict_folder(
#     "./datasets/group00041/00041",
#     out_dir="./outputs/preds",
#     conf=0.6
# )
# print("files:", len(results))
