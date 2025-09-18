# src/run_train.py
from trainer import PCBTrainer, TrainConfig


def main():
    cfg = TrainConfig(
        datasets_root="./datasets",
        out_dir="./models",
        epochs=20,
        batch_size=2,
        imgsz=640,
        workers=4,  # ถ้าอยากง่ายสุดให้ผ่านแน่ ๆ ใส่ 0 ก็ได้
        lr=5e-3,
        amp=True,
        steps_per_epoch=None,
        pretrained_backbone=True,
    )
    trainer = PCBTrainer(cfg)
    best = trainer.fit()
    print("best ckpt:", best)


if __name__ == "__main__":
    # สำหรับ Windows/pyinstaller
    import multiprocessing as mp

    mp.freeze_support()
    main()
