import argparse
from pathlib import Path
from ultralytics import YOLO

def main(a):
    model = YOLO(a.model)  # e.g., 'yolov8n.pt'

    results = model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,            # -1 = autobatch (GPU only)
        device=a.device,          # "0" for GPU0, "cpu" for CPU
        workers=a.workers,        # 0..2 is safe in Docker/WSL
        project=a.project,
        name=a.name,
        seed=a.seed,
        val=True,
        # tiny-ball friendly
        rect=True,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        # scheduler/optimizer
        optimizer=a.optimizer,
        lr0=a.lr0,
        patience=a.patience,
        cos_lr=a.coslr,
        # resource control (subset of data)
        fraction=a.fraction,      # 0<frac<=1
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    print(f"Best weights: {best}")
    print(f"Last weights: {last}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/roboflow/data.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=-1, help="-1 = autobatch on GPU")
    p.add_argument("--device", default="0")             # GPU0 by default
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="squash_yolo8n_gpu")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--coslr", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fraction", type=float, default=1.0,
                   help="Fraction of dataset to use (0<frac<=1). Use 0.10 for a smoke test.")
    a = p.parse_args()
    main(a)
