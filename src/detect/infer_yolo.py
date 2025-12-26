import argparse, os, cv2, time
from ultralytics import YOLO

def main(args):
    # Model: coco-pretrained just to prove the pipe; later load your fine-tuned ball model.
    model = YOLO("yolov8n.pt")
    # 32 is 'sports ball' in COCO; you can remove filter once you have your own class.
    classes = [32] if args.sports_ball_only else None

    cap = cv2.VideoCapture(args.source)
    assert cap.isOpened(), f"Cannot open {args.source}"
    os.makedirs(args.out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(args.out_dir, "yolo_out.mp4"), fourcc, fps, (w, h))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model.predict(frame, imgsz=args.imgsz, verbose=False, classes=classes, conf=args.conf)
        annotated = res[0].plot()
        out.write(annotated)
        if args.show:
            cv2.imshow("YOLOv8", annotated)
            if cv2.waitKey(1) & 0xFF == 27: break
        i += 1
    cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True, help="path to a .mp4")
    p.add_argument("--out_dir", type=str, default="runs/infer")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--sports_ball_only", action="store_true")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    main(args)
