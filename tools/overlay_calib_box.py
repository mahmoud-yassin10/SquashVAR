#!/usr/bin/env python3
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path

def load_poly(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    pts = np.array(cfg.get("pixel_points", []), dtype=np.int32)
    if pts.shape != (4,2):
        return None
    return pts.reshape(-1,1,2)

def put_label(img, poly, text):
    if poly is None: return
    x,y = int(poly[0,0,0]), int(poly[0,0,1])
    cv2.putText(img, text, (x+5,y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=900)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--thickness", type=int, default=2)
    args = ap.parse_args()

    polys = {
        "front_wall": load_poly("/workspace/configs/calib.frontwall.yaml"),
        "floor":      load_poly("/workspace/configs/calib.floor.yaml"),
        "left_wall":  load_poly("/workspace/configs/calib.leftwall.yaml"),
        "right_wall": load_poly("/workspace/configs/calib.rightwall.yaml"),
    }

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps if fps>0 else 30.0, (W,H))

    # seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    i = 0
    while i < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # draw
        for name, poly in polys.items():
            if poly is None: 
                continue
            cv2.polylines(frame, [poly], True, (255,255,255), args.thickness)
            put_label(frame, poly, name)

        cv2.putText(frame, f"frame={args.start+i}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        vw.write(frame)
        i += 1

    cap.release()
    vw.release()
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
