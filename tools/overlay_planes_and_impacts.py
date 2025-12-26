#!/usr/bin/env python3
import argparse, csv
from collections import defaultdict
from pathlib import Path
import yaml
import cv2
import numpy as np

def load_poly(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    pts = cfg.get("pixel_points", [])
    if not pts:
        return None
    pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    return pts

def load_world_by_frame(world_csv):
    by_frame = defaultdict(list)
    with open(world_csv, "r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fr = int(float(r["frame_idx"]))
            by_frame[fr].append(r)
    return by_frame

def load_impacts_by_frame(impacts_csv):
    impact_frames = set()
    with open(impacts_csv, "r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fr = int(float(r["frame_idx"]))
            impact_frames.add(fr)
    return impact_frames

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--world_csv", required=True)
    ap.add_argument("--impacts_csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=600)
    args = ap.parse_args()

    polys = {
        "front_wall": load_poly("/workspace/configs/calib.frontwall.yaml"),
        "floor":      load_poly("/workspace/configs/calib.floor.yaml"),
        "left_wall":  load_poly("/workspace/configs/calib.leftwall.yaml"),
        "right_wall": load_poly("/workspace/configs/calib.rightwall.yaml"),
    }

    # color BGR
    colors = {
        "front_wall": (0, 255, 255),
        "floor":      (0, 255, 0),
        "left_wall":  (255, 0, 0),
        "right_wall": (0, 0, 255),
        "air":        (160, 160, 160),
    }

    world_by_frame = load_world_by_frame(args.world_csv)
    impact_frames = load_impacts_by_frame(args.impacts_csv)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    # seek
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    fr = args.start

    for _ in range(args.max_frames):
        ok, frame = cap.read()
        if not ok:
            break

        # draw polygons
        for name, poly in polys.items():
            if poly is None:
                continue
            cv2.polylines(frame, [poly], True, colors[name], 2)

        # draw detections for this frame
        rows = world_by_frame.get(fr, [])
        for r in rows:
            plane = (r.get("plane") or "air").lower()
            if plane not in colors:
                plane = "air"
            cx = int(float(r["cx_px"]))
            cy = int(float(r["cy_px"]))
            cv2.circle(frame, (cx, cy), 4, colors[plane], -1)

        # draw impact marker (big circle + text)
        if fr in impact_frames:
            cv2.circle(frame, (40, 40), 18, (0, 0, 255), -1)
            cv2.putText(frame, "IMPACT", (70, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(frame, f"frame={fr}", (20, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        vw.write(frame)
        fr += 1

    vw.release()
    cap.release()
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
