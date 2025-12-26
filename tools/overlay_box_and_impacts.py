#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import yaml
import numpy as np
import cv2

def load_poly(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    pts = np.array(cfg.get("pixel_points", []), dtype=np.int32)
    if pts.shape != (4, 2):
        return None
    return pts.reshape(-1, 1, 2)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--world_csv", required=True)
    p.add_argument("--impacts_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--persist", type=int, default=10)
    p.add_argument("--radius", type=int, default=10)
    p.add_argument("--thickness", type=int, default=2)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=999999)
    p.add_argument("--search_k", type=int, default=2, help="search +/-k frames for a matching world row")
    return p.parse_args()

def get_frame_idx(row):
    for k in ("frame_idx", "frame", "impact_frame", "fi"):
        if k in row and row[k] != "":
            return int(float(row[k]))
    return None

def get_track_id(row):
    for k in ("track_id", "tid"):
        if k in row and row[k] != "":
            return int(float(row[k]))
    return -1

def best_world_row(world_rows, frame_idx, tid, k=2):
    # exact frame preferred; if missing, search +/-k
    best = None
    best_c = -1.0
    for df in range(-k, k + 1):
        rows = world_rows.get((frame_idx + df, tid), [])
        for r in rows:
            try:
                c = float(r.get("conf", 0.0) or 0.0)
            except:
                c = 0.0
            if c > best_c:
                best = r
                best_c = c
    return best

def main():
    a = parse_args()

    polys = {
        "front_wall": load_poly("/workspace/configs/calib.frontwall.yaml"),
        "floor":      load_poly("/workspace/configs/calib.floor.yaml"),
        "left_wall":  load_poly("/workspace/configs/calib.leftwall.yaml"),
        "right_wall": load_poly("/workspace/configs/calib.rightwall.yaml"),
    }

    # world rows grouped by (frame, track_id)
    world_rows = {}
    with open(a.world_csv, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fi = get_frame_idx(r)
            if fi is None:
                continue
            tid = get_track_id(r)
            world_rows.setdefault((fi, tid), []).append(r)

    # impacts grouped by frame
    impacts = {}
    with open(a.impacts_csv, newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fi = get_frame_idx(r)
            if fi is None:
                continue
            impacts.setdefault(fi, []).append(r)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {a.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(
        str(outp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps and fps > 0 else 30.0,
        (W, H),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, a.start)

    recent = []  # (impact_frame, x, y, label)
    frame_idx = a.start
    written = 0

    while written < a.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # draw court polygons
        for name, poly in polys.items():
            if poly is None:
                continue
            cv2.polylines(frame, [poly], True, (255, 255, 255), 2)

        # add impacts for this frame (use px_used from world row)
        if frame_idx in impacts:
            for r in impacts[frame_idx]:
                tid = get_track_id(r)
                wrow = best_world_row(world_rows, frame_idx, tid, k=a.search_k)

                if wrow is not None:
                    x = int(float(wrow.get("px_used_x", wrow.get("cx_px", 0)) or 0))
                    y = int(float(wrow.get("px_used_y", wrow.get("cy_px", 0)) or 0))
                else:
                    x, y = 0, 0

                label = r.get("surface", r.get("plane", "impact"))
                recent.append((frame_idx, x, y, label))

        # keep recent impacts visible
        recent = [(fi, x, y, lbl) for (fi, x, y, lbl) in recent if frame_idx - fi <= a.persist]

        # draw impacts
        for (fi, x, y, lbl) in recent:
            cv2.circle(frame, (x, y), a.radius, (255, 255, 255), a.thickness)
            cv2.putText(frame, str(lbl), (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"frame={frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        vw.write(frame)
        frame_idx += 1
        written += 1

    cap.release()
    vw.release()
    print("Wrote", outp)

if __name__ == "__main__":
    main()
