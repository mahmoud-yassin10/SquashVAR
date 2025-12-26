#!/usr/bin/env python3
import argparse, yaml, numpy as np, cv2
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--idx", type=int, required=True)   # 0..3
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    args = ap.parse_args()

    p = Path(args.yaml)
    cfg = yaml.safe_load(p.read_text())

    pts = cfg.get("pixel_points", [])
    wpts = cfg.get("world_points_cm", [])

    if len(pts) != 4 or len(wpts) != 4:
        raise SystemExit("Need exactly 4 pixel_points and 4 world_points_cm")

    pts = np.array(pts, dtype=np.float32)
    wpts = np.array(wpts, dtype=np.float32)

    pts[args.idx, 0] += args.dx
    pts[args.idx, 1] += args.dy

    H, _ = cv2.findHomography(pts, wpts, method=0)
    cfg["pixel_points"] = pts.tolist()
    cfg["homography"] = H.tolist()

    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Updated", p)

if __name__ == "__main__":
    main()
