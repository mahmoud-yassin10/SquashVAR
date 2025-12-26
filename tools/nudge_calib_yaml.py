#!/usr/bin/env python3
import argparse, yaml
import numpy as np
import cv2
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml_in", required=True)
    ap.add_argument("--idx", type=int, required=True)   # 0..3
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.yaml_in, "r"))
    px = np.array(cfg["pixel_points"], dtype=np.float32)
    wp = np.array(cfg["world_points_cm"], dtype=np.float32)

    px[args.idx, 0] += args.dx
    px[args.idx, 1] += args.dy

    H = cv2.getPerspectiveTransform(px, wp)
    cfg["pixel_points"] = px.tolist()
    cfg["homography"] = H.tolist()

    Path(args.yaml_in).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(args.yaml_in, "w"), sort_keys=False)
    print("Updated", args.yaml_in)

if __name__ == "__main__":
    main()
