#!/usr/bin/env python3
import argparse, yaml
import numpy as np
import cv2
from pathlib import Path

FILES = [
    "/workspace/configs/calib.frontwall.yaml",
    "/workspace/configs/calib.floor.yaml",
    "/workspace/configs/calib.leftwall.yaml",
    "/workspace/configs/calib.rightwall.yaml",
]

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dx", type=float, required=True)
    p.add_argument("--dy", type=float, required=True)
    return p.parse_args()

def shift_one(path, dx, dy):
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    px = np.array(cfg["pixel_points"], dtype=np.float32)     # (4,2)
    wp = np.array(cfg["world_points_cm"], dtype=np.float32)  # (4,2)

    px[:,0] += dx
    px[:,1] += dy

    H = cv2.getPerspectiveTransform(px, wp)  # pixel -> world

    cfg["pixel_points"] = [[float(x), float(y)] for x,y in px]
    cfg["homography"] = [[float(v) for v in row] for row in H.tolist()]

    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("Shifted", path)

def main():
    a = parse()
    for f in FILES:
        shift_one(f, a.dx, a.dy)

if __name__ == "__main__":
    main()
