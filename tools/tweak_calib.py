#!/usr/bin/env python3
import argparse
import yaml
import numpy as np
import cv2
from pathlib import Path

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--in_yaml", required=True)
    p.add_argument("--out_yaml", required=True)
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--scale", type=float, default=1.0)
    return p.parse_args()

def main():
    a = parse()
    inp = Path(a.in_yaml)
    outp = Path(a.out_yaml)

    with inp.open("r") as f:
        cfg = yaml.safe_load(f)

    px = np.array(cfg["pixel_points"], dtype=np.float32)          # (4,2)
    wp = np.array(cfg["world_points_cm"], dtype=np.float32)       # (4,2)

    # scale about centroid + shift
    c = px.mean(axis=0, keepdims=True)
    px2 = (px - c) * float(a.scale) + c
    px2[:, 0] += float(a.dx)
    px2[:, 1] += float(a.dy)

    # recompute pixel->world homography
    H = cv2.getPerspectiveTransform(px2, wp)  # pixel -> world (cm)

    cfg["pixel_points"] = [[float(x), float(y)] for x, y in px2]
    cfg["homography"] = [[float(v) for v in row] for row in H.tolist()]

    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote {outp}")
    print(f"dx={a.dx} dy={a.dy} scale={a.scale}")

if __name__ == "__main__":
    main()
