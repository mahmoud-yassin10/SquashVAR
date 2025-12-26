#!/usr/bin/env python3
import argparse, yaml
import numpy as np
import cv2
from pathlib import Path

# Edit these once (your P1..P8)
P = {
    "P1": (345.0,   0.0),
    "P2": (103.0, 422.0),
    "P3": (926.0,   1.0),
    "P4": (1168.0,417.0),
    "P5": (904.0, 398.0),
    "P6": (358.0, 392.0),
    "P7": (126.0, 717.0),
    "P8": (1150.0,720.0),
}

def write_calib(path, world_size, pixel_pts, world_pts):
    H = cv2.getPerspectiveTransform(np.array(pixel_pts, np.float32),
                                    np.array(world_pts, np.float32))  # pixel->world
    cfg = {
        "world_size": [float(world_size[0]), float(world_size[1])],
        "pixel_points": [[float(x), float(y)] for x,y in pixel_pts],
        "world_points_cm": [[float(x), float(y)] for x,y in world_pts],
        "homography": [[float(v) for v in row] for row in H.tolist()],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {path}")

def apply_global_transform(dx, dy, scale):
    pts = np.array(list(P.values()), dtype=np.float32)  # (8,2)
    c = pts.mean(axis=0, keepdims=True)
    pts2 = (pts - c) * scale + c
    pts2[:,0] += dx
    pts2[:,1] += dy
    keys = list(P.keys())
    for i,k in enumerate(keys):
        P[k] = (float(pts2[i,0]), float(pts2[i,1]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out_dir", default="/workspace/configs")
    args = ap.parse_args()

    apply_global_transform(args.dx, args.dy, args.scale)

    out = Path(args.out_dir)

    # FRONT WALL (y,z) size: (640,457)
    # pixel: P1,P3,P5,P6 -> world: (0,457),(640,457),(640,0),(0,0)
    write_calib(out/"calib.frontwall.yaml",
                (640.0, 457.0),
                [P["P1"], P["P3"], P["P5"], P["P6"]],
                [(0.0,457.0), (640.0,457.0), (640.0,0.0), (0.0,0.0)])

    # FLOOR (y,x) size: (640,975)
    # pixel: P6,P5,P8,P7 -> world: (0,0),(640,0),(640,975),(0,975)
    write_calib(out/"calib.floor.yaml",
                (640.0, 975.0),
                [P["P6"], P["P5"], P["P8"], P["P7"]],
                [(0.0,0.0), (640.0,0.0), (640.0,975.0), (0.0,975.0)])

    # LEFT WALL (x,z) size: (975,457)
    # pixel: P1,P2,P7,P6 -> world: (0,457),(975,457),(975,0),(0,0)
    write_calib(out/"calib.leftwall.yaml",
                (975.0, 457.0),
                [P["P1"], P["P2"], P["P7"], P["P6"]],
                [(0.0,457.0), (975.0,457.0), (975.0,0.0), (0.0,0.0)])

    # RIGHT WALL (x,z) size: (975,457)
    # pixel: P3,P4,P8,P5 -> world: (0,457),(975,457),(975,0),(0,0)
    write_calib(out/"calib.rightwall.yaml",
                (975.0, 457.0),
                [P["P3"], P["P4"], P["P8"], P["P5"]],
                [(0.0,457.0), (975.0,457.0), (975.0,0.0), (0.0,0.0)])

if __name__ == "__main__":
    main()
