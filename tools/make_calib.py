#!/usr/bin/env python3
import yaml, cv2, numpy as np, argparse, pathlib

def H_from_pts(px_pts, world_pts):
    src = np.array(px_pts, dtype=np.float32)
    dst = np.array(world_pts, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)  # pixel -> world
    return H

def write_yaml(path, px_pts, world_pts, world_size):
    H = H_from_pts(px_pts, world_pts)
    out = {
        "world_size": list(map(float, world_size)),
        "pixel_points": [[float(x), float(y)] for x,y in px_pts],
        "world_points_cm": [[float(x), float(y)] for x,y in world_pts],
        "homography": H.tolist(),
    }
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"Wrote {path}")

def main():
    # FRONT WALL (u=y, v=z)
    front_px = [(345,0),(926,1),(904,398),(358,392)]          # P1,P3,P5,P6
    front_w  = [(0,457),(640,457),(640,0),(0,0)]
    write_yaml("/workspace/configs/calib.frontwall.yaml", front_px, front_w, world_size=(640,457))

    # FLOOR (u=y, v=x)
    floor_px = [(358,392),(904,398),(1150,720),(126,717)]     # P6,P5,P8,P7
    floor_w  = [(0,0),(640,0),(640,975),(0,975)]
    write_yaml("/workspace/configs/calib.floor.yaml", floor_px, floor_w, world_size=(640,975))

if __name__ == "__main__":
    main()
