
#!/usr/bin/env python3
import yaml, cv2, numpy as np
from pathlib import Path

def compute_H(px_pts, world_pts):
    src = np.array(px_pts, dtype=np.float32)
    dst = np.array(world_pts, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)  # pixel -> world

def write_calib(path, world_size, px_pts, world_pts):
    H = compute_H(px_pts, world_pts)
    out = {
        "world_size": [float(world_size[0]), float(world_size[1])],
        "pixel_points": [[float(x), float(y)] for x, y in px_pts],
        "world_points_cm": [[float(u), float(v)] for u, v in world_pts],
        "homography": H.tolist(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"Wrote {path}")

def main():
    W, L, Z = 640.0, 975.0, 457.0

    # front wall (u=y, v=z): P1,P3,P5,P6
    write_calib(
        "/workspace/configs/calib.frontwall.yaml",
        (W, Z),
        [(345,0), (926,1), (904,398), (358,392)],
        [(0,Z), (W,Z), (W,0), (0,0)],
    )

    # floor (u=y, v=x): P6,P5,P8,P7
    write_calib(
        "/workspace/configs/calib.floor.yaml",
        (W, L),
        [(358,392), (904,398), (1150,720), (126,717)],
        [(0,0), (W,0), (W,L), (0,L)],
    )

    # left wall (u=x, v=z): P1,P2,P7,P6
    write_calib(
        "/workspace/configs/calib.leftwall.yaml",
        (L, Z),
        [(345,0), (103,422), (126,717), (358,392)],
        [(0,Z), (L,Z), (L,0), (0,0)],
    )

    # right wall (u=x, v=z): P3,P4,P8,P5
    write_calib(
        "/workspace/configs/calib.rightwall.yaml",
        (L, Z),
        [(926,1), (1168,417), (1150,720), (904,398)],
        [(0,Z), (L,Z), (L,0), (0,0)],
    )

if __name__ == "__main__":
    main()
