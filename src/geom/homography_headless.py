import argparse, yaml, cv2, numpy as np, os

def parse_points(s: str):
    # Format: "x1,y1 x2,y2 x3,y3 x4,y4" (clockwise)
    pts=[]
    for tok in s.strip().split():
        x,y = tok.split(",")
        pts.append([float(x), float(y)])
    if len(pts)!=4:
        raise ValueError("Need exactly 4 points: x1,y1 x2,y2 x3,y3 x4,y4")
    return np.array(pts, dtype=np.float32)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help='e.g. "123,456 1700,450 1710,980 120,990" (clockwise)')
    ap.add_argument("--world", nargs=2, type=float, required=True, help="W H in chosen units (e.g., cm)")
    ap.add_argument("--out", default="configs/calib.frontwall.yaml")
    args = ap.parse_args()

    px = parse_points(args.points)
    W, H = args.world
    world = np.array([[0,0],[W,0],[W,H],[0,H]], dtype=np.float32)

    Hmat, _ = cv2.findHomography(px, world)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    yaml.safe_dump({
        "homography": Hmat.tolist(),
        "world_size": [W, H],
        "pixel_points": px.tolist()
    }, open(args.out, "w"))
    print("Saved:", args.out)
