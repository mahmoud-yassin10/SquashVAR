#!/usr/bin/env python3
import argparse, yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--idx", type=int, required=True)   # 0..3
    ap.add_argument("--dx", type=float, default=0.0)
    ap.add_argument("--dy", type=float, default=0.0)
    args = ap.parse_args()

    with open(args.yaml, "r") as f:
        cfg = yaml.safe_load(f)

    pts = cfg["pixel_points"]
    x, y = pts[args.idx]
    pts[args.idx] = [float(x) + args.dx, float(y) + args.dy]
    cfg["pixel_points"] = pts

    with open(args.yaml, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("Updated", args.yaml, "point", args.idx, "->", pts[args.idx])

if __name__ == "__main__":
    main()

