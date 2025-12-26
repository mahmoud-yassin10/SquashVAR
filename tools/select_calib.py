# tools/select_calib.py
import argparse, yaml, cv2, sys
from pathlib import Path

def read_video_size(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def scale_points_in_dict(d, s):
    # scale known pixel-key containers safely
    for key in list(d.keys()):
        v = d[key]
        if key.endswith("_px") and isinstance(v, dict):
            for k2, p in v.items():
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    d[key][k2] = [float(p[0]) * s, float(p[1]) * s]
        elif key.endswith("_px") and isinstance(v, list):
            new_list = []
            for p in v:
                if isinstance(p, (list, tuple)) and len(p) == 2:
                    new_list.append([float(p[0]) * s, float(p[1]) * s])
                else:
                    new_list.append(p)
            d[key] = new_list
        elif isinstance(v, dict):
            scale_points_in_dict(v, s)  # recurse (safe; will only scale *_px)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    scale_points_in_dict(item, s)

def make_scaled_yaml(base_yaml, out_yaml, target_w, target_h, base_w=1280, base_h=720):
    s_w = target_w / float(base_w)
    s_h = target_h / float(base_h)
    if abs(s_w - s_h) > 1e-3:
        print(f"[warn] Non-uniform scale (w={s_w:.3f}, h={s_h:.3f}). "
              "This implies a different FOV/AR; results may be off.", file=sys.stderr)
    s = (s_w + s_h) / 2.0  # use average scale if slightly off

    d = yaml.safe_load(open(base_yaml, "r"))
    # image size
    if "image" in d and isinstance(d["image"], dict):
        d["image"]["width"]  = int(target_w)
        d["image"]["height"] = int(target_h)
    # scale pixel points (any key ending with '_px')
    scale_points_in_dict(d, s)
    yaml.safe_dump(d, open(out_yaml, "w"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--front720", default="configs/calib.frontwall.720.yaml")
    ap.add_argument("--floor720", default="configs/calib.floor.720.yaml")
    ap.add_argument("--out_front", default="configs/calib.frontwall.yaml")
    ap.add_argument("--out_floor", default="configs/calib.floor.yaml")
    ap.add_argument("--base_w", type=int, default=1280)
    ap.add_argument("--base_h", type=int, default=720)
    args = ap.parse_args()

    vw, vh = read_video_size(args.video)
    print(f"[auto-calib] video size: {vw}×{vh}")

    # quick AR sanity
    ar_ok = abs((vw / vh) - (16/9)) < 1e-3

    if vw == args.base_w and vh == args.base_h:
        # exact 720p → just copy
        Path(args.out_front).write_text(Path(args.front720).read_text())
        Path(args.out_floor).write_text(Path(args.floor720).read_text())
        print(f"[auto-calib] Applied 720p calib → {args.out_front}, {args.out_floor}")
        return

    # scale from 720p base
    if not ar_ok:
        print("[warn] Aspect ratio is not 16:9; scaling from 720p base may be inaccurate.", file=sys.stderr)

    make_scaled_yaml(args.front720, args.out_front, vw, vh, args.base_w, args.base_h)
    make_scaled_yaml(args.floor720, args.out_floor, vw, vh, args.base_w, args.base_h)
    print(f"[auto-calib] Scaled calib to {vw}×{vh} → {args.out_front}, {args.out_floor}")

if __name__ == "__main__":
    main()
