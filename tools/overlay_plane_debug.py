#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import cv2
import numpy as np
import yaml
import sys

# import CourtGeometry
ROOT = Path(__file__).resolve().parents[0]      # /workspace/tools
sys.path.append(str(ROOT.parent / "src"))       # add /workspace/src
from geom.coords import CourtGeometry


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--out_video", required=True)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--frame_w", type=int, default=None)
    p.add_argument("--frame_h", type=int, default=None)
    p.add_argument("--max_frames", type=int, default=0, help="0 = all")
    p.add_argument("--tol", type=float, default=3.0)
    p.add_argument("--use_bounds_penalty", action="store_true",
                  help="Penalize planes if mapped world coords are out of bounds.")
    p.add_argument("--bounds_margin_cm", type=float, default=20.0)
    return p.parse_args()


def load_yaml_points(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    pts = cfg.get("pixel_points", [])
    if not pts:
        return None
    pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    return pts


def build_label_index(labels_dir: Path):
    files = sorted(Path(labels_dir).glob("*.txt"))
    idx = {}
    counter = 0
    for f in files:
        m = re.search(r"(\d+)$", f.stem)
        if m:
            frame_idx = int(m.group(1))
        else:
            frame_idx = counter
        counter += 1
        idx[frame_idx] = f
    return idx


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    cls_id = int(float(parts[0]))
    cx_n = float(parts[1])
    cy_n = float(parts[2])
    w_n  = float(parts[3])
    h_n  = float(parts[4])
    conf = float(parts[5])
    return cls_id, cx_n, cy_n, w_n, h_n, conf


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_poly(img, poly, name):
    if poly is None:
        return
    cv2.polylines(img, [poly], isClosed=True, color=(255, 255, 255), thickness=2)
    # label near first point
    x, y = int(poly[0,0,0]), int(poly[0,0,1])
    cv2.putText(img, name, (x+6, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


def plane_errs_raw(cg, xy_px):
    return {
        "front_wall": cg._reproj_err(xy_px, cg.Hw, cg.Hw_inv),
        "left_wall":  cg._reproj_err(xy_px, cg.Hl, cg.Hl_inv),
        "right_wall": cg._reproj_err(xy_px, cg.Hr, cg.Hr_inv),
        "floor":      cg._reproj_err(xy_px, cg.Hf, cg.Hf_inv),
    }


def plane_world_uv(cg, plane, xy_px):
    # return plane coords (u,v) in that plane’s world units
    if plane == "front_wall":
        u, v = cg.px_to_wall_plane(xy_px)   # (y,z)
        return float(u), float(v), cg.wall_size
    if plane == "left_wall":
        u, v = cg.px_to_left_plane(xy_px)   # (x,z)
        return float(u), float(v), cg.left_size
    if plane == "right_wall":
        u, v = cg.px_to_right_plane(xy_px)  # (x,z)
        return float(u), float(v), cg.right_size
    if plane == "floor":
        u, v = cg.px_to_floor_plane(xy_px)  # (y,x)
        return float(u), float(v), cg.floor_size
    return None, None, (0.0, 0.0)


def pick_plane(cg, xy_px, tol, use_bounds_penalty=False, margin_cm=20.0):
    errs = plane_errs_raw(cg, xy_px)

    if not use_bounds_penalty:
        plane = min(errs.items(), key=lambda kv: kv[1])[0]
        return plane, errs[plane], errs

    # bounds-penalized selection:
    scored = {}
    for p, e in errs.items():
        u, v, (su, sv) = plane_world_uv(cg, p, xy_px)
        penalty = 0.0

        # If mapped coords are far outside plane size, this plane is not plausible.
        if u is None:
            penalty = 1e9
        else:
            if u < -margin_cm or u > su + margin_cm or v < -margin_cm or v > sv + margin_cm:
                penalty = 1e6  # huge penalty

        scored[p] = e + penalty

    plane = min(scored.items(), key=lambda kv: kv[1])[0]
    return plane, errs[plane], errs


def color_for_plane(plane):
    # BGR
    if plane == "front_wall": return (0, 0, 255)     # red
    if plane == "floor":      return (0, 255, 0)     # green
    if plane == "left_wall":  return (255, 0, 0)     # blue
    if plane == "right_wall": return (0, 255, 255)   # yellow
    return (180, 180, 180)                           # gray


def main():
    args = parse_args()
    video_path = str(args.video)
    labels_dir = Path(args.labels_dir)
    out_path   = str(args.out_video)

    cg = CourtGeometry(
        frontwall_yaml="configs/calib.frontwall.yaml",
        floor_yaml="configs/calib.floor.yaml",
        leftwall_yaml="configs/calib.leftwall.yaml",
        rightwall_yaml="configs/calib.rightwall.yaml",
        on_plane_px_tol=float(args.tol),
    )

    # load polys for drawing
    poly_front = load_yaml_points("configs/calib.frontwall.yaml")
    poly_floor = load_yaml_points("configs/calib.floor.yaml")
    poly_left  = load_yaml_points("configs/calib.leftwall.yaml")
    poly_right = load_yaml_points("configs/calib.rightwall.yaml")

    label_index = build_label_index(labels_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    W = args.frame_w or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = args.frame_h or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps else 60.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    frame_idx = 0
    max_frames = args.max_frames

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # draw calibration quads
        draw_poly(frame, poly_front, "front_wall")
        draw_poly(frame, poly_floor, "floor")
        draw_poly(frame, poly_left,  "left_wall")
        draw_poly(frame, poly_right, "right_wall")

        # load labels for this frame index if present
        f = label_index.get(frame_idx, None)
        if f and f.exists():
            with f.open() as fh:
                for line in fh:
                    parsed = parse_label_line(line)
                    if not parsed:
                        continue
                    cls_id, cx_n, cy_n, w_n, h_n, conf = parsed

                    cx = cx_n * W
                    cy = cy_n * H
                    bw = w_n * W
                    bh = h_n * H

                    x1 = int(clamp(cx - bw/2, 0, W-1))
                    y1 = int(clamp(cy - bh/2, 0, H-1))
                    x2 = int(clamp(cx + bw/2, 0, W-1))
                    y2 = int(clamp(cy + bh/2, 0, H-1))

                    plane, best_err, all_errs = pick_plane(
                        cg, (cx, cy), tol=args.tol,
                        use_bounds_penalty=args.use_bounds_penalty,
                        margin_cm=args.bounds_margin_cm
                    )

                    col = color_for_plane(plane)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                    cv2.circle(frame, (int(cx), int(cy)), 3, col, -1)

                    txt1 = f"{plane} conf={conf:.2f} err={best_err:.2f}px"
                    txt2 = f"fw={all_errs['front_wall']:.1f} lf={all_errs['left_wall']:.1f} rt={all_errs['right_wall']:.1f} fl={all_errs['floor']:.1f}"

                    cv2.putText(frame, txt1, (x1, max(20, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                    cv2.putText(frame, txt2, (x1, max(40, y1+10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 2)

        # header info
        cv2.putText(frame, f"frame={frame_idx}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        out.write(frame)
        frame_idx += 1

        if max_frames and frame_idx >= max_frames:
            break

    cap.release()
    out.release()
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
