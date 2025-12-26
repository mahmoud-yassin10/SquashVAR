#!/usr/bin/env python3
import argparse, csv, re
from pathlib import Path
import yaml
import numpy as np
import cv2
import sys

ROOT = Path(__file__).resolve().parents[0]
sys.path.append(str(ROOT.parent / "src"))
from geom.coords import CourtGeometry

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--frame_w", type=int, required=True)
    p.add_argument("--frame_h", type=int, required=True)
    p.add_argument("--mode", choices=["center","bottom","multi"], default="multi")

    # per-plane margins (px) — allow slightly outside polygon
    p.add_argument("--margin_front", type=float, default=6.0)
    p.add_argument("--margin_floor", type=float, default=18.0)
    p.add_argument("--margin_left",  type=float, default=12.0)
    p.add_argument("--margin_right", type=float, default=12.0)

    return p.parse_args()

def iter_files(d):
    d = Path(d)
    files = sorted(d.glob("*.txt"))
    c = 0
    for f in files:
        m = re.search(r"(\d+)$", f.stem)
        idx = int(m.group(1)) if m else c
        c += 1
        yield idx, f

def parse_line(line):
    parts = line.strip().split()
    if len(parts) < 6: return None
    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    conf = float(parts[5])
    tid = -1
    if len(parts) >= 7:
        try: tid = int(float(parts[6]))
        except: tid = -1
    return cls, cx, cy, w, h, conf, tid


def load_poly(path):
    with open(path,"r") as f:
        cfg = yaml.safe_load(f)
    pts = np.array(cfg.get("pixel_points", []), dtype=np.int32)
    return pts.reshape(-1,1,2) if len(pts) else None

def poly_score(poly, pt):
    if poly is None: return -1e9
    return float(cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), True))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    a = parse()
    W,H = a.frame_w, a.frame_h
    fps = a.fps

    cg = CourtGeometry()

    poly = {
        "front_wall": load_poly("/workspace/configs/calib.frontwall.yaml"),
        "floor":      load_poly("/workspace/configs/calib.floor.yaml"),
        "left_wall":  load_poly("/workspace/configs/calib.leftwall.yaml"),
        "right_wall": load_poly("/workspace/configs/calib.rightwall.yaml"),
    }
    margins = {
        "front_wall": a.margin_front,
        "floor":      a.margin_floor,
        "left_wall":  a.margin_left,
        "right_wall": a.margin_right,
    }
    out_rows = []
    counts = {"front_wall":0,"floor":0,"left_wall":0,"right_wall":0,"air":0}

    for frame_idx, path in iter_files(a.labels_dir):
        t = frame_idx / fps
        with path.open() as fh:
            for line in fh:
                z = parse_line(line)
                if not z: continue
                cls, cx_n, cy_n, w_n, h_n, conf, tid = z
                cx = cx_n * W
                cy = cy_n * H
                bw = w_n * W
                bh = h_n * H

                # points to test
                pts = []
                if a.mode == "center":
                    pts = [(cx, cy)]
                elif a.mode == "bottom":
                    pts = [(cx, cy + 0.5*bh)]
                else:  # multi
                    pts = [
                        (cx, cy),
                        (cx, cy + 0.5*bh),
                        (cx - 0.25*bw, cy + 0.5*bh),
                        (cx + 0.25*bw, cy + 0.5*bh),
                    ]

                # choose best plane by max polygon score across all points,
                # but apply per-plane margins (gate)
                best_plane = "air"
                best_score = -1e9
                best_pt = (cx, cy)

                for pxy in pts:
                    px = (clamp(pxy[0],0,W-1), clamp(pxy[1],0,H-1))
                    for name, pg in poly.items():
                        s = poly_score(pg, px)  # signed dist (+ inside)
                        if s < -margins[name]:
                            continue
                        if s > best_score:
                            best_score = s
                            best_plane = name
                            best_pt = px

                if best_plane == "air":
                    counts["air"] += 1
                    continue


                # map using the chosen point (important!)
                if best_plane == "front_wall":
                    x_cm, y_cm, z_cm = cg.wall_cm(best_pt)
                elif best_plane == "floor":
                    x_cm, y_cm, z_cm = cg.floor_cm(best_pt)
                elif best_plane == "left_wall":
                    x_cm, y_cm, z_cm = cg.left_cm(best_pt)
                else:
                    x_cm, y_cm, z_cm = cg.right_cm(best_pt)

                counts[best_plane] += 1

                out_rows.append({
                    "frame_idx": frame_idx,
                    "t_sec": f"{t:.6f}",

                    # what nano_impacts expects
                    "cx_px": f"{cx:.2f}",
                    "cy_px": f"{cy:.2f}",
                    "w_px":  f"{bw:.2f}",
                    "h_px":  f"{bh:.2f}",
                    "conf":  f"{conf:.4f}",
                    "track_id": tid,

                    # mapping result
                    "plane": best_plane,
                    "x_cm": f"{x_cm:.3f}",
                    "y_cm": f"{y_cm:.3f}",
                    "z_cm": f"{z_cm:.3f}",

                    # debug
                    "px_used_x": f"{best_pt[0]:.2f}",
                    "px_used_y": f"{best_pt[1]:.2f}",
                    "plane_score_px": f"{best_score:.2f}",
                })


    out = Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            w.writeheader()
            w.writerows(out_rows)

    print("[counts]", counts)
    print("[wrote]", out)

if __name__ == "__main__":
    main()
