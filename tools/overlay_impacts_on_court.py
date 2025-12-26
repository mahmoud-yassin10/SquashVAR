#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import yaml
import cv2
import numpy as np

def load_calib(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    H = np.array(cfg["homography"], dtype=np.float64)         # px -> world
    Hinv = np.linalg.inv(H)                                   # world -> px
    poly = np.array(cfg.get("pixel_points", []), dtype=np.int32).reshape(-1,1,2)
    return H, Hinv, poly

def warp_pt(pt, H):
    p = np.array([[pt]], dtype=np.float64)  # (1,1,2)
    out = cv2.perspectiveTransform(p, H)[0][0]
    return float(out[0]), float(out[1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--impacts_csv", required=True)   # nano_impacts output
    ap.add_argument("--world_csv", default=None)      # optional: for z_cm on wall
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=600)
    ap.add_argument("--draw_detections", action="store_true") # optional
    args = ap.parse_args()

    # Load calibs (Hinv used to project world->pixel)
    Hw, Hw_inv, poly_fw = load_calib("/workspace/configs/calib.frontwall.yaml")
    Hf, Hf_inv, poly_fl = load_calib("/workspace/configs/calib.floor.yaml")
    Hl, Hl_inv, poly_lw = load_calib("/workspace/configs/calib.leftwall.yaml")
    Hr, Hr_inv, poly_rw = load_calib("/workspace/configs/calib.rightwall.yaml")

    polys = [
        ("front", poly_fw),
        ("floor", poly_fl),
        ("left",  poly_lw),
        ("right", poly_rw),
    ]

    # Load impacts by frame
    impacts_by_frame = {}
    with open(args.impacts_csv, "r") as f:
        rd = csv.DictReader(f)
        for r in rd:
            fr = int(float(r["frame_idx"]))
            impacts_by_frame.setdefault(fr, []).append(r)

    # Optional: load world_csv (to get z_cm if you want front-wall height markers)
    world_by_frame = {}
    if args.world_csv:
        with open(args.world_csv, "r") as f:
            rd = csv.DictReader(f)
            for r in rd:
                fr = int(float(r["frame_idx"]))
                world_by_frame.setdefault(fr, []).append(r)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    fr = args.start

    for _ in range(args.max_frames):
        ok, frame = cap.read()
        if not ok:
            break

        # draw polygons (same style as your calib debug)
        for _, poly in polys:
            if poly is not None and len(poly) >= 3:
                cv2.polylines(frame, [poly], True, (255,255,255), 2)

        # optional: draw detections as tiny dots (OFF by default)
        if args.draw_detections and fr in world_by_frame:
            for r in world_by_frame[fr]:
                cx = int(float(r["cx_px"]))
                cy = int(float(r["cy_px"]))
                cv2.circle(frame, (cx, cy), 2, (200,200,200), -1)

        # draw impacts projected onto the correct plane
        if fr in impacts_by_frame:
            for ev in impacts_by_frame[fr]:
                X = float(ev["X_court"])
                Y = float(ev["Y_court"])
                surf = (ev["surface"] or "").upper()
                itype = ev.get("impact_type", "")

                # default: project to floor using (u=y, v=x)
                px, py = None, None

                if surf == "FLOOR":
                    u, v = (Y, X)               # floor plane expects (u=y, v=x)
                    px, py = warp_pt((u, v), Hf_inv)

                elif surf in ("FRONT", "FRONT_WALL"):
                    # front wall plane expects (u=y, v=z). If you don't have z, mark near seam.
                    z = 60.0  # fallback height marker (cm)
                    u, v = (Y, z)
                    px, py = warp_pt((u, v), Hw_inv)

                elif surf == "LEFT":
                    # left wall expects (u=x, v=z)
                    z = 60.0
                    u, v = (X, z)
                    px, py = warp_pt((u, v), Hl_inv)

                elif surf == "RIGHT":
                    z = 60.0
                    u, v = (X, z)
                    px, py = warp_pt((u, v), Hr_inv)

                else:
                    # unknown surface: just skip
                    continue

                # draw marker
                if px is not None:
                    px_i = int(np.clip(px, 0, W-1))
                    py_i = int(np.clip(py, 0, H-1))
                    cv2.circle(frame, (px_i, py_i), 8, (0,0,255), -1)
                    cv2.putText(frame, f"{surf}", (px_i+10, py_i),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"frame={fr}", (20, H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        vw.write(frame)
        fr += 1

    vw.release()
    cap.release()
    print("Wrote", out_path)

if __name__ == "__main__":
    main()
