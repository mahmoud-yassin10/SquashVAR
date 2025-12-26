#!/usr/bin/env python3
import argparse, csv, math, pathlib, sys, re
from collections import Counter

# ---------------------------------------------------------------------
# Import CourtGeometry from src/geom/coords.py
# ---------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[0]  # -> /workspace/tools
sys.path.append(str(ROOT.parent / "src"))           # add /workspace/src

from geom.coords import CourtGeometry


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--frame_w", type=int, required=True)
    p.add_argument("--frame_h", type=int, required=True)
    return p.parse_args()


def parse_label_line(line):
    """
    Support both YOLO formats:
      6 fields: cls cx cy w h conf
      7+fields: cls cx cy w h conf track_id ...
    """
    parts = line.strip().split()
    if len(parts) < 6:
        return None

    cls_id = int(float(parts[0]))
    cx_n   = float(parts[1])
    cy_n   = float(parts[2])
    w_n    = float(parts[3])
    h_n    = float(parts[4])
    conf   = float(parts[5])

    if len(parts) >= 7:
        try:
            track_id = int(float(parts[6]))
        except Exception:
            track_id = -1
    else:
        track_id = -1

    return cls_id, cx_n, cy_n, w_n, h_n, conf, track_id


def iter_label_files(labels_dir: pathlib.Path):
    """
    Yield (frame_idx, path) for every *.txt label file.

    We are robust to different naming schemes:
      - "000000.txt"
      - "frame000123.txt"
      - "1080p_3k_000123.txt"
    We take the LAST run of digits in the stem as the frame index.
    If there are no digits at all, we fall back to an increasing counter.
    """
    labels_dir = pathlib.Path(labels_dir)
    files = sorted(labels_dir.glob("*.txt"))
    frame_counter = 0
    for f in files:
        stem = f.stem
        m = re.search(r"(\d+)$", stem)
        if m:
            frame_idx = int(m.group(1))
        else:
            frame_idx = frame_counter
        frame_counter += 1
        yield frame_idx, f


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    labels_dir = pathlib.Path(args.labels_dir)
    out_csv    = pathlib.Path(args.out_csv)
    fps        = float(args.fps)
    W          = int(args.frame_w)
    H          = int(args.frame_h)

    if not labels_dir.is_dir():
        print(f"[impacts_from_labels] ERROR: labels_dir not found: {labels_dir}")
        return

    cg = CourtGeometry()  # uses your calib.frontwall.yaml & calib.floor.yaml

    rows = []

    num_files = 0
    num_lines = 0
    num_bad_lines = 0

    for frame_idx, path in iter_label_files(labels_dir):
        num_files += 1
        t_sec = frame_idx / fps

        with path.open() as fh:
            for line in fh:
                num_lines += 1
                parsed = parse_label_line(line)
                if parsed is None:
                    num_bad_lines += 1
                    continue

                cls_id, cx_n, cy_n, w_n, h_n, conf, track_id = parsed

                cx_px = cx_n * W
                cy_px = cy_n * H
                w_px  = w_n * W
                h_px  = h_n * H

                # Map to court geometry
                plane, (x_cm, y_cm, z_cm), err = cg.px_to_court_xyz((cx_px, cy_px))

                # Ignore clearly "air" points
                if plane == "air":
                    continue

                row = {
                    "frame_idx": frame_idx,
                    "t_sec": f"{t_sec:.6f}",
                    "cls_id": cls_id,
                    "cx_px": f"{cx_px:.2f}",
                    "cy_px": f"{cy_px:.2f}",
                    "w_px":  f"{w_px:.2f}",
                    "h_px":  f"{h_px:.2f}",
                    "conf":  f"{conf:.4f}",
                    "track_id": track_id,
                    "plane": plane,
                    "x_cm": f"{x_cm:.3f}",
                    "y_cm": f"{y_cm:.3f}",
                    "z_cm": f"{z_cm:.3f}",
                    "reproj_err_px": f"{err:.3f}",
                }
                rows.append(row)

    if not rows:
        print(f"[impacts_from_labels] WARNING: no usable detections")
        print(f"[impacts_from_labels] files={num_files}, lines={num_lines}, bad_lines={num_bad_lines}")
        return

    plane_counts = Counter(r["plane"] for r in rows)

    fieldnames = [
        "frame_idx", "t_sec",
        "cls_id", "cx_px", "cy_px", "w_px", "h_px",
        "conf", "track_id",
        "plane", "x_cm", "y_cm", "z_cm",
        "reproj_err_px",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[impacts_from_labels] rows = {len(rows)}  |  plane_counts = {dict(plane_counts)}")
    print(f"[impacts_from_labels] files read = {num_files}, label lines = {num_lines}, bad_lines = {num_bad_lines}")


if __name__ == "__main__":
    main()
