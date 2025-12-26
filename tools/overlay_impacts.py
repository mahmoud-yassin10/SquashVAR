#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
import cv2

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--impacts_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--radius", type=int, default=10)
    p.add_argument("--thickness", type=int, default=2)
    p.add_argument("--show_last", type=int, default=8)  # persistence
    return p.parse_args()

def main():
    a = parse()
    impacts = {}
    with open(a.impacts_csv, newline="") as f:
        rd = csv.DictReader(f)
        # expects frame_idx + cx_px + cy_px (+ optional surface/plane)
        for r in rd:
            fi = int(float(r.get("frame_idx", -1)))
            if fi < 0: 
                continue
            impacts.setdefault(fi, []).append(r)

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {a.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps>0 else 30.0, (W,H))

    recent = []  # list of (frame_idx, x, y, label)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # add new impacts at this frame
        if frame_idx in impacts:
            for r in impacts[frame_idx]:
                x = int(float(r.get("cx_px", r.get("px_used_x", 0))))
                y = int(float(r.get("cy_px", r.get("px_used_y", 0))))
                label = r.get("surface", r.get("plane", "impact"))
                recent.append((frame_idx, x, y, label))

        # keep last N frames worth
        recent = [(fi,x,y,lbl) for (fi,x,y,lbl) in recent if frame_idx - fi <= a.show_last]

        # draw
        for (fi,x,y,lbl) in recent:
            cv2.circle(frame, (x,y), a.radius, (255,255,255), a.thickness)
            cv2.putText(frame, str(lbl), (x+12,y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.putText(frame, f"frame={frame_idx}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        vw.write(frame)
        frame_idx += 1

    cap.release()
    vw.release()
    print("Wrote", outp)

if __name__ == "__main__":
    main()
