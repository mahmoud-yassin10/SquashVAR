#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--start", type=int, default=0, help="start frame index")
    ap.add_argument("--max_frames", type=int, default=999999, help="max frames to write")
    ap.add_argument("--show_time", action="store_true", help="also overlay t_sec using FPS")
    ap.add_argument("--scale", type=float, default=1.0, help="text scale")
    ap.add_argument("--thickness", type=int, default=2, help="text thickness")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(
        str(outp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps and fps > 0 else 30.0,
        (W, H),
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    frame_idx = args.start
    written = 0

    while written < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        label = f"frame={frame_idx}"
        if args.show_time and fps and fps > 0:
            label += f"  t={frame_idx / fps:.3f}s"

        # Draw a black shadow then white text for readability
        org = (20, 50)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, args.scale, (0, 0, 0), args.thickness + 2)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, args.scale, (255, 255, 255), args.thickness)

        vw.write(frame)
        frame_idx += 1
        written += 1

    cap.release()
    vw.release()
    print("Wrote:", outp)

if __name__ == "__main__":
    main()

