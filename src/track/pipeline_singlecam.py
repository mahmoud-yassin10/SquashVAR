# src/track/pipeline_singlecam.py
import argparse, csv, math, os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

from src.geom.coords import CourtGeometry, speed_cm_per_s  # keep if you use elsewhere
from src.fusion.audio_onset import detect_onsets

# ----------------------------
# helpers
# ----------------------------
def in_bounds(plane, x_cm, y_cm):
    if x_cm is None or y_cm is None:
        return False
    if plane == "wall":
        return 0 <= x_cm <= 640 and 0 <= y_cm <= 457
    if plane == "floor":
        return 0 <= x_cm <= 640 and 0 <= y_cm <= 975
    return False

def px_to_world(cg: CourtGeometry, plane, px):
    if plane == "wall":
        x_cm, y_cm = cg.px_to_wall(px)
        x_cm = min(max(x_cm, 0.0), 640.0)
        y_cm = min(max(y_cm, 0.0), 457.0)
    elif plane == "floor":
        x_cm, y_cm = cg.px_to_floor(px)
        x_cm = min(max(x_cm, 0.0), 640.0)
        y_cm = min(max(y_cm, 0.0), 975.0)
    else:
        return None, None
    return float(x_cm), float(y_cm)

def _safe_plane_call(fn, px, tol_px):
    """Call cg.is_on_wall / cg.is_on_floor robustly."""
    try:
        return bool(fn(px))
    except TypeError:
        try:
            return bool(fn(px, tol_px))
        except Exception:
            return False
    except Exception:
        return False

def choose_plane(cg: CourtGeometry, px, tol_px):
    on_wall  = _safe_plane_call(getattr(cg, "is_on_wall", lambda *_: False), px, tol_px)
    on_floor = _safe_plane_call(getattr(cg, "is_on_floor", lambda *_: False), px, tol_px)
    if on_wall and not on_floor: return "wall"
    if on_floor and not on_wall: return "floor"
    if on_wall and on_floor:     return "wall"  # tie-break
    return None

def compute_speed(prev, cur, fps):
    """Return (v_cm_s, v_kmh) or None if not computable."""
    if prev is None or cur is None:
        return None
    if prev["plane"] != cur["plane"]:
        return None
    if cur["frame"] - prev["frame"] != 1:
        return None
    dt = 1.0 / fps
    d_cm = math.hypot(cur["x_cm"] - prev["x_cm"], cur["y_cm"] - prev["y_cm"])
    v_cms = d_cm / dt
    v_kmh = v_cms * 0.036
    return v_cms, v_kmh

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

# ----------------------------
# main
# ----------------------------
def main(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[video] {args.source} | {W}x{H} @ {fps:.3f} fps | frames={total_frames}")

    # audio onsets (robust unpacking)
    onset_frames = set()
    if args.wav and os.path.exists(args.wav):
        res = detect_onsets(args.wav)
        if isinstance(res, tuple):
            if len(res) >= 2: sr, onsets_s = res[0], res[1]
            elif len(res) == 1: sr, onsets_s = None, res[0]
            else: sr, onsets_s = None, []
        else:
            sr, onsets_s = None, res
        if onsets_s is None: onsets_s = []
        try: onsets_s = list(onsets_s)
        except Exception: onsets_s = []
        offset_s = args.av_offset_ms / 1000.0
        for t in onsets_s:
            try:
                f = int(round((float(t) + offset_s) * fps))
                if f >= 0: onset_frames.add(f)
            except Exception:
                pass
        print(f"[audio] {args.wav} | onsets={len(onset_frames)} (offset {args.av_offset_ms} ms, window ±{args.onset_window_frames}f)")
    else:
        print("[audio] none")

    # geometry
    cg = CourtGeometry(args.frontwall_calib, args.floor_calib)
    print(f"[geom] frontwall={args.frontwall_calib} | floor={args.floor_calib}")

    # model
    model = YOLO(args.model)
    print(f"[yolo] weights={args.model} | conf={args.det_conf} | device={args.device} | half={args.half}")

    # overlay
    writer = None
    if args.save_overlay:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "overlay.mp4"), fourcc, fps, (W, H))
        print(f"[viz] writing overlay -> {out_dir/'overlay.mp4'}")

    # csv
    csv_path = out_dir / "tracks.csv"
    fcsv = open(csv_path, "w", newline="")
    cols = ["frame","t_sec","x_px","y_px","conf","plane","x_cm","y_cm","speed_cm_s","speed_kmh","event","onset_window"]
    wr = csv.DictWriter(fcsv, fieldnames=cols)
    wr.writeheader()

    prev_by_plane = {"wall": None, "floor": None}
    frame_idx = 0

    # ---------------------------------------------------------------------
    # ADDED (from your snippet): Background subtractor initialized once
    # Equivalent to: fg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    # ---------------------------------------------------------------------
    fg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # per-frame inference (robust with your Ultralytics build)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_sec = frame_idx / fps

        # -----------------------------------------------------------------
        # ADDED (from your snippet): build grayscale + motion mask per frame
        # Equivalent to: mask = fg.apply(frame_gray)  # per frame
        # -----------------------------------------------------------------
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = fg.apply(frame_gray)

        results = model.predict(
            frame,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.det_conf,
            half=args.half,
            verbose=False,
            max_det=10
        )
        res = results[0]

        # select top-1 detection
        cx = cy = None
        cscore = None
        # (keep bbox vars so we can motion-filter the detection)
        x1 = y1 = x2 = y2 = None
        if res.boxes is not None and len(res.boxes) > 0:
            confs = res.boxes.conf.detach().cpu().numpy()
            best_i = int(np.argmax(confs))
            x1,y1,x2,y2 = res.boxes.xyxy[best_i].detach().cpu().numpy().tolist()
            cscore = float(confs[best_i])
            cx = 0.5*(x1+x2); cy = 0.5*(y1+y2)

            # -------------------------------------------------------------
            # ADAPTED SNIPPET: motion check inside the detection bbox.
            # Original pseudo:
            #   m = mask[max(0,y1):y2, max(0,x1):x2]
            #   if m.size==0 or m.mean() < 5: continue  # static → drop
            # Here we "drop" by clearing cx/cy/cscore so downstream sees no det.
            # -------------------------------------------------------------
            xi1 = max(0, int(math.floor(x1)))
            yi1 = max(0, int(math.floor(y1)))
            xi2 = min(W,  int(math.ceil(x2)))
            yi2 = min(H,  int(math.ceil(y2)))
            if xi2 <= xi1 or yi2 <= yi1:
                # empty crop → treat as static
                cx = cy = cscore = None
            else:
                m = mask[yi1:yi2, xi1:xi2]
                # threshold 5 matches your snippet; tweak if needed later
                if m.size == 0 or float(m.mean()) < 5.0:
                    # static → drop detection
                    cx = cy = cscore = None

        plane = None
        x_cm = y_cm = None
        v_cms = v_kmh = None
        event = ""
        onset_flag = 0

        if cx is not None and cscore is not None and cscore >= args.det_conf:
            plane = choose_plane(cg, (cx, cy), tol_px=args.plane_tol)
            if plane:
                x_cm, y_cm = px_to_world(cg, plane, (cx, cy))
                if not in_bounds(plane, x_cm, y_cm):
                    x_cm = y_cm = None
                    plane = None

        if plane and x_cm is not None:
            cur = {"plane": plane, "x_cm": x_cm, "y_cm": y_cm, "frame": frame_idx}
            sp = compute_speed(prev_by_plane[plane], cur, fps)
            if sp is not None:
                v_cms, v_kmh = sp
                # sanity cap only if we have a float
                if v_cms is not None and v_cms >= 12000:  # 120 m/s
                    v_cms = v_kmh = None
            prev_by_plane[plane] = cur

        if onset_frames:
            hit_onset = any(abs(frame_idx - f) <= args.onset_window_frames for f in onset_frames)
            onset_flag = 1 if hit_onset else 0
            if hit_onset and plane in ("wall", "floor"):
                event = f"impact_{plane}"

        if writer is not None:
            img = frame.copy()
            if cx is not None and cscore is not None:
                cv2.circle(img, (int(cx), int(cy)), 6, (0,255,0), -1)
                cv2.putText(img, f"{cscore:.2f}", (int(cx)+8, int(cy)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            if plane and x_cm is not None:
                cv2.putText(img, f"{plane} ({x_cm:.0f},{y_cm:.0f})", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
            if v_kmh is not None:
                cv2.putText(img, f"{v_kmh:.1f} km/h", (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2, cv2.LINE_AA)
            if event:
                cv2.putText(img, event, (12, 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            writer.write(img)

        wr.writerow({
            "frame": frame_idx,
            "t_sec": f"{t_sec:.3f}",
            "x_px": f"{cx:.1f}" if cx is not None else "",
            "y_px": f"{cy:.1f}" if cy is not None else "",
            "conf": f"{cscore:.3f}" if cscore is not None else "",
            "plane": plane if plane else "",
            "x_cm": f"{x_cm:.2f}" if x_cm is not None else "",
            "y_cm": f"{y_cm:.2f}" if y_cm is not None else "",
            "speed_cm_s": f"{v_cms:.1f}" if v_cms is not None else "",
            "speed_kmh": f"{v_kmh:.1f}" if v_kmh is not None else "",
            "event": event,
            "onset_window": 1 if onset_flag else 0
        })

        frame_idx += 1

    fcsv.close()
    if writer is not None:
        writer.release()
    cap.release()
    print(f"[done] wrote {csv_path}")
    if writer:
        print(f"[done] wrote {out_dir/'overlay.mp4'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="input video")
    ap.add_argument("--wav", default="", help="optional mono wav for audio onsets")
    ap.add_argument("--model", required=True, help="YOLO weights")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--onset_window_frames", type=int, default=6)
    ap.add_argument("--av_offset_ms", type=int, default=0)
    ap.add_argument("--frontwall_calib", default="configs/calib.frontwall.yaml")
    ap.add_argument("--floor_calib", default="configs/calib.floor.yaml")
    ap.add_argument("--plane_tol", type=float, default=4.0, help="px tolerance if your API supports it")
    ap.add_argument("--save_overlay", action="store_true")
    args = ap.parse_args()
    main(args)
