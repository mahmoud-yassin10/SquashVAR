#!/usr/bin/env python3
import argparse, csv, math
from collections import defaultdict
import cv2


def load_events(csv_path, fps):
    """
    Load events from impacts_labeled.csv.

    Preferred:
      - Use integer frame_idx column if present.
    Fallback:
      - Derive frame_idx from t_sec and fps.
    """
    per_frame = defaultdict(list)
    events_seq = []

    with open(csv_path, newline="") as f:
        rd = csv.DictReader(f)
        fieldnames = rd.fieldnames or []
        has_frame_idx = "frame_idx" in fieldnames

        for r in rd:
            # Determine frame index
            fi = None
            t = None

            if has_frame_idx:
                try:
                    fi = int(r["frame_idx"])
                except Exception:
                    fi = None

            # Time stamp
            try:
                t = float(r.get("t_sec", "nan"))
            except Exception:
                t = None

            # Fallback to t_sec if frame_idx is missing or invalid
            if fi is None:
                if t is None or math.isnan(t):
                    continue
                fi = int(round(t * fps))
            else:
                # If t is missing, reconstruct from frame and fps
                if t is None or math.isnan(t):
                    t = fi / fps

            per_frame[fi].append(r)
            events_seq.append((fi, t, r))

    # Sort globally by time (t) if available, else by frame index
    events_seq.sort(key=lambda x: (float("inf") if x[1] is None else x[1], x[0]))
    return per_frame, events_seq


def compute_speeds(events_seq, fps, max_speed_kmh=300.0, max_dt=0.12):
    """
    Compute speed in cm/s from world coords using all events, per track_id.

    max_dt is the maximum allowed time gap between two consecutive events on
    the same track (in seconds). With fps≈60, 0.12 s ≈ 7 frames.
    """

    from collections import defaultdict

    MIN_MOVE_CM = 10.0  # ignore moves smaller than this (cm)

    tracks = defaultdict(list)
    for fi, t, r in events_seq:
        try:
            tid = int(float(r.get("track_id", "-1")))
        except Exception:
            tid = -1
        tracks[tid].append((fi, t, r))

    frame_speed = {}
    max_speed_cm_s = max_speed_kmh / 0.036  # 1 cm/s = 0.036 km/h

    for tid, seq in tracks.items():
        seq.sort(key=lambda x: (float("inf") if x[1] is None else x[1], x[0]))
        prev = None

        for fi, t, r in seq:
            if prev is None:
                prev = (fi, t, r)
                continue

            fi0, t0, r0 = prev

            # Prefer time from t_sec; if missing, fall back to frame delta
            if (t is not None) and (t0 is not None) and not math.isnan(t) and not math.isnan(t0):
                dt = t - t0
            else:
                dt = (fi - fi0) / fps

            # Ignore very small or too large gaps
            if dt <= 0 or dt > max_dt:
                prev = (fi, t, r)
                continue

            try:
                x0 = float(r0["x_cm"]); y0 = float(r0["y_cm"])
                x1 = float(r["x_cm"]);  y1 = float(r["y_cm"])
            except Exception:
                prev = (fi, t, r)
                continue

            # Use full 3D distance (x,y,z). If z is missing, fall back to 2D.
            try:
                z0 = float(r0.get("z_cm", "nan"))
                z1 = float(r.get("z_cm", "nan"))
            except Exception:
                z0 = z1 = float("nan")

            if math.isnan(z0) or math.isnan(z1):
                dist = math.hypot(x1 - x0, y1 - y0)  # 2D fallback (cm)
            else:
                dx = x1 - x0
                dy = y1 - y0
                dz = z1 - z0
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)  # cm
            if dist < MIN_MOVE_CM:
                prev = (fi, t, r)
                continue
            speed_cm_s = dist / dt


            # Ignore physically impossible speeds
            if speed_cm_s > max_speed_cm_s:
                prev = (fi, t, r)
                continue

            start = min(fi0, fi)
            end   = max(fi0, fi)
            for f in range(start, end + 1):
                frame_speed[f] = speed_cm_s

            prev = (fi, t, r)

    if frame_speed:
        speeds_kmh = [v * 0.036 for v in frame_speed.values()]
        print(f"[overlay] speed stats (km/h): min={min(speeds_kmh):.1f}, max={max(speeds_kmh):.1f}")
    else:
        print("[overlay] no valid speeds computed")

    return frame_speed


def best_event_for_frame(events):
    """
    If multiple events in a frame, choose the one with highest conf.
    """
    if not events:
        return None
    return max(events, key=lambda r: float(r.get("conf", "0.0") or 0.0))


def call_to_impact(call, wall_guess, plane):
    """
    Map 'call' + 'wall_guess' + 'plane' to an impact description string.
    """
    call = call or ""
    wall_guess = (wall_guess or "").lower()
    plane = (plane or "").lower()

    if plane == "floor" or call == "IN_FLOOR":
        return "Floor"
    if wall_guess == "front" or "FRONT" in call:
        return "Front wall"
    if wall_guess == "left" or "LEFT" in call:
        return "Left wall"
    if wall_guess == "right" or "RIGHT" in call:
        return "Right wall"
    if wall_guess == "back" or "BACK" in call:
        return "Back wall"
    return "None"


def main(args):
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # If user passes --fps > 0, we treat it as the FPS used to build t_sec & frame_idx
    if args.fps > 0:
        fps_for_mapping = float(args.fps)
    elif video_fps > 0:
        fps_for_mapping = float(video_fps)
    else:
        fps_for_mapping = 60.0  # fallback

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    per_frame, events_seq = load_events(args.impacts_csv, fps_for_mapping)
    frame_speed_cm_s = compute_speeds(events_seq, fps_for_mapping)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = video_fps if video_fps > 0 else fps_for_mapping
    out = cv2.VideoWriter(args.out_video, fourcc, out_fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Debug: how many frames have events?
    if per_frame:
        all_fis = sorted(per_frame.keys())
        print(f"[overlay] events frames: {len(all_fis)} unique frames, min={all_fis[0]}, max={all_fis[-1]}")
    else:
        print("[overlay] WARNING: no events loaded from CSV")

    frame_idx = 0
    game_status = "IN"          # IN until any OUT_* happens
    last_speed_cm_s = None      # carry speed forward between frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        events = per_frame.get(frame_idx, [])
        best = best_event_for_frame(events)

        # ---------- Impact (show ONLY on frames with events) ----------
        if best is not None:
            call = best.get("call", "")
            wall_guess = best.get("wall_guess", "")
            plane = best.get("plane", "")

            if call.startswith("OUT_"):
                game_status = "OUT"
            elif call.startswith("IN_") and game_status != "OUT":
                game_status = "IN"

            impact_desc = call_to_impact(call, wall_guess, plane)
            impact_text = f"Impact: {impact_desc}"
        else:
            impact_text = "Impact: --"

        # ---------- Speed (visible once we have at least one measurement) ----------
        if frame_idx in frame_speed_cm_s:
            last_speed_cm_s = frame_speed_cm_s[frame_idx]

        if last_speed_cm_s is not None:
            speed_kmh = last_speed_cm_s * 0.036
            speed_text = f"Speed: {speed_kmh:5.1f} km/h"
        else:
            speed_text = "Speed: --"

        # ---------- Status (IN green, OUT red) ----------
        if game_status == "OUT":
            status_text = "Status: OUT"
            status_color = (0, 0, 255)
        else:
            status_text = "Status: IN"
            status_color = (0, 255, 0)

        # ---------- Draw HUD ----------
        y0 = 40
        dy = 30

        cv2.putText(frame, status_text, (20, y0), font, 0.9, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, impact_text, (20, y0 + dy), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, speed_text,  (20, y0 + 2*dy), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[overlay] fps_for_mapping={fps_for_mapping:.2f}, fps_out={out_fps:.2f}")
    print(f"[overlay] wrote video with HUD -> {args.out_video}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="input video (e.g., tracked 1080p_3k.avi)")
    p.add_argument("--impacts_csv", required=True, help="impacts_labeled.csv path")
    p.add_argument("--out_video", required=True, help="output video with overlay")
    # If 0, use video FPS; for t_sec-derived mapping this must match impacts_from_labels
    p.add_argument("--fps", type=float, default=0.0,
                   help="FPS used to compute t_sec/frame_idx; 0 = auto from video")
    args = p.parse_args()
    main(args)
