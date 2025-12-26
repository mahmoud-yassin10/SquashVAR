cat > /workspace/tools/overlay_everything.py <<'PY'
import argparse, os, csv, math
import cv2
import yaml

def load_yaml_poly(path):
    y = yaml.safe_load(open(path, "r"))
    pts = y.get("pixel_points", []) or y.get("polygon", []) or y.get("poly", [])
    # Expect list of [x,y]
    return [(int(p[0]), int(p[1])) for p in pts]

def load_world(world_csv):
    # Index by (track_id, frame_idx) -> row
    # Also keep prev row per track for speed computation
    idx = {}
    by_track = {}
    with open(world_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                tid = int(float(row.get("track_id", -1)))
                fi  = int(float(row.get("frame_idx", -1)))
            except:
                continue
            idx[(tid, fi)] = row
            by_track.setdefault(tid, []).append(fi)
    for tid in by_track:
        by_track[tid].sort()
    return idx, by_track

def load_impacts(impacts_csv):
    # Expected columns include: track_id, frame_idx, surface
    impacts = []
    with open(impacts_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                tid = int(float(row.get("track_id", -1)))
                fi  = int(float(row.get("frame_idx", -1)))
            except:
                continue
            surf = (row.get("surface","") or "").upper()
            impacts.append((fi, tid, surf, row))
    impacts.sort(key=lambda x: x[0])
    return impacts

def inout_from_plane_z(plane, z_cm, band_cm=2.0):
    # Rule constants (cm)
    TIN = 48.0
    FRONT_OUT = 457.0
    SIDE_BACK_OUT = 213.0

    if z_cm is None:
        return "NO_DECISION"

    # Normalize plane names
    p = (plane or "").lower()

    def near(val, target): return abs(val - target) <= band_cm

    # Front wall: tin + out-line
    if "front" in p:
        if near(z_cm, TIN) or near(z_cm, FRONT_OUT):
            return "NO_DECISION"
        if z_cm < TIN:
            return "OUT (TIN)"
        if z_cm > FRONT_OUT:
            return "OUT"
        return "IN"

    # Side/back walls: out-line at 213 cm
    if ("left" in p) or ("right" in p) or ("back" in p):
        if near(z_cm, SIDE_BACK_OUT):
            return "NO_DECISION"
        return "OUT" if (z_cm > SIDE_BACK_OUT) else "IN"

    # Floor impacts are in by definition (for this module)
    if "floor" in p:
        return "IN"

    return "NO_DECISION"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_in", required=True)
    ap.add_argument("--video_out", required=True)
    ap.add_argument("--world_csv", required=True)
    ap.add_argument("--impacts_csv", required=True)
    ap.add_argument("--calib_front", required=True)
    ap.add_argument("--calib_floor", required=True)
    ap.add_argument("--calib_left", required=True)
    ap.add_argument("--calib_right", required=True)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--persist", type=int, default=30)
    ap.add_argument("--band_cm", type=float, default=2.0)
    args = ap.parse_args()

    # Load geometry polygons
    polys = {
        "front": load_yaml_poly(args.calib_front),
        "floor": load_yaml_poly(args.calib_floor),
        "left":  load_yaml_poly(args.calib_left),
        "right": load_yaml_poly(args.calib_right),
    }

    world_idx, by_track = load_world(args.world_csv)
    impacts = load_impacts(args.impacts_csv)
    imp_ptr = 0
    recent = []  # list of (frame_idx, track_id, surf, row)

    cap = cv2.VideoCapture(args.video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video_in}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = args.fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
    out = cv2.VideoWriter(args.video_out, fourcc, fps, (w, h))

    # For speed: track last known world point per track
    last_world = {}  # tid -> (fi, x, y, z)
    last_speed_mps = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Add impacts whose frame <= current
        while imp_ptr < len(impacts) and impacts[imp_ptr][0] <= frame_idx:
            recent.append(impacts[imp_ptr])
            imp_ptr += 1
        # Drop old impacts beyond persist
        recent = [x for x in recent if (frame_idx - x[0]) <= args.persist]

        # Draw polygons
        for name, pts in polys.items():
            if len(pts) >= 3:
                cv2.polylines(frame, [cv2.UMat(cv2.UMat(cv2.UMat)).get() if False else cv2.UMat], False, (255,255,255), 2)
        # (Workaround: OpenCV needs ndarray points)
        for name, pts in polys.items():
            if len(pts) >= 3:
                import numpy as np
                arr = np.array(pts, dtype=np.int32).reshape((-1,1,2))
                cv2.polylines(frame, [arr], True, (255,255,255), 2)

        # Compute a "current" speed from the highest-confidence world row at this frame (if any)
        best_row = None
        best_conf = -1.0
        for (tid, fi), row in world_idx.items():
            if fi != frame_idx:
                continue
            try:
                conf = float(row.get("conf", 0.0))
            except:
                conf = 0.0
            if conf > best_conf:
                best_conf = conf
                best_row = row

        if best_row is not None:
            try:
                tid = int(float(best_row.get("track_id", -1)))
                x = float(best_row.get("x_cm", "nan"))
                y = float(best_row.get("y_cm", "nan"))
                z = float(best_row.get("z_cm", "nan")) if best_row.get("z_cm") not in (None,"") else float("nan")
            except:
                tid = -1
                x=y=z=float("nan")

            prev = last_world.get(tid)
            if prev is not None:
                _, px, py, pz = prev
                if all(map(lambda v: not math.isnan(v), [x,y,px,py])):
                    dz = 0.0 if (math.isnan(z) or math.isnan(pz)) else (z - pz)
                    dist_cm = math.sqrt((x-px)**2 + (y-py)**2 + dz**2)
                    speed_mps = (dist_cm * fps) / 100.0
                    last_speed_mps = speed_mps
            last_world[tid] = (frame_idx, x, y, z)

        # Draw impacts markers + IN/OUT decision text
        import numpy as np
        for (fi, tid, surf, row) in recent:
            wrow = world_idx.get((tid, fi))
            if not wrow:
                continue
            try:
                px = int(float(wrow.get("px_used_x", wrow.get("cx_px", 0))))
                py = int(float(wrow.get("px_used_y", wrow.get("cy_px", 0))))
            except:
                continue

            plane = (wrow.get("plane","") or "").lower()
            try:
                z_cm = float(wrow.get("z_cm","nan"))
                if math.isnan(z_cm): z_cm = None
            except:
                z_cm = None

            decision = inout_from_plane_z(plane, z_cm, band_cm=args.band_cm)

            cv2.circle(frame, (px, py), 10, (0,255,255), -1)
            cv2.putText(frame, f"{decision}", (px+12, py-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

        # HUD text: frame + speed
        cv2.putText(frame, f"frame={frame_idx}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
        if last_speed_mps is not None:
            cv2.putText(frame, f"speed={last_speed_mps:.2f} m/s", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)

        out.write(frame)
        frame_idx += 1

    out.release()
    cap.release()
    print("Wrote:", args.video_out)

if __name__ == "__main__":
    main()
PY
