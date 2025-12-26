import os
import glob
import re
import csv
import math
from collections import defaultdict, namedtuple

import numpy as np

TrackPoint = namedtuple("TrackPoint", ["frame", "u", "v", "X", "Y"])
PLANE_BY_TF = {}
CONF_BY_TF = {}


# ------------ CONFIG (tune later) --------------
CFG = {
    "fps": 60.0,
    "smooth_window": 5,
    "min_track_len": 6,

    # WALL: stricter so fewer fake front-wall hits
    "A_THRESH_WALL": 120.0,
    "R_DROP_WALL": 0.6,
    "THETA_THRESH": math.radians(40),
    "A_THRESH_FRONT": 120.0,
    "R_DROP_FRONT": 0.6,
    "THETA_FRONT": math.radians(40),

    # FLOOR: accel + speed-drop are soft filters; main gate is vertical max + plane=='floor'
    "A_THRESH_FLOOR": 0.2,
    "R_DROP_FLOOR": 0.95,

    "V_MIN_BEFORE": 1.0,
    "GAP_MIN": 6,

    "D_FRONT": 20.0,   # cm band near walls
    "D_BACK": 20.0,
    "D_SIDE": 20.0,

    "NO_DECISION_BAND": 0.02,
    "FSEG_LEN_MIN": 1,
"FSEG_LEN_MAX": 8,          # start with 6; try 8 if still underdetecting
"FSEG_MIN_CONF": 0.35,      # set 0.0 to disable confidence gating
"FSEG_MIN_DOWN_PX": 2.0,    # was effectively 0.01; make it real pixels
"FSEG_MIN_UP_PX": 2.0,
"FSEG_DEBOUNCE": 6,         # extra debounce frames (in addition to GAP_MIN)

}



COURT = {
    "front_x": 0.0,
    "back_x": 975.0,
    "left_y": 0.0,
    "right_y": 640.0,
}



# ------------------------------------------------


def load_homography(path):
    """
    Expect a 3x3 homography matrix saved as .npy, or a YAML you parse yourself.
    """
    return np.load(path)  # adjust if YAML


def apply_homography(u, v, H):
    """
    Map image pixel coords (u, v) to court coords (X, Y) using homography H.
    If you don't have H yet, temporarily return (u, v) and treat 'Y' as depth.
    """
    if H is None:
        return float(u), float(v)

    p = np.array([u, v, 1.0], dtype=float)
    q = H @ p
    X = q[0] / q[2]
    Y = q[1] / q[2]
    return float(X), float(Y)
def load_tracks_from_dir(labels_dir, H=None):
    tracks = defaultdict(list)
    paths = sorted(glob.glob(os.path.join(labels_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt files in {labels_dir}")

    for path in paths:
        fname = os.path.basename(path)
        m = re.search(r"(\d+)(?=\.txt$)", fname)
        frame = int(m.group(1)) if m else 0

        with open(path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            continue

        best = None
        best_conf = -1.0

        for line in lines:
            parts = line.split()

            # 6-col: cls cx cy w h conf
            if len(parts) == 6:
                cls, cx, cy, w, h, conf = parts
                tid = 0
                conf = float(conf)
                cx = float(cx); cy = float(cy)

            # 7-col (YOUR FORMAT): cls cx cy w h conf tid
            elif len(parts) == 7:
                cls, cx, cy, w, h, conf, tid_raw = parts
                tid = int(float(tid_raw))
                conf = float(conf)
                cx = float(cx); cy = float(cy)

            else:
                continue

            if conf > best_conf:
                best_conf = conf
                best = (tid, cx, cy)

        if best is None:
            continue

        tid, cx, cy = best
        u = cx
        v = cy
        X, Y = apply_homography(u, v, H)
        tracks[tid].append(TrackPoint(frame=frame, u=u, v=v, X=X, Y=Y))

    for tid in list(tracks.keys()):
        tracks[tid] = sorted(tracks[tid], key=lambda p: p.frame)

    return tracks


def load_tracks_from_world_csv(csv_path):
    """
    Load per-frame world coordinates (x_cm,y_cm,z_cm) + image coords from a CSV and build tracks.

    Expected columns (minimum):
      - frame_idx
      - x_cm, y_cm, z_cm
      - cx_px, cy_px
      - plane
    Optional:
      - track_id
    """
    global PLANE_BY_TF

    tracks = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame_idx"])

            x_cm = float(row["x_cm"])
            y_cm = float(row["y_cm"])
            # z_cm exists but is not useful on the floor plane (always 0), so we won’t rely on it
            z_cm = float(row.get("z_cm", 0.0))

            cx_px = float(row["cx_px"])
            cy_px = float(row["cy_px"])

            plane = row.get("plane", "").lower()  # "floor" or "wall"
            conf = float(row.get("conf", 0.0))

            if "track_id" in row and row["track_id"] != "":
                tid = int(float(row["track_id"]))
            else:
                tid = 0  # single track if not provided

            # store plane for this (track, frame)
            PLANE_BY_TF[(tid, frame)] = plane
            CONF_BY_TF[(tid, frame)] = conf
            # For world mode:
            #   - X, Y = court-plane trajectory (cm)
            #   - v_dummy = cy_px (image vertical in pixels)
            u_dummy = cx_px          # not used much, but numeric
            v_dummy = cy_px          # image vertical – used for floor bounces

            tracks[tid].append(
                TrackPoint(frame=frame, u=u_dummy, v=v_dummy, X=x_cm, Y=y_cm)
            )

    # sort each track by frame index
    for tid in tracks:
        tracks[tid] = sorted(tracks[tid], key=lambda p: p.frame)

    return tracks




def load_tracks_txt(txt_path, H=None):
    """
    Parse one track txt file into: dict[track_id] -> list[TrackPoint], sorted by frame.
    Format assumed: frame_id track_id x1 y1 x2 y2 conf cls
    """
    tracks = defaultdict(list)

    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            frame = int(float(parts[0]))
            tid = int(float(parts[1]))
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])

            # bbox center as image coords
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            X, Y = apply_homography(u, v, H)
            tracks[tid].append(TrackPoint(frame=frame, u=u, v=v, X=X, Y=Y))

    # sort by frame
    for tid in list(tracks.keys()):
        tracks[tid] = sorted(tracks[tid], key=lambda p: p.frame)

    return tracks


def moving_average(values, window):
    assert window % 2 == 1
    k = window // 2
    n = len(values)
    out = [values[i] for i in range(n)]
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        out[i] = sum(values[lo:hi]) / (hi - lo)
    return out


def wrap_angle(a):
    """Wrap angle to (-pi, pi]."""
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def compute_kinematics(track, cfg):
    """
    Given a list[TrackPoint] for one track, compute smoothed positions + vel/accel/heading.
    Returns dict with arrays indexed by i (0..n-1).
    """
    fps = cfg["fps"]
    dt = 1.0 / fps
    n = len(track)

    frames = [p.frame for p in track]
    Xs = [p.X for p in track]
    Ys = [p.Y for p in track]
    us = [p.u for p in track]
    vs = [p.v for p in track]

    # smooth
    w = cfg["smooth_window"]
    if w > 1 and n >= w:
        Xs_s = moving_average(Xs, w)
        Ys_s = moving_average(Ys, w)
        us_s = moving_average(us, w)
        vs_s = moving_average(vs, w)
    else:
        Xs_s, Ys_s, us_s, vs_s = Xs, Ys, us, vs

    # velocity & acceleration (central differences where possible)
    vx = [0.0] * n
    vy = [0.0] * n
    speed = [0.0] * n
    ax = [0.0] * n
    ay = [0.0] * n
    amag = [0.0] * n
    theta = [0.0] * n
    dtheta = [0.0] * n

    for i in range(1, n - 1):
        vx[i] = (Xs_s[i + 1] - Xs_s[i - 1]) / (2 * dt)
        vy[i] = (Ys_s[i + 1] - Ys_s[i - 1]) / (2 * dt)
        speed[i] = math.hypot(vx[i], vy[i])
        ax[i] = (Xs_s[i + 1] - 2 * Xs_s[i] + Xs_s[i - 1]) / (dt * dt)
        ay[i] = (Ys_s[i + 1] - 2 * Ys_s[i] + Ys_s[i - 1]) / (dt * dt)
        amag[i] = math.hypot(ax[i], ay[i])
        theta[i] = math.atan2(vy[i], vx[i])

    for i in range(2, n - 1):
        dtheta[i] = wrap_angle(theta[i] - theta[i - 1])

    return {
        "frames": frames,
        "Xs": Xs_s,
        "Ys": Ys_s,
        "us": us_s,
        "vs": vs_s,
        "vx": vx,
        "vy": vy,
        "speed": speed,
        "ax": ax,
        "ay": ay,
        "amag": amag,
        "theta": theta,
        "dtheta": dtheta,
    }


def classify_surface(X, Y, court, cfg):
    """
    Classify which court boundary (FRONT, BACK, LEFT, RIGHT) this (X,Y) is closest to.

    X, Y are in court floor coordinates (cm).
    """
    x = X
    y = Y

    df = abs(x - court["front_x"])   # distance to front wall line
    db = abs(x - court["back_x"])    # distance to back wall line
    dl = abs(y - court["left_y"])    # distance to left wall
    dr = abs(y - court["right_y"])   # distance to right wall

    # pick nearest wall
    dmin = min(df, db, dl, dr)

    if dmin == df:
        return "FRONT", df
    elif dmin == db:
        return "BACK", db
    elif dmin == dl:
        return "LEFT", dl
    else:
        return "RIGHT", dr




def detect_impacts_for_track(track_id, track, cfg, court):
    if len(track) < cfg["min_track_len"]:
        return []

    kin = compute_kinematics(track, cfg)

    frames = kin["frames"]
    Xs = kin["Xs"]
    Ys = kin["Ys"]
    vs = kin["vs"]          # in world_csv mode: cy_px
    speed = kin["speed"]
    amag = kin["amag"]
    dtheta = kin["dtheta"]

    impacts = []
    last_impact_frame = -10**9

    for i in range(2, len(frames) - 2):
        frame = frames[i]
        if frame - last_impact_frame < cfg["GAP_MIN"]:
            continue

        v_before = speed[i - 1]
        v_after  = speed[i + 1]
        if v_before < cfg["V_MIN_BEFORE"]:
            continue

        a_here = amag[i]
        dth = abs(dtheta[i])
        speed_drop_ratio = v_after / (v_before + 1e-6)

        X, Y = Xs[i], Ys[i]
        surf, d_surf = classify_surface(X, Y, court, cfg)

        # wall-near gate
        if surf in ("FRONT", "BACK"):
            is_wall_near = d_surf < (cfg["D_FRONT"] if surf == "FRONT" else cfg["D_BACK"])
        else:
            is_wall_near = d_surf < cfg["D_SIDE"]

        # front-wall can have separate thresholds
        if surf == "FRONT":
            a_thr = cfg.get("A_THRESH_FRONT", cfg["A_THRESH_WALL"])
            r_thr = cfg.get("R_DROP_FRONT", cfg["R_DROP_WALL"])
            t_thr = cfg.get("THETA_FRONT", cfg["THETA_THRESH"])
        else:
            a_thr = cfg["A_THRESH_WALL"]
            r_thr = cfg["R_DROP_WALL"]
            t_thr = cfg["THETA_THRESH"]

        wall_hit = (
            is_wall_near
            and a_here > a_thr
            and speed_drop_ratio < r_thr
            and dth > t_thr
        )

        # floor bounce (local MAX of cy_px) when plane=='floor'
        v_prev = vs[i - 1]
        v_here = vs[i]
        v_next = vs[i + 1]
        is_local_max = (v_here >= v_prev and v_here >= v_next)

        down = v_here - v_prev
        up   = v_here - v_next

        plane_here = PLANE_BY_TF.get((track_id, frame), None)

        floor_hit = (
            plane_here == "floor"
            and is_local_max
            and down >= cfg["FSEG_MIN_DOWN_PX"]
            and up   >= cfg["FSEG_MIN_UP_PX"]
            and a_here > cfg["A_THRESH_FLOOR"]
            and speed_drop_ratio < cfg["R_DROP_FLOOR"]
        )

        if wall_hit:
            impacts.append({
                "track_id": track_id,
                "frame": frame,
                "X": X,
                "Y": Y,
                "surface": surf,
                "impact_type": "WALL_IMPACT",
            })

            # corner double-hit: count BOTH surfaces when near front+side
            near_front = abs(X - court["front_x"]) < cfg["D_FRONT"]
            near_left  = abs(Y - court["left_y"])  < cfg["D_SIDE"]
            near_right = abs(Y - court["right_y"]) < cfg["D_SIDE"]

            if near_front and near_left and surf != "LEFT":
                impacts.append({
                    "track_id": track_id,
                    "frame": frame,
                    "X": X,
                    "Y": Y,
                    "surface": "LEFT",
                    "impact_type": "WALL_IMPACT_CORNER",
                })
            if near_front and near_right and surf != "RIGHT":
                impacts.append({
                    "track_id": track_id,
                    "frame": frame,
                    "X": X,
                    "Y": Y,
                    "surface": "RIGHT",
                    "impact_type": "WALL_IMPACT_CORNER",
                })

            last_impact_frame = frame
            continue

        if floor_hit:
            impacts.append({
                "track_id": track_id,
                "frame": frame,
                "X": X,
                "Y": Y,
                "surface": "FLOOR",
                "impact_type": "FLOOR_BOUNCE",
            })
            last_impact_frame = frame
            continue

    # EXTRA FLOOR PASS: contiguous plane=="floor" segments (helps missed bounces)
    existing_frames = set(e["frame"] for e in impacts)
    planes = [PLANE_BY_TF.get((track_id, fr), None) for fr in frames]

    segs = []
    k = 0
    while k < len(frames):
        if planes[k] == "floor":
            s = k
            while k + 1 < len(frames) and planes[k + 1] == "floor":
                k += 1
            e = k
            segs.append((s, e))
        k += 1

    for (s, e) in segs:
        L = e - s + 1
        if L < cfg["FSEG_LEN_MIN"] or L > cfg["FSEG_LEN_MAX"]:
            continue

        j = max(range(s, e + 1), key=lambda t: vs[t])
        fr = frames[j]

        if any(abs(fr - f0) < (cfg["GAP_MIN"] + cfg["FSEG_DEBOUNCE"]) for f0 in existing_frames):
            continue
        if j - 1 < 0 or j + 1 >= len(frames):
            continue

        down = vs[j] - vs[j - 1]
        up   = vs[j] - vs[j + 1]
        if down < cfg["FSEG_MIN_DOWN_PX"] or up < cfg["FSEG_MIN_UP_PX"]:
            continue

        c = CONF_BY_TF.get((track_id, fr), 1.0)
        if c < cfg["FSEG_MIN_CONF"]:
            continue

        X, Y = Xs[j], Ys[j]
        impacts.append({
            "track_id": track_id,
            "frame": fr,
            "X": X,
            "Y": Y,
            "surface": "FLOOR",
            "impact_type": "FLOOR_BOUNCE_SEG",
        })
        existing_frames.add(fr)

    return impacts



def write_events_csv(events, out_path, fps):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "match_id",
                "rally_id",
                "track_id",
                "frame_idx",
                "t_sec",
                "X_court",
                "Y_court",
                "surface",
                "impact_type",
            ]
        )
        for e in events:
            # fill match_id / rally_id later or pass them into this function
            t_sec = e["frame"] / fps
            w.writerow(
                [
                    "match0",
                    "rally0",
                    e["track_id"],
                    e["frame"],
                    f"{t_sec:.4f}",
                    f"{e['X']:.4f}",
                    f"{e['Y']:.4f}",
                    e["surface"],
                    e["impact_type"],
                ]
            )



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks_txt", default=None,
                        help="single YOLO track txt (combined file)")
    parser.add_argument("--labels_dir", default=None,
                        help="directory of per-frame YOLO txts")
    parser.add_argument("--world_csv", default=None,
                        help="CSV with frame_idx,x_cm,y_cm,z_cm (CourtGeometry output)")
    parser.add_argument("--H", default=None,
                        help="path to homography .npy (used only with tracks/labels)")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    H = load_homography(args.H) if args.H is not None else None

    # Priority: world_csv → tracks_txt → labels_dir
    if args.world_csv is not None:
        tracks = load_tracks_from_world_csv(args.world_csv)
    elif args.tracks_txt is not None:
        tracks = load_tracks_txt(args.tracks_txt, H=H)
    elif args.labels_dir is not None:
        tracks = load_tracks_from_dir(args.labels_dir, H=H)
    else:
        raise ValueError("Provide one of --world_csv, --tracks_txt, or --labels_dir")

    all_events = []
    for tid, track in tracks.items():
        ev = detect_impacts_for_track(tid, track, CFG, COURT)
        all_events.extend(ev)

    write_events_csv(all_events, args.out_csv, CFG["fps"])
    print(f"wrote {len(all_events)} impacts to {args.out_csv}")


if __name__ == "__main__":
    main()
