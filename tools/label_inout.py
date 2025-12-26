#!/usr/bin/env python3
import csv, math
from pathlib import Path

# Root run directory for this pipeline
RUN      = Path("/workspace/runs/pipeline_v2")
OUT_DIR  = RUN / "calls"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== Court config (cm) =====
COURT_LEN = 975.0   # length x (front wall at x≈0, back at x≈975)
COURT_WID = 640.0   # width  y in [0,640]
NEAR_WALL = 30.0    # how close (cm) to be considered "on that wall"

# ===== Height limits (homography units) =====
OUT_H_F   = 457.0   # front-wall out-line height (top of front wall plane)
OUT_H_BS  = 213.0   # back/side out-line height
TIN_Z_CM = 48.0      # tin height on front wall (cm)
TIN_BAND_CM = 2.0    # no-decision band around tin (like NO_DECISION_CM)

# ===== Floor logic thresholds =====
# These are now *very* conservative. Only pixels really down near the
# floor + low z are treated as "actually floor".
FLOOR_Y_MIN_PX   = 630.0  # strong pixel rule: clearly below front-wall base
FLOOR_Y_SOFT_PX  = 600.0  # softer band; must also have small z
FLOOR_Z_SOFT_CM  = 18.0   # z below this + y>SOFT => floor
FLOOR_Z_HARD_CM  = 10.0   # super low z, still need some y support
# z thresholds (cm)
FLOOR_Z_MAX = 15.0   # anything below this is essentially floor height


INP = RUN / "impacts_from_tracks.csv"
OUT = RUN / "impacts_labeled.csv"


def infer_wall(x, y):
    """
    Decide which *geometric* wall is closest given (x_cm, y_cm) on the court plane.
    front: x ≈ 0
    back:  x ≈ COURT_LEN
    left:  y ≈ 0
    right: y ≈ COURT_WID
    """
    d_front = abs(x - 0.0)
    d_back  = abs(x - COURT_LEN)
    d_left  = abs(y - 0.0)
    d_right = abs(y - COURT_WID)
    m = min(d_front, d_back, d_left, d_right)
    if m > NEAR_WALL:
        return "floor_or_mid"  # probably floor bounce or mid-air
    if m == d_front:
        return "front"
    if m == d_back:
        return "back"
    if m == d_left:
        return "left"
    return "right"


def _parse_float(r, key, default=None):
    try:
        v = float(r.get(key, "nan"))
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _is_floor_like(plane, z):
    """
    Decide if an impact should be treated as floor, based on plane + z.

    - Any explicit floor-plane hit => floor.
    - For wall-plane hits, only treat as floor if height is very close to 0.
    """
    plane = (plane or "").lower()
    if plane == "floor":
        return True
    if z is None:
        return False
    return z < FLOOR_Z_MAX



def side_out_height(x_cm):
    # linear slope from 457 at x=0 to 213 at x=975
    if x_cm is None:
        return OUT_H_BS
    t = max(0.0, min(1.0, x_cm / COURT_LEN))
    return OUT_H_F + (OUT_H_BS - OUT_H_F) * t  # 457 -> 213

NO_DECISION_CM = 2.0  # band around the line

def call_event(r):
    plane = (r.get("plane") or "").lower()

    x = _parse_float(r, "x_cm")
    y = _parse_float(r, "y_cm")
    z = _parse_float(r, "z_cm", default=None)

    # Floor-like hits
    if _is_floor_like(plane, z):
        return "IN_FLOOR"

    # --- New: explicit planes from coords.py ---
    if plane == "wall" and x is not None and y is not None:
        wall = infer_wall(x, y)
        if z is None:
            return f"IN_WALL_{wall.upper()}"

        if wall == "front":
            # --- TIN first ---
            if z < TIN_Z_CM - TIN_BAND_CM:
                return "OUT_TIN"
            if z <= TIN_Z_CM + TIN_BAND_CM:
                return "NO_DECISION_TIN"

            # --- then front out-line ---
            if z > OUT_H_F + NO_DECISION_CM:
                return "OUT_LINE_FRONT"
            if z < OUT_H_F - NO_DECISION_CM:
                return "IN_WALL_FRONT"
            return "NO_DECISION_FRONT"

        if wall in ("back", "left", "right"):
            return f"OUT_LINE_{wall.upper()}" if z > OUT_H_BS else f"IN_WALL_{wall.upper()}"



    if plane in ("left_wall", "right_wall") and z is not None:
        z_out = side_out_height(x)
        side = "LEFT" if plane == "left_wall" else "RIGHT"
        if z > z_out + NO_DECISION_CM:
            return f"OUT_LINE_{side}"
        if z < z_out - NO_DECISION_CM:
            return f"IN_WALL_{side}"
        return f"NO_DECISION_{side}"

    # If you later add back-wall homography and plane=="back_wall":
    # if plane == "back_wall" and z is not None:
    #     if z > OUT_H_BS + NO_DECISION_CM: return "OUT_LINE_BACK"
    #     if z < OUT_H_BS - NO_DECISION_CM: return "IN_WALL_BACK"
    #     return "NO_DECISION_BACK"

    # Fallback: old method if plane is still generic "wall"
    if plane == "wall" and x is not None and y is not None:
        wall = infer_wall(x, y)
        if z is None:
            return f"IN_WALL_{wall.upper()}"

        if wall == "front":
            return "OUT_LINE_FRONT" if z > OUT_H_F else "IN_WALL_FRONT"

        if wall in ("back", "left", "right"):
            return f"OUT_LINE_{wall.upper()}" if z > OUT_H_BS else f"IN_WALL_{wall.upper()}"

    return "UNKNOWN"



# ===== Main labeling pass =====

rows = []
with open(INP, newline="") as f:
    rd = csv.DictReader(f)
    fieldnames = rd.fieldnames or []
    for r in rd:
        call = call_event(r)
        # save guessed wall too
        x = _parse_float(r, "x_cm", default=float("nan"))
        y = _parse_float(r, "y_cm", default=float("nan"))
        wg = infer_wall(x, y) if (not math.isnan(x) and not math.isnan(y)) else ""
        r2 = {k: r.get(k) for k in fieldnames}
        r2["call"] = call
        r2["wall_guess"] = wg
        rows.append(r2)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames + ["call", "wall_guess"])
    w.writeheader()
    w.writerows(rows)

lists = {}
for r in rows:
    lists.setdefault(r["call"], []).append(float(r["t_sec"]))

for name, ts in lists.items():
    path = OUT_DIR / f"{name}.times.txt"
    with open(path, "w") as f:
        for t in sorted(ts):
            f.write(f"{t:.3f}\n")

print(f"[wrote] {OUT}  |  rows={len(rows)}  |  classes={len(lists)}")
for k, v in sorted(lists.items(), key=lambda kv: -len(kv[1]))[:8]:
    print(f"  {k:18s} -> {len(v)}")
print(f"[lists] {OUT_DIR}/*.times.txt for clip export")
