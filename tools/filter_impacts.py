#!/usr/bin/env python3
# filters audio-aligned impacts for basic quality: plane present, in-bounds, conf range, speed range
import argparse, csv, math, pathlib

def flt(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def in_bounds(plane, x, y):
    if math.isnan(x) or math.isnan(y):
        return False
    if plane == "wall":
        return 0 <= x <= 640 and 0 <= y <= 457
    if plane == "floor":
        return 0 <= x <= 640 and 0 <= y <= 975
    return False

def main(a):
    inp = pathlib.Path(a.impacts_csv)
    out = pathlib.Path(a.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    kept = []
    with inp.open(newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            plane = (r.get("plane") or "").strip()
            conf  = flt(r.get("conf", ""))
            spd   = flt(r.get("speed_cm_s", ""))
            xcm   = flt(r.get("x_cm", ""))
            ycm   = flt(r.get("y_cm", ""))

            keep = True
            # must have a known plane
            keep &= plane in ("wall", "floor")
            # confidence gate
            keep &= (not math.isnan(conf)) and (conf >= a.conf_min)
            # speed gates (allow NaN only if you set --allow_nan_speed)
            if not a.allow_nan_speed:
                keep &= (not math.isnan(spd))
            if not math.isnan(spd):
                keep &= (spd >= a.speed_min_cm_s) and (spd <= a.speed_max_cm_s)
            # geometry in-bounds
            keep &= in_bounds(plane, xcm, ycm)

            if keep:
                kept.append(r)

    if kept:
        with out.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(kept[0].keys()))
            wr.writeheader()
            wr.writerows(kept)

    # summary
    w = sum(1 for r in kept if r.get("plane") == "wall")
    fl = sum(1 for r in kept if r.get("plane") == "floor")
    print(f"read: {inp}  -> kept: {len(kept)} rows")
    print(f"kept by plane: wall={w}  floor={fl}")
    print(f"wrote: {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--impacts_csv", required=True, help="output from postprocess_impacts.py")
    p.add_argument("--out_csv", required=True, help="filtered impacts destination")
    p.add_argument("--conf_min", type=float, default=0.35)
    p.add_argument("--speed_min_cm_s", type=float, default=900.0)
    p.add_argument("--speed_max_cm_s", type=float, default=12000.0, help="cap ~120 m/s")
    p.add_argument("--allow_nan_speed", action="store_true", help="keep impacts even if speed is NaN")
    a = p.parse_args()
    main(a)
