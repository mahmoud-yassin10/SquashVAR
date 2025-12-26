#!/usr/bin/env python3
import csv, argparse, math, os
from collections import defaultdict

def flt(x):
    try: return float(x)
    except: return math.nan

def in_bounds(pl, x, y):
    if math.isnan(x) or math.isnan(y): return False
    if pl == "wall":  return 0 <= x <= 640 and 0 <= y <= 457
    if pl == "floor": return 0 <= x <= 640 and 0 <= y <= 975
    return False

def load_tracks(path):
    rows = []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            # normalize numeric fields
            r["_frame"] = int(r["frame"])
            r["_t"] = flt(r["t_sec"])
            r["_xcm"] = flt(r["x_cm"]); r["_ycm"] = flt(r["y_cm"])
            r["_conf"] = flt(r["conf"]); r["_v"] = flt(r["speed_cm_s"])
            r["_on"] = 1 if r.get("onset_window","0") in ("1", "True", "true") else 0
            rows.append(r)
    return rows

def segment_onset_windows(rows, gap_frames=1):
    """
    Group consecutive frames where onset_window==1 into segments,
    split when a gap > gap_frames occurs.
    Returns list of lists of indices into rows.
    """
    segs = []
    cur = []
    last_f = None
    for i, r in enumerate(rows):
        if r["_on"] != 1:
            if cur:
                segs.append(cur); cur=[]
            last_f=None
            continue
        f = r["_frame"]
        if not cur:
            cur=[i]; last_f=f
        else:
            if f - last_f <= gap_frames:
                cur.append(i); last_f=f
            else:
                segs.append(cur); cur=[i]; last_f=f
    if cur: segs.append(cur)
    return segs

def choose_one(rows, idxs, speed_min, conf_min):
    """
    Pick one index from a segment:
    1) in-bounds & conf>=conf_min & speed>=speed_min, choose max speed
    2) else in-bounds & conf>=conf_min, choose max conf
    3) else in-bounds, choose max conf
    4) else any (fallback) choose max conf
    """
    cands = []
    for i in idxs:
        r = rows[i]
        pl = r.get("plane","")
        ok_bounds = in_bounds(pl, r["_xcm"], r["_ycm"])
        cands.append((i, ok_bounds, r["_conf"], r["_v"]))
    # filters in order
    fltrs = [
        lambda t: t[1] and (not math.isnan(t[3]) and t[3] >= speed_min) and (not math.isnan(t[2]) and t[2] >= conf_min),
        lambda t: t[1] and (not math.isnan(t[2]) and t[2] >= conf_min),
        lambda t: t[1],
        lambda t: True
    ]
    # selectors
    def argmax_speed(ts): return max(ts, key=lambda t: (t[3] if not math.isnan(t[3]) else -1e9, t[2] if not math.isnan(t[2]) else -1e9))
    def argmax_conf(ts):  return max(ts, key=lambda t: (t[2] if not math.isnan(t[2]) else -1e9, t[3] if not math.isnan(t[3]) else -1e9))
    selectors = [argmax_speed, argmax_conf, argmax_conf, argmax_conf]

    for fltfun, sel in zip(fltrs, selectors):
        pool = [t for t in cands if fltfun(t)]
        if pool:
            return sel(pool)[0]
    return idxs[0]  # should not happen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", required=True, help="input tracks.csv from pipeline")
    ap.add_argument("--out_csv", required=True, help="cleaned tracks CSV (events deduped)")
    ap.add_argument("--impacts_csv", required=True, help="one-line-per-impact CSV")
    ap.add_argument("--gap_frames", type=int, default=1, help="max gap to still belong to same onset segment (default 1)")
    ap.add_argument("--speed_min_cm_s", type=float, default=800.0, help="min speed (cm/s) preferred for impact choice")
    ap.add_argument("--conf_min", type=float, default=0.25, help="min confidence preferred")
    args = ap.parse_args()

    rows = load_tracks(args.tracks)
    segs = segment_onset_windows(rows, gap_frames=args.gap_frames)

    chosen_idx = []
    for seg in segs:
        i = choose_one(rows, seg, speed_min=args.speed_min_cm_s, conf_min=args.conf_min)
        chosen_idx.append(i)

    # Build impacts table
    impacts = []
    for i in chosen_idx:
        r = rows[i]
        plane = r.get("plane","")
        label = "impact_wall" if plane=="wall" else ("impact_floor" if plane=="floor" else "impact")
        impacts.append(dict(
            frame=r["_frame"], t_sec=r["_t"], plane=plane,
            x_cm=r["_xcm"], y_cm=r["_ycm"],
            speed_cm_s=r["_v"], conf=r["_conf"], event=label
        ))

    # Rewrite events: only mark chosen frames
    fields = list(rows[0].keys())
    # ensure 'event' exists
    if "event" not in fields: fields.append("event")
    # reset all
    for r in rows: r["event"] = ""
    # set chosen
    for i, imp in zip(chosen_idx, impacts):
        rows[i]["event"] = imp["event"]

    # write out_csv
    with open(args.out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    # write impacts_csv
    imp_fields = ["frame","t_sec","plane","x_cm","y_cm","speed_cm_s","conf","event"]
    with open(args.impacts_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=imp_fields)
        wr.writeheader()
        wr.writerows(impacts)

    # Print quick summary
    n_rows = len(rows)
    n_imp = len(impacts)
    print(f"rows: {n_rows}")
    print(f"onset segments: {len(segs)}")
    print(f"impacts after dedup: {n_imp}")
    if n_imp>1:
        # rough spacing in frames
        frs = sorted([imp["frame"] for imp in impacts])
        gaps = [b-a for a,b in zip(frs[:-1], frs[1:])]
        print(f"median gap (frames): {sorted(gaps)[len(gaps)//2]}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.impacts_csv}")

if __name__ == "__main__":
    main()