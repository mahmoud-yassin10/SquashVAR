#!/usr/bin/env python3
import argparse, csv, cv2, numpy as np, math

def fnum(s):
    try: return float(s)
    except: return float("nan")

def main(a):
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise SystemExit(f"could not open video: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    fgbg = cv2.createBackgroundSubtractorMOG2(history=a.history, varThreshold=a.var, detectShadows=False)

    # pre-read requested frames (ordered)
    with open(a.tracks, newline="") as fh:
        frames = [int(r["frame"]) for r in csv.DictReader(fh)]
    minf, maxf = (min(frames), max(frames)) if frames else (0,0)

    # build a map frame_idx -> list of rows
    fh = open(a.tracks, newline=""); rd = csv.DictReader(fh)
    rows_by_f = {}
    for r in rd:
        fi = int(r["frame"])
        rows_by_f.setdefault(fi, []).append(r)
    fh.close()

    out = open(a.out_csv, "w", newline="")
    fieldnames = rd.fieldnames + (["motion_ok"] if "motion_ok" not in rd.fieldnames else [])
    wr = csv.DictWriter(out, fieldnames=fieldnames); wr.writeheader()

    kept = dropped = total=0
    cur_f = -1
    while True:
        cur_f += 1
        ok, frame = cap.read()
        if not ok: break
        if cur_f < minf: 
            fgbg.apply(frame); 
            continue
        if cur_f > maxf: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = fgbg.apply(gray)

        if cur_f in rows_by_f:
            for r in rows_by_f[cur_f]:
                total += 1
                x = fnum(r.get("x_px","")); y = fnum(r.get("y_px",""))
                motion_ok = ""
                if x==x and y==y:  # not NaN
                    xi, yi = int(round(x)), int(round(y))
                    x1 = max(0, xi - a.patch//2); x2 = min(fg.shape[1], xi + a.patch//2 + 1)
                    y1 = max(0, yi - a.patch//2); y2 = min(fg.shape[0], yi + a.patch//2 + 1)
                    patch = fg[y1:y2, x1:x2]
                    score = float(np.mean(patch))  # 0..255
                    motion_ok = 1 if score >= a.thresh else 0
                r["motion_ok"] = motion_ok
                if a.drop_fail and motion_ok==0:
                    dropped += 1
                else:
                    wr.writerow(r); kept += 1

    cap.release(); out.close()
    print(f"rows: {total}  kept: {kept}  dropped_by_motion: {dropped}")
    print(f"wrote: {a.out_csv}")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--tracks", required=True, help="CSV aligned to this video")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--patch", type=int, default=21, help="square patch size around (x,y)")
    p.add_argument("--thresh", type=float, default=8.0, help="mean FG value threshold (0..255)")
    p.add_argument("--history", type=int, default=500)
    p.add_argument("--var", type=float, default=16.0, help="MOG2 varThreshold")
    p.add_argument("--drop_fail", action="store_true")
    main(p.parse_args())
