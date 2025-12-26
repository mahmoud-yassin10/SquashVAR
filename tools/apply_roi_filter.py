#!/usr/bin/env python3
import argparse, csv, cv2

def ufloat(s):
    try: return float(s)
    except: return float("nan")

def main(a):
    mask = cv2.imread(a.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None: raise SystemExit(f"could not read mask: {a.mask}")
    h,w = mask.shape[:2]

    kept=flagged=rows=0
    with open(a.tracks, newline="") as fh_in, open(a.out_csv, "w", newline="") as fh_out:
        rd = csv.DictReader(fh_in)
        fieldnames = rd.fieldnames + (["roi_ok"] if "roi_ok" not in rd.fieldnames else [])
        wr = csv.DictWriter(fh_out, fieldnames=fieldnames); wr.writeheader()
        for r in rd:
            rows += 1
            x = ufloat(r.get("x_px","")); y = ufloat(r.get("y_px",""))
            roi_ok = ""
            if x==x and y==y:  # not NaN
                xi, yi = int(round(x)), int(round(y))
                if 0<=xi<w and 0<=yi<h:
                    roi_ok = 1 if mask[yi,xi] > 0 else 0
                else:
                    roi_ok = 0
            r["roi_ok"] = roi_ok
            if a.drop_fail and roi_ok==0: 
                flagged += 1
                continue
            wr.writerow(r); kept += 1
    print(f"rows: {rows}  kept: {kept}  dropped_by_roi: {flagged}")
    print(f"wrote: {a.out_csv}")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--tracks", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--drop_fail", action="store_true", help="drop detections inside ignore region")
    main(p.parse_args())
