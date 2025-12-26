# src/eval/summary.py
import argparse, csv, sys
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="tracks.csv from pipeline")
    ap.add_argument("--hist", action="store_true",
                    help="save speed histogram as runs/summary_speed_hist.png (requires matplotlib)")
    return ap.parse_args()

def main(csv_path, save_hist=False):
    frames = 0
    detected = 0
    impacts = 0
    plane_counts = {"wall": 0, "floor": 0, "air": 0}
    speeds = []

    with open(csv_path, newline="") as f:
        rd = csv.DictReader(f)
        required = {"frame","x_px","y_px","plane","speed_cm_s","event"}
        missing = required - set([c.strip() for c in rd.fieldnames or []])
        if missing:
            print(f"[warn] CSV missing columns: {sorted(missing)} — continuing with what’s available.")

        for row in rd:
            frames += 1
            plane = (row.get("plane") or "air").strip()
            if plane not in plane_counts: plane_counts[plane] = 0
            plane_counts[plane] += 1

            if (row.get("x_px") or "").strip():
                detected += 1

            ev = (row.get("event") or "")
            if ev.startswith("impact_"):
                impacts += 1

            s = (row.get("speed_cm_s") or "").strip()
            if s:
                try:
                    speeds.append(float(s))
                except ValueError:
                    pass  # ignore bad values

    print(f"frames: {frames}")
    if frames:
        print(f"detected frames: {detected} ({detected/frames*100:.1f}%)")
    else:
        print("detected frames: 0 (0.0%)")
    print(f"impacts flagged: {impacts}")
    print("plane counts:", {k:int(v) for k,v in plane_counts.items()})

    if speeds:
        s = np.array(speeds, dtype=float)
        print(f"speed cm/s: mean {s.mean():.1f}, p50 {np.percentile(s,50):.1f}, "
              f"p90 {np.percentile(s,90):.1f}, max {s.max():.1f}")
        if save_hist:
            try:
                import matplotlib.pyplot as plt, os
                os.makedirs("runs", exist_ok=True)
                plt.figure()
                plt.hist(s, bins=30)
                plt.xlabel("speed (cm/s)")
                plt.ylabel("count")
                plt.title("Speed histogram")
                outp = "runs/summary_speed_hist.png"
                plt.savefig(outp, bbox_inches="tight"); plt.close()
                print(f"saved histogram → {outp}")
            except Exception as e:
                print(f"[warn] could not save histogram: {e}")
    else:
        print("no speed values found")

if __name__ == "__main__":
    args = parse_args()
    main(args.csv_path, save_hist=args.hist)
