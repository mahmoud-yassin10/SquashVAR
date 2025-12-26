#!/usr/bin/env python3
import argparse
import pandas as pd

def norm_surface(s: str):
    if s is None: return None
    s = str(s).strip().lower().replace(" ", "").replace("-", "_")
    # common variants
    if s in ("front", "frontwall", "front_wall"): return "front_wall"
    if s in ("left", "leftwall", "left_wall"):   return "left_wall"
    if s in ("right", "rightwall", "right_wall"):return "right_wall"
    if s in ("floor",):                          return "floor"
    if s in ("back", "backwall", "back_wall"):   return "back_wall"
    return s

def load_any_surface(df):
    for c in ["surface_pred","surface"]:
        if c in df.columns: return c
    raise SystemExit("No surface column found in pred csv (expected 'surface' or 'surface_pred').")

def load_any_frame(df):
    for c in ["frame_idx","frame","impact_frame","fi"]:
        if c in df.columns: return c
    raise SystemExit("No frame column found in csv.")

def match_events(gt, pred, tol, require_surface=True):
    used = set()
    TP = 0
    for _, g in gt.iterrows():
        gfr = int(g["frame"])
        gsf = g["surface"]
        cand = pred[(pred["frame"] >= gfr - tol) & (pred["frame"] <= gfr + tol)]
        if require_surface:
            cand = cand[cand["surface"] == gsf]
        best = None
        best_dist = 10**9
        for idx, p in cand.iterrows():
            if idx in used: 
                continue
            d = abs(int(p["frame"]) - gfr)
            if d < best_dist:
                best_dist = d
                best = idx
        if best is not None:
            used.add(best)
            TP += 1
    FP = len(pred) - len(used)
    FN = len(gt) - TP
    P = TP / (TP + FP) if (TP + FP) else 0.0
    R = TP / (TP + FN) if (TP + FN) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return TP, FP, FN, P, R, F1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--tol", type=int, default=2)
    ap.add_argument("--no_surface", action="store_true")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt_csv)
    pred = pd.read_csv(args.pred_csv)

    gt_frame = load_any_frame(gt)
    gt_surface = "surface_gt" if "surface_gt" in gt.columns else ("surface" if "surface" in gt.columns else None)
    if gt_surface is None:
        raise SystemExit("GT csv missing surface column (expected 'surface_gt' or 'surface').")

    pred_frame = load_any_frame(pred)
    pred_surface = load_any_surface(pred)

    gt2 = pd.DataFrame({
        "frame": gt[gt_frame].astype(int),
        "surface": gt[gt_surface].map(norm_surface),
    })
    pred2 = pd.DataFrame({
        "frame": pred[pred_frame].astype(int),
        "surface": pred[pred_surface].map(norm_surface),
    })

    req = not args.no_surface
    TP, FP, FN, P, R, F1 = match_events(gt2, pred2, args.tol, require_surface=req)
    print("OVERALL")
    print(f" tol=±{args.tol} frames, require_surface={req}")
    print(f" TP={TP} FP={FP} FN={FN}")
    print(f" P={P:.3f} R={R:.3f} F1={F1:.3f}")

    print("\nPER SURFACE")
    for s in ["floor","front_wall","left_wall","right_wall","back_wall"]:
        gts = gt2[gt2["surface"] == s]
        if len(gts) == 0:
            continue
        preds = pred2[pred2["surface"] == s]
        TP, FP, FN, P, R, F1 = match_events(gts, preds, args.tol, require_surface=False)
        print(f"{s:10} TP={TP:3d} FP={FP:3d} FN={FN:3d}  P={P:.3f} R={R:.3f} F1={F1:.3f}")

if __name__ == "__main__":
    main()
