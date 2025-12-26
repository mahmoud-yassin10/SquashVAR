import argparse, pandas as pd, numpy as np
from collections import defaultdict

# ---------- normalization ----------
SURF_MAP = {
  "front":"FRONT","front_wall":"FRONT","fw":"FRONT",
  "back":"BACK","back_wall":"BACK","bw":"BACK",
  "left":"LEFT","left_wall":"LEFT","lw":"LEFT",
  "right":"RIGHT","right_wall":"RIGHT","rw":"RIGHT",
  "floor":"FLOOR","f":"FLOOR",
}
TYPE_MAP = {
  "wall_impact":"WALL_IMPACT",
  "wall_impact_corner":"WALL_IMPACT_CORNER",
  "floor_bounce":"FLOOR_BOUNCE",
  "floor_bounce_seg":"FLOOR_BOUNCE_SEG",
}

def norm_surface(x):
  if pd.isna(x): return ""
  s=str(x).strip().lower()
  return SURF_MAP.get(s, str(x).strip().upper())

def norm_type(x):
  if pd.isna(x): return ""
  s=str(x).strip().lower()
  return TYPE_MAP.get(s, str(x).strip().upper())

def load_pred(path):
  df=pd.read_csv(path)
  # expected nano headers: surface, impact_type, frame_idx
  if "frame_idx" not in df.columns and "frame" in df.columns:
    df["frame_idx"]=df["frame"]
  df["surface"]=df.get("surface","").apply(norm_surface)
  df["impact_type"]=df.get("impact_type","").apply(norm_type)
  return df[["frame_idx","surface","impact_type"]].copy()

def load_gt(path):
  df=pd.read_csv(path)
  if "frame_idx" not in df.columns:
    raise ValueError("GT must have frame_idx column")
  if "surface_gt" not in df.columns or "impact_type_gt" not in df.columns:
    raise ValueError("GT must have surface_gt and impact_type_gt columns")
  df["surface_gt"]=df["surface_gt"].apply(norm_surface)
  df["impact_type_gt"]=df["impact_type_gt"].apply(norm_type)
  return df[["frame_idx","surface_gt","impact_type_gt"]].copy()

# ---------- matching ----------
def match_events(gt, pr, tol_frames=2, require_type=True, require_surface=True):
  """
  Greedy 1-1 matching by smallest |frame diff| within tolerance,
  optionally requiring same surface/type.
  Returns: list of (gt_i, pr_j, dt), plus unmatched indexes.
  """
  gt = gt.reset_index(drop=True)
  pr = pr.reset_index(drop=True)

  used_pr=set()
  matches=[]

  # pre-index preds by frame window for speed
  pr_by_frame=defaultdict(list)
  for j,row in pr.iterrows():
    pr_by_frame[int(row.frame_idx)].append(j)

  for i,g in gt.iterrows():
    gi=int(g.frame_idx)
    cand=[]
    for f in range(gi - tol_frames, gi + tol_frames + 1):
      for j in pr_by_frame.get(f, []):
        if j in used_pr: 
          continue
        p = pr.loc[j]
        if require_surface and p.surface != g.surface_gt: 
          continue
        if require_type and p.impact_type != g.impact_type_gt:
          continue
        cand.append((abs(int(p.frame_idx)-gi), j))
    if not cand:
      continue
    cand.sort(key=lambda x: x[0])
    dt, j = cand[0]
    used_pr.add(j)
    matches.append((i, j, dt))

  unmatched_gt = [i for i in range(len(gt)) if i not in {m[0] for m in matches}]
  unmatched_pr = [j for j in range(len(pr)) if j not in used_pr]
  return matches, unmatched_gt, unmatched_pr

def prf1(tp, fp, fn):
  P = tp / (tp + fp + 1e-9)
  R = tp / (tp + fn + 1e-9)
  F1 = 2*P*R/(P+R+1e-9)
  return P,R,F1

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--gt", required=True)
  ap.add_argument("--pred", required=True)
  ap.add_argument("--tol", type=int, default=2, help="tolerance in frames (±tol)")
  ap.add_argument("--ignore_type", action="store_true")
  ap.add_argument("--ignore_surface", action="store_true")
  args=ap.parse_args()

  gt=load_gt(args.gt)
  pr=load_pred(args.pred)

  matches, ug, up = match_events(
    gt, pr, tol_frames=args.tol,
    require_type=not args.ignore_type,
    require_surface=not args.ignore_surface
  )

  tp=len(matches); fn=len(ug); fp=len(up)
  P,R,F1=prf1(tp,fp,fn)

  print("=== Overall ===")
  print(f"GT={len(gt)}  Pred={len(pr)}  TP={tp}  FP={fp}  FN={fn}")
  print(f"Precision={P:.4f}  Recall={R:.4f}  F1={F1:.4f}  tol=±{args.tol} frames")
  print()

  # per-surface
  print("=== Per surface (same matching rules) ===")
  for surf in sorted(set(gt.surface_gt.unique()) | set(pr.surface.unique())):
    gts = gt[gt.surface_gt==surf].reset_index(drop=True)
    prs = pr[pr.surface==surf].reset_index(drop=True)
    m, ug2, up2 = match_events(
      gts, prs, tol_frames=args.tol,
      require_type=not args.ignore_type,
      require_surface=False  # already filtered
    )
    tp2=len(m); fn2=len(ug2); fp2=len(up2)
    P2,R2,F12=prf1(tp2,fp2,fn2)
    print(f"{surf:6}  GT={len(gts):4} Pred={len(prs):4} TP={tp2:4} FP={fp2:4} FN={fn2:4}  P={P2:.3f} R={R2:.3f} F1={F12:.3f}")
  print()

  # per-type
  print("=== Per impact_type (same matching rules) ===")
  for typ in sorted(set(gt.impact_type_gt.unique()) | set(pr.impact_type.unique())):
    gtt = gt[gt.impact_type_gt==typ].reset_index(drop=True)
    prt = pr[pr.impact_type==typ].reset_index(drop=True)
    m, ug2, up2 = match_events(
      gtt, prt, tol_frames=args.tol,
      require_type=False,  # already filtered
      require_surface=not args.ignore_surface
    )
    tp2=len(m); fn2=len(ug2); fp2=len(up2)
    P2,R2,F12=prf1(tp2,fp2,fn2)
    print(f"{typ:16} GT={len(gtt):4} Pred={len(prt):4} TP={tp2:4} FP={fp2:4} FN={fn2:4}  P={P2:.3f} R={R2:.3f} F1={F12:.3f}")

if __name__=="__main__":
  main()
