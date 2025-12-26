#!/usr/bin/env bash
set -euo pipefail

VID="${1:?Usage: $0 /path/to/video}"
MODEL="${2:?Usage: $0 /path/to/best.pt}"
OUTDIR="${3:-/workspace/runs/full_overlay}"

IMGZ="${IMGZ:-1920}"
CONF="${CONF:-0.45}"
IOU="${IOU:-0.50}"
HALF="${HALF:-True}"
FPS="${FPS:-60}"

mkdir -p "$OUTDIR"

# 1) Track (produces labels + tracks)
yolo track model="$MODEL" source="$VID" imgsz="$IMGZ" conf="$CONF" iou="$IOU" device=0 half="$HALF" \
  tracker=botsort.yaml max_det=5 classes=0 save=True save_txt=True save_conf=True \
  hide_labels=True hide_conf=False line_thickness=2 project="$OUTDIR" name=track

LABEL_DIR="$OUTDIR/track/labels"

# 2) Map to world (polygons + homographies) -> world_multi.csv
python /workspace/tools/world_from_labels_modes.py \
  --labels_dir "$LABEL_DIR" \
  --video "$VID" \
  --yaml_front /workspace/configs/calib.front.yaml \
  --yaml_floor /workspace/configs/calib.floor.yaml \
  --yaml_left  /workspace/configs/calib.left.yaml \
  --yaml_right /workspace/configs/calib.right.yaml \
  --mode multi \
  --margin_floor 28 \
  --margin_front 2 \
  --out_csv "$OUTDIR/world_multi.csv"

# 3) Impacts from kinematics -> impacts.csv
python /workspace/tools/nano_impacts.py \
  --world_csv "$OUTDIR/world_multi.csv" \
  --out_csv "$OUTDIR/impacts.csv" \
  --fps "$FPS"

# 4) In/Out classification (rule-based) -> impacts_labeled.csv
python - <<'PY'
import pandas as pd, math
imp = pd.read_csv("/workspace/runs/full_overlay/impacts.csv")
# expected columns include: frame_idx, plane, x_cm, y_cm, z_cm (adjust if your schema differs)
TIN_Z=48.0
FRONT_OUT_Z=457.0
SIDE_BACK_OUT_Z=213.0
BAND=2.0
X0,X1=0.0,975.0
Y0,Y1=0.0,640.0

def band_no_decision(val, a, b):
    return (abs(val-a) <= BAND) or (abs(val-b) <= BAND)

def classify(row):
    plane=str(row.get("plane","")).lower()
    x=row.get("x_cm", float("nan"))
    y=row.get("y_cm", float("nan"))
    z=row.get("z_cm", float("nan"))

    if plane in ("front_wall","front"):
        if abs(z-TIN_Z)<=BAND or abs(z-FRONT_OUT_Z)<=BAND: return "NO_DECISION"
        if z < TIN_Z: return "OUT_TIN"
        if z > FRONT_OUT_Z: return "OUT"
        return "IN"

    if plane in ("left_wall","right_wall","back_wall","left","right","back"):
        if abs(z-SIDE_BACK_OUT_Z)<=BAND: return "NO_DECISION"
        return "OUT" if z > SIDE_BACK_OUT_Z else "IN"

    if plane == "floor":
        # abstain near boundaries
        if band_no_decision(x, X0, X1) or band_no_decision(y, Y0, Y1): return "NO_DECISION"
        if (X0 <= x <= X1) and (Y0 <= y <= Y1): return "IN"
        return "OUT"

    return "NO_DECISION"

imp["inout"] = imp.apply(classify, axis=1)
imp.to_csv("/workspace/runs/full_overlay/impacts_labeled.csv", index=False)
print("Wrote /workspace/runs/full_overlay/impacts_labeled.csv")
PY

# 5) Render final overlay video: boxes + polygons + impacts + speed + in/out
python /workspace/tools/overlay_box_and_impacts.py \
  --video "$VID" \
  --labels_dir "$LABEL_DIR" \
  --world_csv "$OUTDIR/world_multi.csv" \
  --impacts_csv "$OUTDIR/impacts_labeled.csv" \
  --yaml_front /workspace/configs/calib.front.yaml \
  --yaml_floor /workspace/configs/calib.floor.yaml \
  --yaml_left  /workspace/configs/calib.left.yaml \
  --yaml_right /workspace/configs/calib.right.yaml \
  --out_video "$OUTDIR/overlay_full.mp4" \
  --persist 18 --radius 8 --thickness 2
echo "DONE -> $OUTDIR/overlay_full.mp4"
