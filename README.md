# SquashVAR

**A single-camera, geometry-aware assistant referee for squash.**
Detect and track the ball in broadcast-style video, map it onto the four court surfaces, find wall and floor impacts, and turn them into **IN / OUT / NO_DECISION** line calls.

> **Status: research prototype.** This code accompanies a manuscript that is not yet published. It is not a certified officiating system, and the in/out calls have only been checked qualitatively (by inspecting overlay videos), not against a labelled set of real referee decisions.

## How it works

```
video ──► ball detector ──► tracker ──► plane assignment ──► court coordinates ──► impact detection ──► line call
         (YOLOv8)          (BoT-SORT)   front wall / floor     (homographies,       (speed, acceleration,   IN / OUT /
                                        / left / right wall     centimetres)         direction change)       NO_DECISION
```

1. **Detect** the ball with a fine-tuned YOLOv8 model.
2. **Track** detections over time into ball tracks.
3. **Assign** each detection to one of four court planes (front wall, floor, left wall, right wall). The plane is chosen as the one whose homography gives the lowest reprojection error.
4. **Map** image points to real court coordinates (cm) using one homography per plane, calibrated from four court points.
5. **Detect impacts** deterministically from changes in speed, acceleration and direction near a surface.
6. **Call** the impact against the court lines. If it lands within ±2 cm of a line, the system returns `NO_DECISION` instead of guessing.

Court geometry used: singles court 9750 × 6400 mm, tin 480 mm, front-wall out line 4570 mm, back-wall line 2130 mm.

## Repository layout

```
SquashVAR/
├── src/
│   ├── detect/      YOLOv8 inference
│   ├── track/       tracking, ballistic Kalman filter, plane probing
│   ├── geom/        homographies and pixel → court coordinates   (coords.py)
│   ├── fusion/      audio-onset helper
│   ├── eval/        evaluation summaries
│   ├── train/       detector training
│   └── utils/       visualisation helpers
├── tools/           calibration helpers, impact detection (nano_impacts.py),
│                    in/out labelling (label_inout.py), scoring, overlay renderers
├── configs/         per-plane court calibration files (front wall, floor, left, right)
├── figs/            figures used in the write-up
├── docker/          Dockerfile (CUDA 12.1, PyTorch 2.2)
├── notebooks/       exploration notebooks
└── requirements.txt pinned dependencies
```

## Setup

```bash
pip install -r requirements.txt     # pins ultralytics 8.2.103, numpy 1.26.4, opencv, scipy, ...
```

or build the container:

```bash
docker build -f docker/Dockerfile -t squashvar .
```

## Using the tools

Every script has its own options; run it with `--help` to see them.

```bash
# find impacts from tracked, court-mapped ball positions
python tools/nano_impacts.py --world_csv world.csv --out_csv impacts.csv

# render planes, ball and impacts onto the video
python tools/overlay_planes_and_impacts.py --video match.mp4 \
    --world_csv world.csv --impacts_csv impacts.csv --out overlay.mp4
```

Court calibration lives in `configs/` (one YAML per plane) and can be created or adjusted with the `tools/make_calib*.py` and `tools/nudge_calib*.py` helpers.

## What is not included

* the match video used for evaluation,
* the fine-tuned ball-detector weights (only the stock Ultralytics `yolov8*.pt` files are in the repository),
* the hand-labelled impact annotations.

## Limitations

* One fixed broadcast-style camera view per court calibration.
* The detector and impact thresholds were developed on a single match.
* A fast, motion-blurred ball is sometimes missed, and glass reflections can produce false detections.
* No labelled in/out dataset exists yet, so line-call accuracy has not been measured.

## License

MIT. See [LICENSE](LICENSE).

Author: Mahmoud Yassin
