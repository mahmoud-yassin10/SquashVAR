#!/usr/bin/env python3
import argparse, cv2, numpy as np, os, sys

def load_frame(path, t_sec=0.0):
    if os.path.splitext(path)[1].lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]:
        img = cv2.imread(path); 
        if img is None: raise SystemExit(f"could not read image: {path}")
        return img
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise SystemExit(f"could not open video: {path}")
    if t_sec>0: cap.set(cv2.CAP_PROP_POS_MSEC, t_sec*1000.0)
    ok, frame = cap.read(); cap.release()
    if not ok: raise SystemExit("failed to grab frame")
    return frame

def parse_points(s):
    pts=[]
    for p in s.strip().split():
        x,y = p.split(","); pts.append((int(float(x)), int(float(y))))
    return np.array(pts, dtype=np.int32)

def main(a):
    frame = load_frame(a.source, a.t)
    h,w = frame.shape[:2]
    mask = np.full((h,w), 255, np.uint8)  # 255 = keep; 0 = ignore
    if a.rect:
        x1,y1 = [int(v) for v in a.rect[0].split(",")]
        x2,y2 = [int(v) for v in a.rect[1].split(",")]
        x1,x2 = sorted([x1,x2]); y1,y2 = sorted([y1,y2])
        mask[y1:y2, x1:x2] = 0
    elif a.poly:
        poly = parse_points(a.poly)
        cv2.fillPoly(mask, [poly], 0)
    else:
        # default: top-right 28% width x 20% height as a guess for scoreboard
        x1 = int(0.72*w); y1 = 0; x2 = w; y2 = int(0.20*h)
        mask[y1:y2, x1:x2] = 0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cv2.imwrite(a.out, mask)
    print(f"wrote ROI mask -> {a.out}")
    # small preview image
    overlay = frame.copy()
    overlay[mask==0] = (0,0,255)
    alpha=0.35
    preview = cv2.addWeighted(frame, 1.0, overlay, alpha, 0)
    prev_path = os.path.splitext(a.out)[0] + ".preview.jpg"
    cv2.imwrite(prev_path, preview)
    print(f"preview -> {prev_path}")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="video or image to size mask")
    p.add_argument("--t", type=float, default=0.0, help="timestamp (s) if source is video")
    p.add_argument("--rect", nargs=2, help='rectangle as "x1,y1 x2,y2" to IGNORE')
    p.add_argument("--poly", help='polygon "x1,y1 x2,y2 x3,y3 ..." to IGNORE')
    p.add_argument("--out", required=True, help="output mask .png; 255=keep, 0=ignore")
    main(p.parse_args())
