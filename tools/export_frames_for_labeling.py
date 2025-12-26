#!/usr/bin/env python3
import argparse, csv, cv2, os

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def grab_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None

def main(a):
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise SystemExit(f"could not open video: {a.video}")
    ensure_dir(os.path.join(a.out_dir, "images"))
    manifest = open(os.path.join(a.out_dir, "manifest.csv"), "w", newline="")
    wr = csv.writer(manifest); wr.writerow(["image_path","frame"])

    frames = []
    with open(a.impacts_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            for k in range(-a.window, a.window+1):
                frames.append(max(0, f+k))
    # dedup and limit
    frames = sorted(list(dict.fromkeys(frames)))[:a.limit] if a.limit>0 else sorted(list(dict.fromkeys(frames)))

    saved=0
    for f in frames:
        img = grab_frame(cap, f)
        if img is None: continue
        out_path = os.path.join(a.out_dir, "images", f"frame_{f:06d}.jpg")
        cv2.imwrite(out_path, img); wr.writerow([out_path, f]); saved+=1
    cap.release(); manifest.close()
    print(f"exported {saved} frames -> {os.path.join(a.out_dir, 'images')}")
    print(f"manifest -> {os.path.join(a.out_dir,'manifest.csv')}")
    print("Labeling tip: draw a single class 'ball'. Leave empty if no ball is visible.")

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--impacts_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--window", type=int, default=3, help="+/- frames around each impact")
    p.add_argument("--limit", type=int, default=0, help="0=no limit")
    main(p.parse_args())
