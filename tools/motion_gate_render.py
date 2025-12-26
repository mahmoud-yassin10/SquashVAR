import os, cv2, math, argparse

def parse_label_line(line):
    p = line.strip().split()
    if len(p) < 6: return None
    cls = int(float(p[0])); cx, cy, w, h = map(float, p[1:5])
    conf = float(p[5]); tid = int(float(p[6])) if len(p) >= 7 else -1
    return cls, cx, cy, w, h, conf, tid

def load_frame_labels(labels_dir, t):
    f = os.path.join(labels_dir, f"{t:06d}.txt")
    if not os.path.isfile(f): return []
    out=[]; 
    with open(f, "r") as fh:
        for ln in fh:
            r = parse_label_line(ln)
            if r: out.append(r)
    return out

def denorm_to_xyxy(cx, cy, w, h, W, H):
    x1 = (cx - w/2) * W; y1 = (cy - h/2) * H
    x2 = (cx + w/2) * W; y2 = (cy + h/2) * H
    return int(x1), int(y1), int(x2), int(y2)

def build_tracks(labels_dir, total_frames, use_ids=True):
    tracks={}; fallback=0
    for t in range(total_frames):
        for (cls,cx,cy,w,h,conf,tid) in load_frame_labels(labels_dir, t):
            key = tid if (use_ids and tid>=0) else (t<<16)+fallback; fallback+=1
            tracks.setdefault(key, []).append((t,cx,cy,conf,(cx,cy,w,h)))
    return tracks

def gate_tracks(tracks, fps, W, H, min_len, vmin, vmax, ema, conf_th):
    kept={}
    for tid, tr in tracks.items():
        if len(tr) < min_len: continue
        speeds=[]
        for i in range(len(tr)-1):
            _, cx1, cy1, _, _ = tr[i]
            _, cx2, cy2, _, _ = tr[i+1]
            x1, y1 = cx1*W, cy1*H; x2, y2 = cx2*W, cy2*H
            speeds.append(((x2-x1)**2 + (y2-y1)**2) ** 0.5)
        if not speeds: continue
        if not (vmin <= max(speeds) <= vmax): continue
        cema=0.0; cmax=0.0
        for _,_,_,c,_ in tr:
            cema = ema*c + (1-ema)*cema
            if cema > cmax: cmax = cema
        if cmax < conf_th: continue
        kept[tid]=tr
    return kept

def render_overlay(video, labels_dir, out_path, conf_th, min_len, vmin, vmax, ema, use_ids):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));  H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT));  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    tracks = build_tracks(labels_dir, N, use_ids=use_ids)
    tracks = gate_tracks(tracks, fps, W, H, min_len, vmin, vmax, ema, conf_th)

    per_frame = {t: [] for t in range(N)}
    for tid, tr in tracks.items():
        for (t,cx,cy,conf,(ccx,ccy,w,h)) in tr:
            x1,y1,x2,y2 = denorm_to_xyxy(ccx,ccy,w,h,W,H)
            per_frame[t].append((x1,y1,x2,y2,conf))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (W,H))
    t=0
    GREEN=(0,255,0)
    while True:
        ok, frame = cap.read()
        if not ok: break
        for (x1,y1,x2,y2,conf) in per_frame.get(t, []):
            cv2.rectangle(frame, (x1,y1), (x2,y2), GREEN, 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, max(0,y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2, cv2.LINE_AA)
        out.write(frame); t+=1
    cap.release(); out.release()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.56)
    ap.add_argument("--minlen", type=int, default=4)
    ap.add_argument("--vmin", type=float, default=5)
    ap.add_argument("--vmax", type=float, default=220)
    ap.add_argument("--ema", type=float, default=0.4)
    ap.add_argument("--no_ids", action="store_true")
    args = ap.parse_args()
    render_overlay(args.video, args.labels, args.out,
                   conf_th=args.conf, min_len=args.minlen,
                   vmin=args.vmin, vmax=args.vmax, ema=args.ema,
                   use_ids=not args.no_ids)
