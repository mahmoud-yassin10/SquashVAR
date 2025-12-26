#!/usr/bin/env python3
import argparse, csv, pathlib, subprocess

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print("ffmpeg:", p.stdout)

def main(a):
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(a.impacts_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            # if --call filter is set, skip rows that don't match
            if a.call is not None:
                if (r.get("call") or "").strip() != a.call:
                    continue
            rows.append(r)

    if a.limit and a.limit > 0:
        import random
        random.seed(a.seed)
        rows = random.sample(rows, min(a.limit, len(rows)))

    for i, r in enumerate(rows, 1):
        t = float(r["t_sec"])
        ss = max(0.0, t - a.pre_s)
        dur = a.pre_s + a.post_s
        of = out / f"impact_{i:04d}_t{t:.2f}s_{r['plane']}_c{float(r['conf']):.2f}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", f"{ss:.3f}",
            "-i", a.video,
            "-t", f"{dur:.3f}",
            "-an",
            "-vf", "fps=25",
            str(of),
        ]
        run(cmd)

    print(f"wrote {len(rows)} clips -> {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Input video (e.g. overlay.mp4)")  # use your overlay.mp4
    p.add_argument("--impacts_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--pre_s", type=float, default=0.15)
    p.add_argument("--post_s", type=float, default=0.35)
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--call",
        default=None,
        help="If set, only export rows with this call label (e.g. OUT_TIN)",
    )
    a = p.parse_args()
    main(a)
