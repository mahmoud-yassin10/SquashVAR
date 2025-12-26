#!/usr/bin/env python3
import argparse, csv, numpy as np, math
from src.fusion.audio_onset import detect_onsets

def main(a):
    # impacts → times (s)
    t_imp = []
    with open(a.impacts_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            try: t_imp.append(float(r["t_sec"]))
            except: pass
    t_imp = np.array(sorted(t_imp))
    if len(t_imp)==0:
        print("no impacts in csv"); return

    # audio onsets (seconds)
    on = detect_onsets(a.wav)
    on = np.array(on)

    win_s = (a.window_frames / a.fps)
    offs = np.arange(a.min_ms, a.max_ms + 1e-9, a.step_ms, dtype=float)
    best = None
    for off in offs:
        shifted = on + off/1000.0
        # for each onset, find distance to nearest impact
        idx = np.searchsorted(t_imp, shifted)
        left  = np.clip(idx-1, 0, len(t_imp)-1)
        right = np.clip(idx,   0, len(t_imp)-1)
        d = np.minimum(np.abs(shifted - t_imp[left]), np.abs(shifted - t_imp[right]))
        matches = np.sum(d <= win_s)
        if best is None or matches > best[1]:
            best = (off, matches)
    off, m = best
    print(f"best offset: {off:.0f} ms | matches={m}/{len(on)} within ±{a.window_frames} frames (~±{win_s:.3f}s)")
    # small table
    print("offset_ms\tmatches")
    for off in offs:
        shifted = on + off/1000.0
        idx = np.searchsorted(t_imp, shifted)
        left  = np.clip(idx-1, 0, len(t_imp)-1)
        right = np.clip(idx,   0, len(t_imp)-1)
        d = np.minimum(np.abs(shifted - t_imp[left]), np.abs(shifted - t_imp[right]))
        print(f"{int(off)}\t{int(np.sum(d <= win_s))}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--impacts_csv", required=True)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--window_frames", type=int, default=6)
    p.add_argument("--min_ms", type=int, default=-100)
    p.add_argument("--max_ms", type=int, default=100)
    p.add_argument("--step_ms", type=int, default=5)
    a = p.parse_args(); main(a)
