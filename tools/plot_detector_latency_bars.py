# tools/plot_detector_latency_bars.py
# Regenerates: figs/detector_latency_bars.png
# Fixes: 12pt fonts, short labels, tight crop, reduced right whitespace, readable bar-end values

import numpy as np
import matplotlib.pyplot as plt

def main():
    # ---- Data (from your table) ----
    labels = ["v8-S 1920", "v8-S 1280", "v8-n 1280"]
    ms_img = [22.4, 11.6, 8.0]

    # ---- Style: figure hygiene ----
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })

    # Sort so the slowest is on top (nice reading)
    order = np.argsort(ms_img)[::-1]
    labels = [labels[i] for i in order]
    ms_img = [ms_img[i] for i in order]

    # Half-width figure* subfigure size
    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=300)

    y = np.arange(len(labels))
    bars = ax.barh(y, ms_img)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # top = largest latency

    ax.set_xlabel("ms/image")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # Remove dead space on the right
    xmax = max(ms_img)
    ax.set_xlim(0, xmax * 1.10)

    # Value labels at end of bars (readable)
    for b, v in zip(bars, ms_img):
        ax.text(v + xmax * 0.02, b.get_y() + b.get_height()/2,
                f"{v:.1f}", va="center", ha="left", fontsize=12)

    fig.tight_layout(pad=0.2)
    out = "figs/detector_latency_bars.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print("Wrote:", out)

if __name__ == "__main__":
    main()
