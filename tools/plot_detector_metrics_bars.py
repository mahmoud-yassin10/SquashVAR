# tools/plot_detector_metrics_bars.py
# Regenerates: figs/detector_metrics_bars.png
# Changes: tight crop, 12pt fonts, NO value numbers on bars, legend moved outside (no overlap)

import numpy as np
import matplotlib.pyplot as plt

def main():
    # ---- Data (from your table) ----
    labels = ["v8-S\n1920", "v8-S\n1280", "v8-n\n1280"]
    P     = [0.937, 0.862, 0.835]
    R     = [0.816, 0.890, 0.622]
    m50   = [0.906, 0.911, 0.711]
    m5095 = [0.487, 0.461, 0.544]

    # ---- Style: figure hygiene ----
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    x = np.arange(len(labels))
    w = 0.18

    # Size tuned for half-width of a figure* row in CVPR
    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=300)

    ax.bar(x - 1.5*w, P,     w, label="P")
    ax.bar(x - 0.5*w, R,     w, label="R")
    ax.bar(x + 0.5*w, m50,   w, label="mAP50")
    ax.bar(x + 1.5*w, m5095, w, label="mAP50-95")

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Legend outside to avoid covering bars
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=True,
        borderpad=0.3,
        handlelength=1.4,
        columnspacing=1.0
    )

    fig.tight_layout(pad=0.2)
    out = "figs/detector_metrics_bars.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print("Wrote:", out)

if __name__ == "__main__":
    main()
