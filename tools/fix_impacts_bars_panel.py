# tools/plot_impacts_bars_fixed.py
import matplotlib.pyplot as plt
import numpy as np

def main():
    # --- Panel A: overall (±25 frames) ---
    modes = ["surface-aware", "surface-agnostic"]
    P_mode  = [0.804, 0.922]
    R_mode  = [0.526, 0.603]
    F1_mode = [0.636, 0.729]

    # --- Panel B: per-surface (±25 frames), surface-aware ---
    surfaces = ["front", "floor", "left", "right"]
    P_s  = [1.000, 0.667, 0.500, 0.000]
    R_s  = [0.735, 0.500, 0.182, 0.000]
    F1_s = [0.847, 0.571, 0.267, 0.000]

    fig = plt.figure(figsize=(12.5, 4.2), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)

    # ---------- (A) ----------
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(modes))
    w = 0.22
    ax1.bar(x - w, P_mode,  width=w, label="P")
    ax1.bar(x,     R_mode,  width=w, label="R")
    ax1.bar(x + w, F1_mode, width=w, label="F1")
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Score")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, rotation=0)
    ax1.set_title("Impact detection (±25 frames)", pad=6)
    ax1.text(-0.12, 1.02, "(A)", transform=ax1.transAxes, fontsize=13, fontweight="bold")

    # ---------- (B) ----------
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    x2 = np.arange(len(surfaces))
    ax2.bar(x2 - w, P_s,  width=w, label="P")
    ax2.bar(x2,     R_s,  width=w, label="R")
    ax2.bar(x2 + w, F1_s, width=w, label="F1")
    ax2.set_ylim(0, 1.0)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(surfaces, rotation=0)
    ax2.set_title("Per-surface (surface-aware)", pad=6)
    ax2.text(-0.12, 1.02, "(B)", transform=ax2.transAxes, fontsize=13, fontweight="bold")

    # Make zero-bars explicit without adding numeric labels
    # (draw a thin baseline marker at 0 for the 'right' group)
    right_idx = surfaces.index("right")
    ax2.plot([right_idx - 0.35, right_idx + 0.35], [0, 0], linewidth=2)

    # Single legend for the whole figure (prevents duplicated whitespace)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(pad=0.4)
    out = "figs/impacts_bars_fixed.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
