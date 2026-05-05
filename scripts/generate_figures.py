"""
Regenerate all charts used in the VideoEditPG report.
Output: report/assets/chart_*.png
Run: python scripts/generate_figures.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "assets")
os.makedirs(OUT_DIR, exist_ok=True)

# Claude design palette
PURPLE = "#7C3AED"
VIOLET = "#9B59B6"
BLUE   = "#3B82F6"
TEAL   = "#06B6D4"
GREEN  = "#10B981"
GRAY   = "#6B7280"
BG     = "white"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "font.family":      "sans-serif",
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
})

# Load real Newton results
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "test_results.json")
with open(RESULTS_PATH) as f:
    data = json.load(f)


# ── Figure 1: Generation Times ────────────────────────────────────────────────
def chart_gen_times():
    videos = data["identity_generation"]["videos"]
    subjects = [v["name"].replace("_", "\n") for v in videos]
    times    = [v["time_sec"] for v in videos]
    mean_t   = np.mean(times)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(subjects, times, color=PURPLE, width=0.55, zorder=3)
    ax.axhline(mean_t, color=TEAL, linewidth=1.5, linestyle="--", zorder=4,
               label=f"Mean: {mean_t:.1f}s")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("Identity Generation — Inference Time per Subject\n(Wan2.1-VACE-1.3B, V100-32GB)")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(times) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{t}s", ha="center", va="bottom", fontsize=9, color=PURPLE, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "chart_gen_times.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 2: LoRA Injection Overhead ─────────────────────────────────────────
def chart_lora_overhead():
    lora = data["lora_injection"]
    labels = ["Baseline\n(no LoRA)", "With LoRA\n(rank 4)"]
    times  = [lora["baseline_time"], lora["lora_time"]]
    colors = [BLUE, PURPLE]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, times, color=colors, width=0.45, zorder=3)
    ax.set_ylabel("Inference Time (s)")
    ax.set_title("LoRA Injection Overhead\n(Wan2.1-VACE-1.3B, V100-32GB)")
    ax.set_ylim(0, max(times) * 1.35)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    overhead = (times[1] - times[0]) / times[0] * 100
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{t}s", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.annotate(
        f"+{overhead:.1f}% overhead",
        xy=(1, times[1]), xycoords="data",
        xytext=(0.97, 0.30), textcoords="axes fraction",
        ha="right", fontsize=10, color=PURPLE,
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5),
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "chart_lora_overhead.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 3: LoRA Parameter Share ────────────────────────────────────────────
def chart_lora_params():
    lora = data["lora_injection"]
    lora_p  = lora["lora_params"]
    total_p = 1_300_000_000
    other_p = total_p - lora_p

    sizes  = [lora_p, other_p]
    colors = [PURPLE, "#E5E7EB"]
    labels = [f"LoRA\n({lora['lora_percent']}%)", "Frozen backbone"]
    explode = (0.06, 0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, explode=explode,
        autopct=lambda p: f"{p:.2f}%" if p > 1 else "",
        startangle=90, pctdistance=0.6,
        wedgeprops=dict(linewidth=1.5, edgecolor="white"),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("LoRA Parameter Share\n(614K of 1.3B total)", pad=14)
    legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_patches, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(top=0.82)
    ax.annotate(
        "0.04% of\n1.3B params",
        xy=(0.0, 0.0), xycoords="axes fraction",
        xytext=(0.55, 0.55), textcoords="axes fraction",
        ha="center", fontsize=9, color=PURPLE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2),
    )
    path = os.path.join(OUT_DIR, "chart_lora_params.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 4: Storage Comparison ──────────────────────────────────────────────
def chart_storage():
    methods = ["Full\nFine-tune", "HyperDreamBooth", "VideoEditPG\n(ours)"]
    sizes_gb = [2.6, 120e-6, 50e-6]
    colors   = [GRAY, BLUE, PURPLE]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(methods, sizes_gb, color=colors, width=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Storage per Identity (GB, log scale)")
    ax.set_title("Storage Cost per Identity")
    ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.35, zorder=0)

    labels_str = ["~2.6 GB", "~120 KB", "~50 KB (ours)"]
    for bar, lbl in zip(bars, labels_str):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.6,
                lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "chart_storage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ── Figure 5: Speed Comparison ────────────────────────────────────────────────
def chart_speed():
    methods = ["DreamBooth\n(fine-tune)", "HyperDreamBooth\n(image)", "VideoEditPG\n(ours, video)"]
    times   = [600, 18, 35]
    colors  = [GRAY, BLUE, PURPLE]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(methods, times, color=colors, width=0.5, zorder=3)
    ax.set_ylabel("Personalization Time (s)")
    ax.set_title("Personalization Speed at Inference\n(single subject, single pass)")
    ax.set_ylim(0, max(times) * 1.25)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    for bar, t in zip(bars, times):
        label = f"{t}s" if t < 100 else f"{t//60}m"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4,
                label, ha="center", va="bottom", fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "chart_speed.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


if __name__ == "__main__":
    print("Generating figures...")
    chart_gen_times()
    chart_lora_overhead()
    chart_lora_params()
    chart_storage()
    chart_speed()
    print("Done. All charts saved to report/assets/")
