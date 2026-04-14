#!/usr/bin/env python3
"""Generate overview charts for the performance overview page.

Reads benchmark data from the generated MDX files and produces PNG charts
in both light and dark themes for Mintlify.

Usage:
    python3 scripts/generate-overview-charts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import numpy as np


# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------

THEMES = {
    "light": {
        "bg": "#ffffff",
        "text": "#111827",
        "muted": "#6b7280",
        "grid": "#e5e7eb",
        "baseline": "#9ca3af",
        "faster": "#3179C7",
        "slower": "#ef4444",
        "neutral": "#f59e0b",
        "accent_blue": "#3b82f6",
        "accent_teal": "#14b8a6",
        "accent_purple": "#8b5cf6",
        # dtype color families
        "float": "#3b6cc9",
        "int_signed": "#2a8a82",
        "int_unsigned": "#c98a2e",
        "complex": "#8b5cc6",
    },
    "dark": {
        "bg": "#1c1c1c",
        "text": "#d4d4d4",
        "muted": "#888888",
        "grid": "#2a2a2a",
        "baseline": "#555555",
        "faster": "#5a9fd4",
        "slower": "#ef4444",
        "neutral": "#f59e0b",
        "accent_blue": "#60a5fa",
        "accent_teal": "#2dd4bf",
        "accent_purple": "#a78bfa",
        "float": "#6196e2",
        "int_signed": "#55bfb7",
        "int_unsigned": "#e4b85e",
        "complex": "#a478d4",
    },
}

DTYPE_COLOR_MAP = {
    "float64": "float", "float32": "float", "float16": "float",
    "int64": "int_signed", "int32": "int_signed", "int16": "int_signed", "int8": "int_signed",
    "uint64": "int_unsigned", "uint32": "int_unsigned", "uint16": "int_unsigned", "uint8": "int_unsigned",
    "complex128": "complex", "complex64": "complex",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_data_from_mdx(mdx_path: Path) -> dict:
    """Extract the benchmarkData JSON from an MDX file."""
    text = mdx_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("export const benchmarkData"):
            blob = line.split(" = ", 1)[1].rstrip(";")
            return json.loads(blob)
    raise ValueError(f"No benchmarkData found in {mdx_path}")


# ---------------------------------------------------------------------------
# Chart 1: Category speedup (diverging horizontal bars)
# ---------------------------------------------------------------------------

def chart_category_speedup(categories: list[dict], theme: dict, out_path: Path) -> None:
    """Diverging horizontal bar chart of category speedups vs NumPy."""
    # Sort by speedup descending
    cats = sorted(categories, key=lambda c: c["avgSpeedup"])

    names = [c["name"] for c in cats]
    speedups = [c["avgSpeedup"] for c in cats]

    fig, ax = plt.subplots(figsize=(9.5, 6.8))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])

    y_pos = np.arange(len(names))
    # Bars: diverge from 1.0x
    offsets = [s - 1.0 for s in speedups]
    colors = [theme["faster"] if s >= 1.0 else theme["slower"] for s in speedups]

    bars = ax.barh(y_pos, offsets, left=1.0, height=0.65, color=colors, zorder=3,
                   edgecolor="none", alpha=0.85)

    # Add value labels at bar tips — for slower bars, place label outside (left of bar)
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        if speedup >= 1.0:
            ax.text(speedup + 0.04, i, f"{speedup:.2f}x", va="center", ha="left",
                    fontsize=9, fontweight="600", color=theme["text"], zorder=4)
        else:
            # Place label to the right of the bar tip (between bar and baseline)
            ax.text(speedup + 0.04, i, f"{speedup:.2f}x", va="center", ha="left",
                    fontsize=9, fontweight="600", color=theme["text"], zorder=4)

    # Baseline at 1.0x
    ax.axvline(x=1.0, color=theme["baseline"], linewidth=1.5, zorder=2, linestyle="-")

    ax.set_ylim(-0.5, len(names) - 0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, color=theme["text"])

    # X-axis — add padding so labels don't clip
    x_min = min(speedups) - 0.08
    x_max = max(speedups) + 0.25
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Speedup vs NumPy", fontsize=10, color=theme["muted"], labelpad=8)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))
    ax.tick_params(axis="x", colors=theme["muted"], labelsize=9)

    # Grid
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.5, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    # Spine cleanup
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=theme["faster"], alpha=0.85, label="numpy-ts faster"),
        Patch(facecolor=theme["slower"], alpha=0.85, label="NumPy faster"),
    ]
    leg = ax.legend(handles=legend_elements, loc="lower right", frameon=True,
                    fontsize=9, facecolor=theme["bg"], edgecolor=theme["grid"])
    for text in leg.get_texts():
        text.set_color(theme["text"])

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=theme["bg"],
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  {out_path}")


# ---------------------------------------------------------------------------
# Chart 2: DType performance (horizontal bars)
# ---------------------------------------------------------------------------

def chart_dtype_speedup(dtype_stats: list[dict], theme: dict, out_path: Path) -> None:
    """Horizontal bar chart of dtype speedups vs NumPy."""
    dtypes = list(reversed(dtype_stats))  # reversed so highest is at top

    names = [d["dtype"] for d in dtypes]
    speedups = [d["avgSpeedup"] for d in dtypes]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])

    y_pos = np.arange(len(names))
    offsets = [s - 1.0 for s in speedups]
    colors = [theme["faster"] if s >= 1.0 else theme["slower"] for s in speedups]

    bars = ax.barh(y_pos, offsets, left=1.0, height=0.6, color=colors,
                   zorder=3, edgecolor="none", alpha=0.85)

    # Value labels — always to the right of bar tip
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        ax.text(speedup + 0.02, i, f"{speedup:.2f}x", va="center", ha="left",
                fontsize=9, fontweight="600", color=theme["text"], zorder=4)

    # Baseline
    ax.axvline(x=1.0, color=theme["baseline"], linewidth=1.5, zorder=2, linestyle="-")
    ax.set_ylim(-0.5, len(names) - 0.5)

    x_min = min(speedups) - 0.08
    x_max = max(speedups) + 0.2
    ax.set_xlim(x_min, x_max)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, fontfamily="monospace", color=theme["text"])
    ax.set_xlabel("Speedup vs NumPy", fontsize=10, color=theme["muted"], labelpad=8)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))
    ax.tick_params(axis="x", colors=theme["muted"], labelsize=9)

    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.5, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=theme["faster"], alpha=0.85, label="numpy-ts faster"),
        Patch(facecolor=theme["slower"], alpha=0.85, label="NumPy faster"),
    ]
    leg = ax.legend(handles=legend_elements, loc="lower right", frameon=True,
                    fontsize=9, facecolor=theme["bg"], edgecolor=theme["grid"])
    for text in leg.get_texts():
        text.set_color(theme["text"])

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=theme["bg"],
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  {out_path}")


# ---------------------------------------------------------------------------
# Chart 3: Size scaling (grouped bars)
# ---------------------------------------------------------------------------

def chart_size_scaling(sizes: list[dict], theme: dict, out_path: Path) -> None:
    """Grouped bar chart: NumPy (1.0x baseline) vs numpy-ts at each array size."""
    labels = [s["label"] for s in sizes]
    speedups = [s["summary"]["avgSpeedup"] for s in sizes]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor(theme["bg"])
    ax.set_facecolor(theme["bg"])

    x_pos = np.arange(len(labels))
    bar_width = 0.32

    # NumPy bars (always 1.0x)
    numpy_bars = ax.bar(x_pos - bar_width / 2, [1.0] * len(labels), width=bar_width,
                        color=theme["slower"], zorder=3, edgecolor="none", alpha=0.7,
                        label="NumPy")

    # numpy-ts bars
    npts_bars = ax.bar(x_pos + bar_width / 2, speedups, width=bar_width,
                       color=theme["faster"], zorder=3, edgecolor="none", alpha=0.85,
                       label="numpy-ts")

    # Value labels on numpy-ts bars
    for i, (bar, speedup) in enumerate(zip(npts_bars, speedups)):
        ax.text(bar.get_x() + bar.get_width() / 2, speedup + 0.02,
                f"{speedup:.2f}x", ha="center", va="bottom",
                fontsize=11, fontweight="700", color=theme["text"], zorder=4)

    # "1.0x" labels on NumPy bars
    for bar in numpy_bars:
        ax.text(bar.get_x() + bar.get_width() / 2, 1.0 + 0.02,
                "1.0x", ha="center", va="bottom",
                fontsize=9, fontweight="600", color=theme["muted"], zorder=4)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10, color=theme["text"])
    ax.set_ylabel("Relative Performance", fontsize=10, color=theme["muted"], labelpad=8)
    ax.tick_params(axis="y", colors=theme["muted"], labelsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fx"))

    ax.yaxis.grid(True, color=theme["grid"], linewidth=0.5, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    ax.set_ylim(0, max(speedups) * 1.2)

    for spine in ax.spines.values():
        spine.set_visible(False)

    leg = ax.legend(loc="upper left", frameon=True, fontsize=9,
                    facecolor=theme["bg"], edgecolor=theme["grid"])
    for text in leg.get_texts():
        text.set_color(theme["text"])

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=theme["bg"],
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    # Detect version
    docs_config = repo_root / "docs" / "docs.json"
    version = "v1.3.x"
    if docs_config.exists():
        try:
            data = json.loads(docs_config.read_text(encoding="utf-8"))
            versions = data.get("navigation", {}).get("versions", [])
            if versions and isinstance(versions[0], dict):
                version = versions[0]["version"]
        except Exception:
            pass

    perf_dir = repo_root / "docs" / version / "performance"
    out_dir = repo_root / "docs" / "assets" / version

    # Load data
    vs_numpy = extract_data_from_mdx(perf_dir / "vs-numpy.mdx")
    size_scaling = extract_data_from_mdx(perf_dir / "size-scaling.mdx")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating overview charts for {version}...")
    for theme_name, theme in THEMES.items():
        suffix = f"-{theme_name}"
        chart_category_speedup(
            vs_numpy["categories"], theme,
            out_dir / f"overview-categories{suffix}.png",
        )
        chart_dtype_speedup(
            vs_numpy["dtypeStats"], theme,
            out_dir / f"overview-dtypes{suffix}.png",
        )
        chart_size_scaling(
            size_scaling["sizes"], theme,
            out_dir / f"overview-scaling{suffix}.png",
        )

    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
