"""Paper-style plotting for the supplementary cube demo.

The figure layout mirrors the manuscript examples: Poisson on the top row,
biharmonic on the bottom row, residual-versus-time on the left, and stacked
timing bars on the right.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PAPER_FONT_FAMILY = "serif"
PAPER_FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Nimbus Roman No9 L",
    "Liberation Serif",
    "DejaVu Serif",
]
PAPER_USE_TEX = False
PAPER_FIGURE_DPI = 300
PAPER_FIGURE_WIDTH = 3.35
PAPER_FIGURE_HEIGHT = 2.5
PAPER_LINE_WIDTH = 1.5
PAPER_LINE_ALPHA = 0.85
PAPER_ITERATION_MARKER = "o"
PAPER_ITERATION_MARKER_SIZE_MULTIPLIER = 1.3
PAPER_MAIN_METHOD_OUTLINE_COLOR = "white"
PAPER_MAIN_METHOD_OUTLINE_WIDTH = 1.5
PAPER_MAIN_METHOD_OUTLINE_ALPHA = 0.5
PAPER_MAIN_METHOD_LINE_WIDTH_MULTIPLIER = 1.3
PAPER_FONT_SIZE = 9
PAPER_LABEL_SIZE = 9
PAPER_TITLE_SIZE = 10
PAPER_LEGEND_SIZE = 8
PAPER_TICK_SIZE = 8
REFERENCE_LINE_WIDTH = 1.0
OURS_DARK = "#00441b"
OURS_MEDIUM = "#238b45"
OURS_LIGHT = "#74c476"


def apply_paper_style() -> None:
    """Match the paper-style plotting defaults used in the main project."""
    plt.rcParams.update({
        "font.family": PAPER_FONT_FAMILY,
        "font.serif": PAPER_FONT_SERIF,
        "text.usetex": PAPER_USE_TEX,
        "mathtext.fontset": "cm",
        "font.size": PAPER_FONT_SIZE,
        "axes.labelsize": PAPER_LABEL_SIZE,
        "axes.titlesize": PAPER_TITLE_SIZE,
        "legend.fontsize": PAPER_LEGEND_SIZE,
        "xtick.labelsize": PAPER_TICK_SIZE,
        "ytick.labelsize": PAPER_TICK_SIZE,
        "figure.dpi": PAPER_FIGURE_DPI,
        "savefig.dpi": PAPER_FIGURE_DPI,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": PAPER_LINE_WIDTH,
    })


def _plot_convergence(ax, result: dict, title: str) -> None:
    """Plot relative residual against cumulative solver time."""
    residual_history = np.asarray(result["residual_history"], dtype=float)
    cycle_time_ms_history = np.asarray(result.get("cycle_time_ms_history", []), dtype=float)

    if residual_history.size == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes)
        return

    relative = residual_history / max(residual_history[0], 1e-30)
    relative = np.where(relative > 0.0, relative, 1e-16)

    time_points = np.concatenate(([0.0], np.cumsum(cycle_time_ms_history) / 1000.0))
    if time_points.size > relative.size:
        time_points = time_points[: relative.size]
    elif time_points.size < relative.size:
        # Fallback should not happen, but preserve a valid plot if it does.
        pad = np.full(relative.size - time_points.size, time_points[-1] if time_points.size else 0.0)
        time_points = np.concatenate((time_points, pad))

    lw = PAPER_LINE_WIDTH * PAPER_MAIN_METHOD_LINE_WIDTH_MULTIPLIER
    ms = PAPER_LINE_WIDTH * PAPER_ITERATION_MARKER_SIZE_MULTIPLIER * PAPER_MAIN_METHOD_LINE_WIDTH_MULTIPLIER
    pe = [
        path_effects.Stroke(
            linewidth=lw + PAPER_MAIN_METHOD_OUTLINE_WIDTH,
            foreground=PAPER_MAIN_METHOD_OUTLINE_COLOR,
            alpha=PAPER_MAIN_METHOD_OUTLINE_ALPHA,
        ),
        path_effects.Normal(),
    ]

    ax.plot(
        time_points,
        relative,
        color=OURS_MEDIUM,
        linewidth=lw,
        alpha=PAPER_LINE_ALPHA,
        label="Ours",
        zorder=5,
        path_effects=pe,
    )
    ax.plot(
        time_points,
        relative,
        linestyle="None",
        marker=PAPER_ITERATION_MARKER,
        markersize=ms,
        color=OURS_MEDIUM,
        alpha=PAPER_LINE_ALPHA,
        zorder=6,
    )

    tol = float(result["tolerance"])
    ax.axhline(tol, color="gray", linestyle="--", linewidth=REFERENCE_LINE_WIDTH * 0.8, alpha=0.6)
    exp = int(np.floor(np.log10(tol))) if tol > 0 else 0
    tol_text = fr"$\epsilon = 10^{{{exp}}}$"
    bottom_limit_here = 10 ** (np.floor(np.log10(tol)) - 1) if tol > 0 else 1e-10
    mid_y = np.sqrt(max(tol, 1e-16) * bottom_limit_here)
    max_time = time_points[-1] if time_points.size else 1.0
    tol_x = max_time * 0.03
    tol_y = max(tol * 2.0, 1e-16)
    ax.text(
        tol_x,
        tol_y,
        tol_text,
        ha="left",
        va="bottom",
        fontsize=PAPER_FONT_SIZE - 1,
        color="gray",
        alpha=0.9,
        zorder=15,
        path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
    )

    ax.set_title(title)
    ax.set_xlabel("Solver Time [s]")
    ax.set_ylabel("Relative Residual")
    ax.set_yscale("log", base=10)

    min_exp = int(np.floor(np.log10(max(min(relative.min(), tol), 1e-16))))
    ticks = [10.0 ** exp for exp in range(0, min_exp - 1, -1)]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.set_ylim(max(10.0 ** (min_exp - 0.5), 1e-16), 2.0)

    iteration_marker = Line2D(
        [0], [0], linestyle="None", marker=PAPER_ITERATION_MARKER, color="gray",
        label="Iteration", markerfacecolor="gray", markeredgecolor="gray",
        markersize=PAPER_LINE_WIDTH * PAPER_ITERATION_MARKER_SIZE_MULTIPLIER * 2,
    )
    leg = ax.legend(handles=[ax.lines[0], iteration_marker], labels=["Ours", "Iteration"], loc="upper right", framealpha=0.8)
    leg.set_zorder(20)


def _plot_timing(ax, result: dict, title: str, annotate_reuse: bool, shared_hierarchy_ms: float) -> None:
    """Plot a single stacked computation-time histogram in paper style."""
    hierarchy_s = shared_hierarchy_ms / 1000.0
    setup_s = float(result["setup_time_ms"]) / 1000.0
    solve_s = float(result["total_time_ms"]) / 1000.0
    total_s = hierarchy_s + setup_s + solve_s

    segments = [
        (hierarchy_s, OURS_DARK, "Hierarchy"),
        (setup_s, OURS_MEDIUM, "Setup"),
        (solve_s, OURS_LIGHT, "Solve"),
    ]

    x_pos = -0.18
    bar_width = 0.30
    bottom = 0.0
    for height, color, _label in segments:
        if height <= 0:
            continue
        ax.bar(x_pos, height, width=bar_width, bottom=bottom, color=color, zorder=3)
        ax.text(
            x_pos,
            bottom + height / 2.0,
            f"{height:.2f}",
            ha="center",
            va="center",
            color="white",
            fontsize=PAPER_FONT_SIZE - 2,
            fontweight="bold",
            zorder=4,
        )
        bottom += height

    ax.text(
        x_pos,
        total_s,
        f"{total_s:.2f}",
        ha="center",
        va="bottom",
        fontsize=PAPER_FONT_SIZE - 1,
        fontweight="bold",
        path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
    )

    ax.set_title(title)
    ax.set_ylabel("Time [s]")
    ax.set_xticks([x_pos])
    ax.set_xticklabels([f"Ours\n#It: {result['num_cycles']}"])
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_ylim(0.0, max(total_s * 1.20, 0.05))
    ax.set_xlim(-0.45, 0.45)

    leg = ax.legend(
        handles=[
            Patch(facecolor=OURS_DARK, label="Hierarchy"),
            Patch(facecolor=OURS_MEDIUM, label="Setup"),
            Patch(facecolor=OURS_LIGHT, label="Solve"),
        ],
        loc="upper right",
        fontsize=PAPER_LEGEND_SIZE,
        framealpha=0.8,
    )
    leg.set_zorder(20)


def save_combined_figure(summary: dict, output_dir: Path) -> Path:
    """Save the combined 2x2 supplementary figure."""
    apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(PAPER_FIGURE_WIDTH * 2.0, PAPER_FIGURE_HEIGHT * 2.0),
        constrained_layout=True,
    )

    poisson = summary["problems"]["poisson"]
    biharmonic = summary["problems"]["biharmonic"]
    shared_hierarchy_ms = float(summary["hierarchy_build_ms"])

    _plot_convergence(axes[0, 0], poisson, "Poisson")
    _plot_timing(axes[0, 1], poisson, "Poisson", annotate_reuse=False, shared_hierarchy_ms=shared_hierarchy_ms)
    _plot_convergence(axes[1, 0], biharmonic, "Biharmonic")
    _plot_timing(axes[1, 1], biharmonic, "Biharmonic", annotate_reuse=True, shared_hierarchy_ms=shared_hierarchy_ms)

    num_verts = summary["problems"]["poisson"]["num_vertices"]
    if num_verts >= 1_000_000:
        verts_str = f"{num_verts / 1_000_000:.1f}M"
    elif num_verts >= 1_000:
        verts_str = f"{num_verts // 1_000}k"
    else:
        verts_str = str(num_verts)

    fig.suptitle(
        f"Cube{summary['resolution']}  |  #Vertices = {verts_str}",
        fontsize=PAPER_TITLE_SIZE + 2,
        fontweight="bold",
        y=1.01,
    )

    stem = output_dir / f"combined_cube{summary['resolution']}"
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path
