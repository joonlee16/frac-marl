"""Plot averaged reward curves from the per-seed .npy files saved by train.py.

Expected filenames:
    <mode><seed>.npy

Examples:
    naive0.npy
    normal3.npy
    resilient1.npy

Each file should contain a 1D reward vector over training episodes. The script groups
files in --results-dir by mode (the filename prefix), averages across seeds, smooths
with a moving-average window, and draws one curve per mode with a shaded uncertainty
band (std by default) -- e.g. all of rewards/F=1/*.npy onto one figure.

Usage:
    python plot_reward.py --results-dir rewards/F=1 --out reward_summary.svg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


FILE_PATTERN = re.compile(r"^(?P<mode>[A-Za-z_]+?)(?P<seed>\d+)\.npy$")
MODE_COLORS = {
    "normal": "green",
    "resilient": "blue",
    "naive": "red",
    "projection": "orange",
    "trimmed_mean": "purple",
}
MODE_LINESTYLES = {
    "resilient": (0, (10, 6)),
}
MODE_MARKERS = {
}
DISPLAY_LABELS = {
    "normal": "Normal",
    "naive": "Naive",
    "projection": "Projection",
    "trimmed_mean": "Trimmed Mean",
    "resilient": "Ours",
}
PREFERRED_ORDER = ["normal", "naive", "projection", "trimmed_mean", "resilient"]


def apply_openai_article_style() -> None:
    """Set a clean, article-like plotting style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#D0D7DE",
        "axes.labelcolor": "#111827",
        "text.color": "#111827",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "axes.titleweight": "regular",
        "axes.labelsize": 17,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.frameon": False,
        "grid.color": "#E5E7EB",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    })


def load_mode_curves(results_dir: Path) -> Dict[str, List[np.ndarray]]:
    """Load all reward vectors in a directory and group them by mode name."""
    grouped: Dict[str, List[np.ndarray]] = {}
    for path in sorted(results_dir.glob("*.npy")):
        match = FILE_PATTERN.match(path.name)
        if match is None:
            continue
        mode = match.group("mode")
        arr = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
        grouped.setdefault(mode, []).append(arr)
    return grouped


def align_curves(curves: List[np.ndarray], align: str = "min") -> np.ndarray:
    """Stack curves into a 2D array after aligning lengths.

    align='min' truncates every curve to the shortest length.
    align='max' pads shorter curves with NaN values.
    """
    if not curves:
        raise ValueError("No curves to align.")

    if align == "min":
        length = min(len(c) for c in curves)
        return np.vstack([c[:length] for c in curves])

    if align == "max":
        length = max(len(c) for c in curves)
        stacked = np.full((len(curves), length), np.nan, dtype=np.float64)
        for i, curve in enumerate(curves):
            stacked[i, : len(curve)] = curve
        return stacked

    raise ValueError("align must be 'min' or 'max'")


def convolve_smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Smooth a 1D curve with a normalized convolution kernel.

    This is NaN-aware, so it works even when curves are aligned with padding.
    """
    if window <= 1:
        return x

    kernel = np.ones(window, dtype=np.float64)
    left = window // 2
    right = window - 1 - left

    x = np.asarray(x, dtype=np.float64)
    valid = np.isfinite(x).astype(np.float64)
    x_filled = np.nan_to_num(x, nan=0.0)

    num = np.convolve(np.pad(x_filled, (left, right), mode="constant", constant_values=0.0),
                      kernel, mode="valid")
    den = np.convolve(np.pad(valid, (left, right), mode="constant", constant_values=0.0),
                      kernel, mode="valid")
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def summarize_curves(curves: List[np.ndarray], align: str = "min") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, mean-std, mean+std) across seeds. Currently unused -- plot_results()
    below reimplements this inline so it can also support 'sem'/'minmax' bands; kept
    here as a simple std-only helper in case a caller just wants that."""
    stacked = align_curves(curves, align=align)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    return mean, mean - std, mean + std


def plot_results(
    results_dir: str,
    out_path: str = "reward_summary.svg",
    shade: str = "std",
    align: str = "min",
    smooth_window: int = 100,
    title: str | None = None,
) -> str:
    """Plot mean reward curves per mode with a shaded uncertainty band, one line per
    mode found in results_dir (see load_mode_curves), smoothed by convolve_smooth and
    saved to out_path (format taken from its extension). Raises FileNotFoundError if
    no *.npy file in results_dir matches FILE_PATTERN.

    shade options:
        - std: mean ± 1 standard deviation
        - sem: mean ± standard error of the mean
        - minmax: min/max envelope across seeds

    If title is None, defaults to "(a) Reward Curves with F=1" -- a leftover from one
    specific F=1 figure; pass --title explicitly when plotting any other F_BUDGET_G.
    """
    results_path = Path(results_dir)
    grouped = load_mode_curves(results_path)
    if not grouped:
        raise FileNotFoundError(f"No matching .npy files found in {results_path}")

    apply_openai_article_style()
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=180)

    ordered_modes = [mode for mode in PREFERRED_ORDER if mode in grouped]
    ordered_modes.extend(sorted(mode for mode in grouped if mode not in PREFERRED_ORDER))

    for mode in ordered_modes:
        curves = grouped[mode]
        stacked = align_curves(curves, align=align)
        color = MODE_COLORS.get(mode, None)

        mean = np.nanmean(stacked, axis=0)
        if shade == "std":
            spread = np.nanstd(stacked, axis=0)
            lower, upper = mean - spread, mean + spread
            band_label = "±1 std"
        elif shade == "sem":
            spread = np.nanstd(stacked, axis=0) / np.sqrt(np.sum(~np.isnan(stacked), axis=0))
            lower, upper = mean - spread, mean + spread
            band_label = "±1 sem"
        elif shade == "minmax":
            lower = np.nanmin(stacked, axis=0)
            upper = np.nanmax(stacked, axis=0)
            band_label = "min-max"
        else:
            raise ValueError("shade must be one of: std, sem, minmax")

        if smooth_window > 1:
            mean = convolve_smooth(mean, smooth_window)
            lower = convolve_smooth(lower, smooth_window)
            upper = convolve_smooth(upper, smooth_window)

        x = np.arange(len(mean))
        linestyle = MODE_LINESTYLES.get(mode, "-")
        marker = MODE_MARKERS.get(mode, None)
        markevery = max(1, len(mean) // 18) if marker is not None else None
        linewidth = 2.8 if mode == "normal" else 2.4 if mode == "resilient" else 2.5
        ax.plot(x, mean, linewidth=linewidth, color=color, linestyle=linestyle,
                marker=marker, markevery=markevery, markersize=4.5,
                label=DISPLAY_LABELS.get(mode, mode), rasterized=True)
        ax.fill_between(x, lower, upper, alpha=0.1, color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, alpha=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#D0D7DE")
    ax.spines["bottom"].set_color("#D0D7DE")
    ax.margins(x=0)
    if title is None:
        title = "(a) Reward Curves with F=1"
    fig.suptitle(title, x=0.5, y=0.98, ha="center", fontsize=17, fontweight="bold")
    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=max(1, len(ordered_modes)),
        fontsize=12,
        frameon=False,
        columnspacing=1.6,
        handlelength=2.8,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out_path, format=Path(out_path).suffix.lstrip("."), bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    """CLI entry point: parse args and call plot_results()."""
    parser = argparse.ArgumentParser(description="Plot averaged reward curves across seeds.")
    parser.add_argument("--results-dir", default="rewards/F=1", help="Directory containing mode+seed .npy files")
    parser.add_argument("--out", default="reward_summary.svg", help="Output figure path")
    parser.add_argument("--shade", default="std", choices=["std", "sem", "minmax"], help="Shaded uncertainty band")
    parser.add_argument("--align", default="min", choices=["min", "max"], help="How to align curves of different lengths")
    parser.add_argument("--smooth-window", type=int, default=100, help="Convolution window for smoothing the curves")
    parser.add_argument("--title", default=None, help="Figure title")
    args = parser.parse_args()

    out = plot_results(
        results_dir=args.results_dir,
        out_path=args.out,
        shade=args.shade,
        align=args.align,
        smooth_window=args.smooth_window,
        title=args.title,
    )
    print(f"Saved plot -> {out}")


if __name__ == "__main__":
    main()
