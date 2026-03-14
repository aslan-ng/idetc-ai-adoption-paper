"""
Plot minimum required AI accuracy to reach a target adoption level.

What this file does:
1. Uses the surrogate ODE model to predict steady-state adoption (Q + L).
2. For each organizational structure setting, finds the smallest
   `technology_success_rate` that achieves a chosen target adoption.
3. Caches the resulting grids to CSV so layout iteration is fast.
4. Visualizes that threshold as heatmaps over:
   - number of teams
   - team size
5. Saves a publication-ready figure to `figures/`.

How it is used in the pipeline:
- Supports decision analysis by translating adoption targets into minimum
  AI-performance requirements under different organizational structures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict_surrogate import predict_full_surrogate_output
from utils import BASE_DIR


# ---- Global plotting style for publication figures ----
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 13,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 15,
    "figure.titlesize": 17,
})
HEATMAP_FONT_SIZE = 11

FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / "figures" / "cached_min_accuracy"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def predict_adoption(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    surrogate_path: Path | None = None,
) -> float:
    out = predict_full_surrogate_output(
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
        surrogate_path=surrogate_path,
    )
    ss = out["steady_state"]
    return float(ss["Q"] + ss["L"])


def find_min_accuracy_for_target_adoption(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    target_adoption: float,
    surrogate_path: Path | None = None,
    accuracy_grid: np.ndarray | None = None,
) -> float:
    """
    Return the minimum technology_success_rate in [0, 1] such that
    adoption >= target_adoption.

    Returns np.nan if the target is not reached anywhere on the grid.
    """
    if accuracy_grid is None:
        accuracy_grid = np.linspace(0.0, 1.0, 1001)

    for acc in accuracy_grid:
        adoption = predict_adoption(
            teams_num=teams_num,
            teams_size=teams_size,
            agents_average_initial_opinion=agents_average_initial_opinion,
            technology_success_rate=float(acc),
            surrogate_path=surrogate_path,
        )
        if adoption >= target_adoption:
            return float(acc)

    return float("nan")


def compute_min_accuracy_grid(
    *,
    agents_average_initial_opinion: float,
    target_adoption: float,
    teams_num_list: list[int],
    teams_size_list: list[int],
    surrogate_path: Path | None = None,
    accuracy_grid: np.ndarray | None = None,
) -> np.ndarray:
    Z = np.zeros((len(teams_num_list), len(teams_size_list)), dtype=float)

    for i, teams_num in enumerate(teams_num_list):
        for j, teams_size in enumerate(teams_size_list):
            Z[i, j] = find_min_accuracy_for_target_adoption(
                teams_num=float(teams_num),
                teams_size=float(teams_size),
                agents_average_initial_opinion=float(agents_average_initial_opinion),
                target_adoption=float(target_adoption),
                surrogate_path=surrogate_path,
                accuracy_grid=accuracy_grid,
            )

    return Z


def _grid_cache_filename(
    *,
    target_adoption: float,
    negative_initial_opinion: float,
    positive_initial_opinion: float,
    teams_num_list: list[int],
    teams_size_list: list[int],
) -> str:
    teams_num_str = "_".join(map(str, teams_num_list))
    teams_size_str = "_".join(map(str, teams_size_list))
    return (
        f"min_accuracy_cache_"
        f"target_{target_adoption:.3f}_"
        f"neg_{negative_initial_opinion:.3f}_"
        f"pos_{positive_initial_opinion:.3f}_"
        f"teamsnum_{teams_num_str}_"
        f"teamssize_{teams_size_str}.csv"
    )


def save_min_accuracy_cache(
    *,
    target_adoption: float,
    negative_initial_opinion: float,
    positive_initial_opinion: float,
    teams_num_list: list[int],
    teams_size_list: list[int],
    surrogate_path: Path | None = None,
    accuracy_grid: np.ndarray | None = None,
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"Cache not found. Computing grids and saving to: {cache_path}")

    Z_neg = compute_min_accuracy_grid(
        agents_average_initial_opinion=negative_initial_opinion,
        target_adoption=target_adoption,
        teams_num_list=teams_num_list,
        teams_size_list=teams_size_list,
        surrogate_path=surrogate_path,
        accuracy_grid=accuracy_grid,
    )

    Z_pos = compute_min_accuracy_grid(
        agents_average_initial_opinion=positive_initial_opinion,
        target_adoption=target_adoption,
        teams_num_list=teams_num_list,
        teams_size_list=teams_size_list,
        surrogate_path=surrogate_path,
        accuracy_grid=accuracy_grid,
    )

    rows: list[dict[str, float | int | str]] = []
    for i, teams_num in enumerate(teams_num_list):
        for j, teams_size in enumerate(teams_size_list):
            rows.append(
                {
                    "teams_num": teams_num,
                    "teams_size": teams_size,
                    "min_accuracy_negative": Z_neg[i, j],
                    "min_accuracy_positive": Z_pos[i, j],
                }
            )

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)

    print(f"Saved cache to: {cache_path}")
    return Z_neg, Z_pos


def load_min_accuracy_cache(
    *,
    cache_path: Path,
    teams_num_list: list[int],
    teams_size_list: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(cache_path)

    Z_neg = np.full((len(teams_num_list), len(teams_size_list)), np.nan, dtype=float)
    Z_pos = np.full((len(teams_num_list), len(teams_size_list)), np.nan, dtype=float)

    for i, teams_num in enumerate(teams_num_list):
        for j, teams_size in enumerate(teams_size_list):
            match = df[
                (df["teams_num"] == teams_num) &
                (df["teams_size"] == teams_size)
            ]
            if match.empty:
                raise ValueError(
                    f"Missing cached value for teams_num={teams_num}, teams_size={teams_size}"
                )

            Z_neg[i, j] = float(match.iloc[0]["min_accuracy_negative"])
            Z_pos[i, j] = float(match.iloc[0]["min_accuracy_positive"])

    print(f"Loaded cache from: {cache_path}")
    return Z_neg, Z_pos


def get_or_create_min_accuracy_cache(
    *,
    target_adoption: float,
    negative_initial_opinion: float,
    positive_initial_opinion: float,
    teams_num_list: list[int],
    teams_size_list: list[int],
    surrogate_path: Path | None = None,
    accuracy_grid: np.ndarray | None = None,
    force_recompute: bool = False,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, Path]:
    if cache_path is None:
        cache_name = _grid_cache_filename(
            target_adoption=target_adoption,
            negative_initial_opinion=negative_initial_opinion,
            positive_initial_opinion=positive_initial_opinion,
            teams_num_list=teams_num_list,
            teams_size_list=teams_size_list,
        )
        cache_path = CACHE_DIR / cache_name

    if cache_path.exists() and not force_recompute:
        Z_neg, Z_pos = load_min_accuracy_cache(
            cache_path=cache_path,
            teams_num_list=teams_num_list,
            teams_size_list=teams_size_list,
        )
    else:
        Z_neg, Z_pos = save_min_accuracy_cache(
            target_adoption=target_adoption,
            negative_initial_opinion=negative_initial_opinion,
            positive_initial_opinion=positive_initial_opinion,
            teams_num_list=teams_num_list,
            teams_size_list=teams_size_list,
            surrogate_path=surrogate_path,
            accuracy_grid=accuracy_grid,
            cache_path=cache_path,
        )

    return Z_neg, Z_pos, cache_path


def plot_min_accuracy_two_panel(
    *,
    target_adoption: float = 0.5,
    negative_initial_opinion: float = -0.25,
    positive_initial_opinion: float = 0.25,
    teams_num_list: list[int] | None = None,
    teams_size_list: list[int] | None = None,
    surrogate_path: Path | None = None,
    save_name: str | None = None,
    color_min: float = 0.6,
    color_max: float = 1.0,
    force_recompute_cache: bool = False,
    cache_path: Path | None = None,
) -> None:
    if teams_num_list is None:
        teams_num_list = [1, 5, 10, 15, 20, 25, 30]

    if teams_size_list is None:
        teams_size_list = [5, 10, 15, 20]

    accuracy_grid = np.linspace(0.0, 1.0, 1001)

    Z_neg, Z_pos, used_cache_path = get_or_create_min_accuracy_cache(
        target_adoption=target_adoption,
        negative_initial_opinion=negative_initial_opinion,
        positive_initial_opinion=positive_initial_opinion,
        teams_num_list=teams_num_list,
        teams_size_list=teams_size_list,
        surrogate_path=surrogate_path,
        accuracy_grid=accuracy_grid,
        force_recompute=force_recompute_cache,
        cache_path=cache_path,
    )

    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1, 1, 0.05],
        left=0.10,
        right=0.92,
        bottom=0.14,
        top=0.82,
        wspace=0.03,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    cbar_ax = fig.add_subplot(gs[0, 2])

    axes = [ax0, ax1]

    cmap = "viridis"

    im0 = axes[0].imshow(
        Z_neg,
        origin="lower",
        aspect="equal",
        vmin=color_min,
        vmax=color_max,
        cmap=cmap,
    )

    im1 = axes[1].imshow(
        Z_pos,
        origin="lower",
        aspect="equal",
        vmin=color_min,
        vmax=color_max,
        cmap=cmap,
    )

    subplot_titles = ["Negative initial opinion", "Positive initial opinion"]
    grids = [Z_neg, Z_pos]
    threshold = (color_min + color_max) / 2.0

    for ax, Z, subtitle in zip(axes, grids, subplot_titles):
        ax.set_title(subtitle, fontweight="bold")
        ax.set_xticks(range(len(teams_size_list)))
        ax.set_xticklabels(teams_size_list)
        ax.set_yticks(range(len(teams_num_list)))
        ax.set_yticklabels(teams_num_list)
        ax.set_xlabel("Team size")

        for i in range(len(teams_num_list)):
            for j in range(len(teams_size_list)):
                val = Z[i, j]
                label = "NA" if np.isnan(val) else f"{val:.2f}"
                text_color = "black" if (not np.isnan(val) and val > threshold) else "white"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=HEATMAP_FONT_SIZE,
                    color=text_color,
                )

    axes[0].set_ylabel("Number of teams")

    cbar = fig.colorbar(
        im1,
        cax=cbar_ax,
        ticks=np.arange(0.6, 1.01, 0.1),
    )
    cbar.set_label("Minimum Required AI Accuracy")

    fig.suptitle(
        "Minimum AI accuracy for 50% adoption",
        y=0.94,
    )

    if save_name is not None:
        save_path = FIGURES_DIR / save_name
        fig.savefig(save_path, bbox_inches="tight", dpi=400)
        print(f"Saved figure to: {save_path}")

    plt.close(fig)

    print(f"Used cache file: {used_cache_path}")


if __name__ == "__main__":
    plot_min_accuracy_two_panel(
        target_adoption=0.5,
        negative_initial_opinion=-0.25,
        positive_initial_opinion=0.25,
        teams_num_list=[1, 5, 10, 15, 20, 25, 30],
        teams_size_list=[5, 10, 15, 20],
        save_name="figure_structure_min_accuracy.pdf",
        color_min=0.6,
        color_max=1.0,
        force_recompute_cache=False,
    )