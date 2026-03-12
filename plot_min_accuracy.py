"""
Plot minimum required AI accuracy to reach a target adoption level.

What this file does:
1. Uses the surrogate ODE model to predict steady-state adoption (Q + L).
2. For each organizational structure setting, finds the smallest
   `technology_success_rate` that achieves a chosen target adoption.
3. Visualizes that threshold as heatmaps over:
   - number of teams
   - team size
4. Saves a publication-ready figure to `figures/`.

How it is used in the pipeline:
- Supports decision analysis by translating adoption targets into minimum
  AI-performance requirements under different organizational structures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from predict_surrogate import predict_full_surrogate_output
from utils import BASE_DIR


FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


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
            )

    return Z


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
) -> None:
    if teams_num_list is None:
        teams_num_list = [1, 5, 10, 15, 20, 25, 30]

    if teams_size_list is None:
        teams_size_list = [5, 10, 15, 20]

    # Compute both grids
    Z_neg = compute_min_accuracy_grid(
        agents_average_initial_opinion=negative_initial_opinion,
        target_adoption=target_adoption,
        teams_num_list=teams_num_list,
        teams_size_list=teams_size_list,
        surrogate_path=surrogate_path,
    )

    Z_pos = compute_min_accuracy_grid(
        agents_average_initial_opinion=positive_initial_opinion,
        target_adoption=target_adoption,
        teams_num_list=teams_num_list,
        teams_size_list=teams_size_list,
        surrogate_path=surrogate_path,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.14,
        top=0.82,
        wspace=0.12,
    )

    cmap = "viridis"

    im0 = axes[0].imshow(
        Z_neg,
        origin="lower",
        aspect="auto",
        vmin=color_min,
        vmax=color_max,
        cmap=cmap,
    )
    im1 = axes[1].imshow(
        Z_pos,
        origin="lower",
        aspect="auto",
        vmin=color_min,
        vmax=color_max,
        cmap=cmap,
    )

    # Axis formatting
    subplot_titles = ["Negative initial sentiment", "Positive initial sentiment"]
    grids = [Z_neg, Z_pos]

    threshold = (color_min + color_max) / 2.0

    for ax, Z, subtitle in zip(axes, grids, subplot_titles):
        ax.set_title(subtitle)
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
                    fontsize=8,
                    color=text_color,
                )

    axes[0].set_ylabel("Number of teams")

    # Dedicated colorbar axis fully outside the right subplot
    cbar_ax = fig.add_axes([0.88, 0.14, 0.025, 0.62])
    cbar = fig.colorbar(
        im1,
        cax=cbar_ax,
        ticks=np.arange(0.6, 1.01, 0.1),
    )
    cbar.set_label("Minimum required AI accuracy")

    fig.suptitle(
        "Minimum AI accuracy for 50% adoption",
        fontsize=13,
        y=0.95,
    )

    if save_name is not None:
        save_path = FIGURES_DIR / save_name
        fig.savefig(
            save_path,
            bbox_inches="tight",
            dpi=400
        )
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    plot_min_accuracy_two_panel(
        target_adoption=0.5,
        negative_initial_opinion=-0.25,
        positive_initial_opinion=0.25,
        teams_num_list=[1, 5, 10, 15, 20, 25, 30],
        teams_size_list=[5, 10, 15, 20],
        #save_name="figure_structure_min_accuracy.png",
        save_name="figure_structure_min_accuracy.pdf",
        color_min=0.6,
        color_max=1.0,
    )