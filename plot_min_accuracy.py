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


def plot_structure_min_accuracy_heatmap(
    *,
    agents_average_initial_opinion: float = 0.0,
    target_adoption: float = 0.7,
    teams_num_list: list[int] | None = None,
    teams_size_list: list[int] | None = None,
    surrogate_path: Path | None = None,
    save_name: str | None = None,
    color_min: float = 0.0,
    color_max: float = 1.0,
) -> None:
    if teams_num_list is None:
        teams_num_list = [1, 5, 10, 15, 20, 25, 30]

    if teams_size_list is None:
        teams_size_list = [5, 10, 15, 20]

    Z = np.zeros((len(teams_num_list), len(teams_size_list)), dtype=float)

    for i, teams_num in enumerate(teams_num_list):
        for j, teams_size in enumerate(teams_size_list):
            acc_min = find_min_accuracy_for_target_adoption(
                teams_num=float(teams_num),
                teams_size=float(teams_size),
                agents_average_initial_opinion=float(agents_average_initial_opinion),
                target_adoption=float(target_adoption),
                surrogate_path=surrogate_path,
            )
            Z[i, j] = acc_min

    plt.figure(figsize=(7, 5))

    im = plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        vmin=color_min,
        vmax=color_max,
    )

    plt.colorbar(im, label="Minimum required AI success rate")
    plt.xticks(range(len(teams_size_list)), teams_size_list)
    plt.yticks(range(len(teams_num_list)), teams_num_list)

    plt.xlabel("Team size")
    plt.ylabel("Number of teams")
    plt.title(
        f"Minimum AI accuracy required for adoption\n"
        f"(target adoption={target_adoption:.2f}, initial opinion={agents_average_initial_opinion:.2f})"
    )

    for i in range(len(teams_num_list)):
        for j in range(len(teams_size_list)):
            val = Z[i, j]
            label = "NA" if np.isnan(val) else f"{val:.2f}"
            plt.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )

    plt.tight_layout()

    if save_name is not None:
        save_path = FIGURES_DIR / save_name
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    plot_structure_min_accuracy_heatmap(
        agents_average_initial_opinion=0.25,
        target_adoption=0.5,
        teams_num_list=[1, 5, 10, 15, 20, 25, 30],
        teams_size_list=[5, 10, 15, 20],
        save_name="figure_structure_min_accuracy_pos.png",
        color_min=0.6,
        color_max=1.0,
    )

    plot_structure_min_accuracy_heatmap(
        agents_average_initial_opinion=-0.25,
        target_adoption=0.5,
        teams_num_list=[1, 5, 10, 15, 20, 25, 30],
        teams_size_list=[5, 10, 15, 20],
        save_name="figure_structure_min_accuracy_neg.png",
        color_min=0.6,
        color_max=1.0,
    )