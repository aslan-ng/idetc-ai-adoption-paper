"""
Plot adoption composition versus AI accuracy using the surrogate ODE model.

What this file does:
1. Predicts steady-state SQLB composition for each parameter setting.
2. Computes adoption composition shares:
   - loud share  = L / (Q + L)
   - quiet share = Q / (Q + L)
3. Builds a configurable grid across organization size and initial opinion.
4. Saves a publication-ready figure to `figures/`.

How it is used in the pipeline:
- Supports the interpretation layer of the paper by showing how AI accuracy
  changes both total adoption and the loud/quiet composition of adopters.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from predict_surrogate import predict_full_surrogate_output
from utils import BASE_DIR


FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Figure style settings
# Change fonts here only
# =========================
FIG_WIDTH = 15
FIG_HEIGHT = 7

FONT_TITLE = 16
FONT_SUBTITLE = 13
FONT_AXIS_LABEL = 13
FONT_TICK = 11
FONT_ROW_LABEL = 12
FONT_LEGEND = 13

LINEWIDTH_MAIN = 2.0
GRID_ALPHA = 0.3

COLOR_LOUD = "#E69F00"   # orange
COLOR_QUIET = "#0072B2"  # blue
COLOR_TOTAL = "0.85"     # neutral gray


def predict_adoption_composition(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    surrogate_path: Path | None = None,
) -> dict[str, float]:
    out = predict_full_surrogate_output(
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
        surrogate_path=surrogate_path,
    )

    ss = out["steady_state"]

    S = float(ss["S"])
    Q = float(ss["Q"])
    L = float(ss["L"])
    B = float(ss["B"])

    total = S + Q + L + B

    adoption = (Q + L) / total
    quiet_share = Q / total
    loud_share = L / total

    return {
        "S": S,
        "Q": Q,
        "L": L,
        "B": B,
        "adoption": adoption,
        "quiet_share": quiet_share,
        "loud_share": loud_share,
    }


def plot_adoption_composition_grid(
    *,
    org_configs: list[tuple[int, int]] | None = None,
    initial_opinions: list[float] | None = None,
    surrogate_path: Path | None = None,
    row_labels: list[str] | None = None,
    column_titles: list[str] | None = None,
    accuracy_min: float = 0.5,
    accuracy_max: float = 1.0,
    n_accuracy_points: int = 201,
    save_name: str = "figure_loud_share_grid.pdf",
) -> None:
    """
    Parameters
    ----------
    org_configs
        List of (teams_num, teams_size) tuples, one per column.
        Example: [(5, 5), (15, 15), (25, 25)]
    initial_opinions
        List of initial opinion values, one per row.
        Example: [-0.25, 0.25]
    """
    if org_configs is None:
        org_configs = [(5, 5), (15, 15), (25, 25)]

    if initial_opinions is None:
        initial_opinions = [-0.25, 0.25]

    n_rows = len(initial_opinions)
    n_cols = len(org_configs)

    accuracy_grid = np.linspace(accuracy_min, accuracy_max, n_accuracy_points)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    if column_titles is None:
        column_titles = [
            f"{teams_num} teams, size {teams_size}"
            for teams_num, teams_size in org_configs
        ]

    if row_labels is None:
        row_labels = [
            f"Initial opinion = {opinion:+.2f}"
            for opinion in initial_opinions
        ]

    for r, opinion in enumerate(initial_opinions):
        for c, (teams_num, teams_size) in enumerate(org_configs):
            ax = axes[r, c]

            loud_share_list = []
            quiet_share_list = []
            adoption_list = []

            for acc in accuracy_grid:
                res = predict_adoption_composition(
                    teams_num=float(teams_num),
                    teams_size=float(teams_size),
                    agents_average_initial_opinion=float(opinion),
                    technology_success_rate=float(acc),
                    surrogate_path=surrogate_path,
                )

                loud_share_list.append(res["loud_share"])
                quiet_share_list.append(res["quiet_share"])
                adoption_list.append(res["adoption"])

            loud_share = np.asarray(loud_share_list, dtype=float)
            quiet_share = np.asarray(quiet_share_list, dtype=float)
            adoption = np.asarray(adoption_list, dtype=float)

            ax.plot(
                accuracy_grid,
                loud_share,
                linewidth=LINEWIDTH_MAIN,
                linestyle="--",
                color=COLOR_LOUD,
                label="Loud adopters",
            )
            ax.plot(
                accuracy_grid,
                quiet_share,
                linewidth=LINEWIDTH_MAIN,
                linestyle="--",
                color=COLOR_QUIET,
                label="Quiet adopters",
            )
            ax.plot(
                accuracy_grid,
                adoption,
                linewidth=LINEWIDTH_MAIN,
                color=COLOR_TOTAL,
                label="Total adoption",
            )

            ax.set_xlim(accuracy_min, accuracy_max)
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.grid(True, alpha=GRID_ALPHA)
            ax.tick_params(axis="both", labelsize=FONT_TICK)

            if r == 0:
                ax.set_title(
                    column_titles[c],
                    fontsize=FONT_SUBTITLE,
                    pad=14,
                    fontweight="bold",
                )

            if r == n_rows - 1:
                ax.set_xlabel("AI accuracy", fontsize=FONT_AXIS_LABEL)

            if c == 0:
                ax.set_ylabel("Share", fontsize=FONT_AXIS_LABEL)

    # Row labels on the left side
    # Positions are spaced automatically based on number of rows
    # Apply layout first so axis positions are final
    plt.tight_layout(rect=[0.08, 0.08, 1, 0.94])

    # Row labels aligned to the true vertical centers of each row
    for r, label in enumerate(row_labels):
        pos = axes[r, 0].get_position()
        y = 0.47 * (pos.y0 + pos.y1)
        fig.text(
            0.03,
            y,
            label,
            va="center",
            ha="center",
            rotation=90,
            fontsize=FONT_ROW_LABEL,
            fontweight="bold",
        )

    fig.suptitle(
        "Adoption Composition vs. AI accuracy",
        fontsize=FONT_TITLE,
        y=0.95,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=FONT_LEGEND,
    )

    plt.tight_layout(rect=[0.05, 0.08, 1, 0.94])

    save_path = FIGURES_DIR / save_name
    plt.savefig(save_path, dpi=400, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")


if __name__ == "__main__":
    plot_adoption_composition_grid(
        org_configs=[(5, 5), (10, 10), (15, 15), (20, 20)],
        initial_opinions=[0.3, 0.0, -0.3],
        save_name="figure_loud_share_grid.pdf",
        row_labels=[
            "Positive\ninitial opinion",
            "Neutral\ninitial opinion",
            "Negative\ninitial opinion",
        ],
        column_titles=[
            "5 teams × 5 members",
            "10 teams × 10 members",
            "15 teams × 15 members",
            "20 teams × 20 members",
        ]
    )