from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from predict_surrogate import predict_full_surrogate_output
from utils import BASE_DIR


FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def predict_adoption_composition(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    surrogate_path: Path | None = None,
    min_adoption_for_composition: float = 0.001,
) -> dict[str, float]:
    """
    Predict steady-state SQLB composition and adoption shares.

    Returns
    -------
    dict
        Keys:
            S, Q, L, B
            adoption         = Q + L
            quiet_share      = Q / (Q + L), or np.nan if adoption too small
            loud_share       = L / (Q + L), or np.nan if adoption too small
    """
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

    adoption = Q + L

    if adoption < min_adoption_for_composition:
        quiet_share = np.nan
        loud_share = np.nan
    else:
        quiet_share = Q / adoption
        loud_share = L / adoption

    return {
        "S": S,
        "Q": Q,
        "L": L,
        "B": B,
        "adoption": adoption,
        "quiet_share": quiet_share,
        "loud_share": loud_share,
    }


def plot_loud_share_vs_accuracy(
    *,
    agents_average_initial_opinion: float,
    teams_num: int,
    teams_size: int,
    surrogate_path: Path | None = None,
    accuracy_grid: np.ndarray | None = None,
    save_name: str | None = None,
    show_adoption_on_second_axis: bool = False,
    accuracy_min: float = 0.5,
    accuracy_max: float = 1.0,
    min_adoption_for_composition: float = 0.05,
) -> None:
    """
    Plot loud-share composition versus AI accuracy for a fixed organization.

    Composition shares are shown only when total adoption Q+L is at least
    min_adoption_for_composition. Otherwise they are set to NaN.
    """
    if accuracy_grid is None:
        accuracy_grid = np.linspace(accuracy_min, accuracy_max, 201)

    loud_share_list: list[float] = []
    quiet_share_list: list[float] = []
    adoption_list: list[float] = []
    Q_list: list[float] = []
    L_list: list[float] = []

    for acc in accuracy_grid:
        res = predict_adoption_composition(
            teams_num=float(teams_num),
            teams_size=float(teams_size),
            agents_average_initial_opinion=float(agents_average_initial_opinion),
            technology_success_rate=float(acc),
            surrogate_path=surrogate_path,
            min_adoption_for_composition=min_adoption_for_composition,
        )

        loud_share_list.append(res["loud_share"])
        quiet_share_list.append(res["quiet_share"])
        adoption_list.append(res["adoption"])
        Q_list.append(res["Q"])
        L_list.append(res["L"])

    accuracy_grid = np.asarray(accuracy_grid, dtype=float)
    loud_share_arr = np.asarray(loud_share_list, dtype=float)
    quiet_share_arr = np.asarray(quiet_share_list, dtype=float)
    adoption_arr = np.asarray(adoption_list, dtype=float)
    Q_arr = np.asarray(Q_list, dtype=float)
    L_arr = np.asarray(L_list, dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        accuracy_grid,
        loud_share_arr,
        linewidth=2.5,
        label="Loud share  L / (Q + L)",
    )
    ax.plot(
        accuracy_grid,
        quiet_share_arr,
        linewidth=2.0,
        linestyle="--",
        label="Quiet share  Q / (Q + L)",
    )

    ax.set_xlim(accuracy_min, accuracy_max)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("AI accuracy")
    ax.set_ylabel("Adoption composition share")
    ax.set_title(
        "Adoption composition vs AI accuracy\n"
        f"(initial opinion={agents_average_initial_opinion:.2f}, "
        f"teams={teams_num}, team size={teams_size}, "
        f"composition shown only if Q+L ≥ {min_adoption_for_composition:.2f})"
    )
    ax.grid(True, alpha=0.3)

    if show_adoption_on_second_axis:
        ax2 = ax.twinx()
        ax2.plot(
            accuracy_grid,
            adoption_arr,
            linewidth=2.0,
            linestyle=":",
            label="Total adoption  Q + L",
        )
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("Total adoption")

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    else:
        ax.legend(loc="best")

    plt.tight_layout()

    if save_name is not None:
        save_path = FIGURES_DIR / save_name
        plt.savefig(save_path, dpi=400, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")
    else:
        plt.show()

    idx_mid = int(np.argmin(np.abs(accuracy_grid - 0.5)))
    idx_high = int(np.argmin(np.abs(accuracy_grid - 0.8)))

    print("\nExample values:")
    print(
        f"accuracy={accuracy_grid[idx_mid]:.2f} | "
        f"Q={Q_arr[idx_mid]:.3f}, L={L_arr[idx_mid]:.3f}, "
        f"Q+L={adoption_arr[idx_mid]:.3f}, "
        f"L/(Q+L)={loud_share_arr[idx_mid]}"
    )
    print(
        f"accuracy={accuracy_grid[idx_high]:.2f} | "
        f"Q={Q_arr[idx_high]:.3f}, L={L_arr[idx_high]:.3f}, "
        f"Q+L={adoption_arr[idx_high]:.3f}, "
        f"L/(Q+L)={loud_share_arr[idx_high]}"
    )


if __name__ == "__main__":
    plot_loud_share_vs_accuracy(
        agents_average_initial_opinion=-0.25,
        teams_num=5,
        teams_size=5,
        save_name="figure_loud_share_vs_accuracy_1.png",
        show_adoption_on_second_axis=True,
    )

    plot_loud_share_vs_accuracy(
        agents_average_initial_opinion=0.25,
        teams_num=5,
        teams_size=5,
        save_name="figure_loud_share_vs_accuracy_2.png",
        show_adoption_on_second_axis=True,
    )

    plot_loud_share_vs_accuracy(
        agents_average_initial_opinion=-0.25,
        teams_num=7,
        teams_size=7,
        save_name="figure_loud_share_vs_accuracy_3.png",
        show_adoption_on_second_axis=True,
    )

    plot_loud_share_vs_accuracy(
        agents_average_initial_opinion=0.25,
        teams_num=7,
        teams_size=7,
        save_name="figure_loud_share_vs_accuracy_4.png",
        show_adoption_on_second_axis=True,
    )

    plot_loud_share_vs_accuracy(
        agents_average_initial_opinion=-0.25,
        teams_num=10,
        teams_size=10,
        save_name="figure_loud_share_vs_accuracy_5.png",
        show_adoption_on_second_axis=True,
    )

    plot_loud_share_vs_accuracy(
        agents_average_initial_opinion=0.25,
        teams_num=10,
        teams_size=10,
        save_name="figure_loud_share_vs_accuracy_6.png",
        show_adoption_on_second_axis=True,
    )

    plot_loud_share_vs_accuracy(
        agents_average_initial_opinion=-0.25,
        teams_num=20,
        teams_size=20,
        save_name="figure_loud_share_vs_accuracy_7.png",
        show_adoption_on_second_axis=True,
    )

    plot_loud_share_vs_accuracy(
        agents_average_initial_opinion=0.25,
        teams_num=20,
        teams_size=20,
        save_name="figure_loud_share_vs_accuracy_8.png",
        show_adoption_on_second_axis=True,
    )