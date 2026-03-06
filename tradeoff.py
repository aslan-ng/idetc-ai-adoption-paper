"""
Render the decision surface: AI accuracy vs average initial opinion.

This module visualizes the organizational adoption response surface
derived from ODE-reduced SQLB dynamics. It plots final predicted
adoption as a function of two controllable levers:

    1. technology_success_rate         (AI accuracy)
    2. agents_average_initial_opinion  (baseline organizational sentiment)

The adoption metric used is:

    final_adoption = Q(T) + L(T)

where Q and L are the fractions of agents in adoption states at the
end of the simulation horizon T, predicted by the fitted ODE surrogate.

Workflow:
    - Load final_adoption.csv (generated from ODE-based adoption calculation).
    - Load settings.csv (via load_settings()).
    - Merge on model name.
    - Filter to organizations with exactly 100 agents.
    - For each teams_num in teams_num_list = [1, 10, 20, 30]:
        rows    → average initial opinion
        columns → AI accuracy
        values  → final adoption
    - Plot a filled contour / regime map of adoption across the design space.

Interpretation (decision framework):
    - The surface shows how improvements in AI accuracy can compensate
      for lower initial sentiment, and vice versa.
    - Regions of high adoption indicate feasible zones for successful rollout.
    - If iso-adoption contours are enabled, each contour represents a
      constant adoption target (a decision boundary).

Assumptions:
    - The experimental design forms a complete grid (no missing combinations)
      within each selected teams_num.
    - Adoption values lie in [0, 1]; color scaling is constrained accordingly.
    - Each grid point represents an independent simulation configuration.

Output:
    - Displays figures interactively if show=True.
    - Optionally saves publication-ready figures to disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils import BASE_DIR, load_settings

# Fixed fragmentation levels to plot, after filtering to exactly 100 agents.
teams_num_list = [1, 10, 20, 30]


def plot_tradeoff_iso_adoption(
    base_dir: Path,
    *,
    csv_name: str = "final_adoption.csv",
    agents_num_target: int = 100,
    show: bool = True,
    save_path: str | Path | None = None,
):
    """
    Plot final adoption tradeoff surfaces for models with a fixed number of agents.

    Parameters
    ----------
    base_dir : Path
        Base project directory.

    csv_name : str, default="final_adoption.csv"
        Name of the CSV file containing:
            name, final_adoption

    agents_num_target : int, default=100
        Keep only organizations with this total number of agents.

    show : bool, default=True
        Whether to display each figure.

    save_path : str | Path | None, default=None
        If provided:
            - directory path  -> save one file per teams_num into that directory
            - file path       -> use stem/suffix and append teams_num to filename

    Returns
    -------
    dict[int, pd.DataFrame]
        Dictionary mapping teams_num -> pivot grid used for plotting.
    """
    path = base_dir / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    # ---- load final adoption ----
    df = pd.read_csv(path)

    required = {"name", "final_adoption"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")

    df["final_adoption"] = pd.to_numeric(df["final_adoption"], errors="raise")

    # ---- load settings ----
    settings = pd.DataFrame(load_settings())

    required_settings = {
        "name",
        "agents_num",
        "teams_num",
        "agents_average_initial_opinion",
        "technology_success_rate",
    }
    missing_settings = required_settings - set(settings.columns)
    if missing_settings:
        raise ValueError(f"Missing columns in settings.csv: {sorted(missing_settings)}")

    settings["agents_num"] = pd.to_numeric(settings["agents_num"], errors="raise").astype(int)
    settings["teams_num"] = pd.to_numeric(settings["teams_num"], errors="raise").astype(int)
    settings["agents_average_initial_opinion"] = pd.to_numeric(
        settings["agents_average_initial_opinion"], errors="raise"
    )
    settings["technology_success_rate"] = pd.to_numeric(
        settings["technology_success_rate"], errors="raise"
    )

    # ---- merge final adoption with settings on model name ----
    df = df.merge(
        settings[
            [
                "name",
                "agents_num",
                "teams_num",
                "agents_average_initial_opinion",
                "technology_success_rate",
            ]
        ],
        on="name",
        how="inner",
    )

    if df.empty:
        raise ValueError(
            "After merging final_adoption.csv with settings.csv, no rows remained. "
            "Check that model names match."
        )

    # ---- filter to organizations with exactly agents_num_target agents ----
    df = df[df["agents_num"] == agents_num_target].copy()

    if df.empty:
        raise ValueError(f"No rows found with agents_num == {agents_num_target}.")

    # ---- saving behavior ----
    out_dir: Path | None = None
    out_base: Path | None = None
    if save_path is not None:
        sp = Path(save_path)
        if sp.suffix:
            out_base = sp
            out_base.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = sp
            out_dir.mkdir(parents=True, exist_ok=True)

    grids: dict[int, pd.DataFrame] = {}

    # ---- plot one figure per teams_num ----
    for tn in teams_num_list:
        df_tn = df[df["teams_num"] == tn].copy()

        if df_tn.empty:
            print(f"Skipping teams_num={tn}: no matching rows found for agents_num={agents_num_target}.")
            continue

        # Optional diagnostic: warn if duplicate design points exist.
        dupes = df_tn.duplicated(
            subset=["agents_average_initial_opinion", "technology_success_rate"],
            keep=False,
        )
        if dupes.any():
            print(
                f"Warning: teams_num={tn} has duplicate "
                "(agents_average_initial_opinion, technology_success_rate) points; "
                "pivot_table will average them."
            )

        # rows -> opinion, columns -> success rate, values -> final adoption
        grid = df_tn.pivot_table(
            index="agents_average_initial_opinion",
            columns="technology_success_rate",
            values="final_adoption",
            aggfunc="mean",
        ).sort_index(axis=0).sort_index(axis=1)

        if grid.empty:
            print(f"Skipping teams_num={tn}: pivot grid is empty.")
            continue

        if grid.isna().any().any():
            nan_locs = np.argwhere(grid.isna().to_numpy())
            raise ValueError(
                f"teams_num={tn}: Grid has missing cells (NaNs). "
                f"Example missing at {nan_locs[:5].tolist()}. "
                "Ensure all combinations were simulated."
            )

        x_vals = grid.columns.to_numpy(dtype=float)
        y_vals = grid.index.to_numpy(dtype=float)
        Z = grid.to_numpy(dtype=float)
        X, Y = np.meshgrid(x_vals, y_vals)

        plt.figure()

        # Regime boundaries: [0, 0.2) low, [0.2, 0.8) partial, [0.8, 1.0] high
        color_levels = np.array([0.0, 0.2, 0.8, 1.0])
        cf = plt.contourf(X, Y, Z, levels=color_levels, vmin=0.0, vmax=1.0)

        plt.xlabel("AI Accuracy", labelpad=-10)
        plt.ylabel("Average Initial Opinion", labelpad=-30)
        plt.title(f"Number of teams: {tn}")

        ax = plt.gca()

        # Major ticks / semantic labels
        ax.set_xticks([0.0, 1.0])
        ax.set_xticklabels(["Low", "High"])

        ax.set_yticks([-1.0, 1.0])
        ax.set_yticklabels(["Negative", "Positive"])

        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)

        # Minor spacing aligned to your design grid
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.2))

        # Neutral-opinion divider
        ax.axhline(
            y=0.0,
            color="gray",
            linewidth=1.2,
            alpha=0.5,
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

        cbar = plt.colorbar(cf)

        # Midpoints for regime labels
        region_midpoints = [0.1, 0.5, 0.9]
        cbar.set_ticks(region_midpoints)
        cbar.set_ticklabels(["Low", "Partial", "High"])
        cbar.set_label("Adoption Regime")

        # Decide output filename for this teams_num
        out = None
        if out_dir is not None:
            out = out_dir / f"final_adoption_agents_{agents_num_target}_teams_{tn}.png"
        elif out_base is not None:
            out = out_base.with_name(
                f"{out_base.stem}_agents_{agents_num_target}_teams_{tn}{out_base.suffix}"
            )

        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, bbox_inches="tight", dpi=200)

        if show:
            plt.show()
        else:
            plt.close()

        grids[tn] = grid

    if not grids:
        raise ValueError(
            f"No figures were produced. After filtering to agents_num == {agents_num_target}, "
            f"no rows were available for teams_num_list = {teams_num_list}."
        )

    return grids


if __name__ == "__main__":
    plot_tradeoff_iso_adoption(
        BASE_DIR,
        csv_name="final_adoption.csv",
        agents_num_target=100,
        show=False,
        save_path=Path(BASE_DIR) / "figures",
    )