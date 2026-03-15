"""
Validate the ABM -> surrogate-ODE pipeline across all model configurations.

This module compares ABM SQLB state-fraction trajectories against
surrogate-predicted ODE trajectories for every model in the experiment set.

Validation outputs:
    1. Per-model trajectory metrics
    2. Summary statistics across all models
    3. Error histograms
    4. Optional example overlay for a selected model

Definitions:
    - ABM reference trajectory:
          x_abm(t) = [S(t), Q(t), L(t), B(t)]
    - Surrogate trajectory:
          x_sur(t) obtained by:
              (a) predicting CTMC generator Q from the surrogate
              (b) simulating dx/dt = xQ from the ABM initial condition
    - Final adoption:
          adoption(T) = Q(T) + L(T)

Expected files:
    - states/{model_name}.csv
    - settings.csv
    - saved surrogate model file (loaded by load_surrogate_model)

Outputs:
    - figures/validation_abm_to_surrogate_ode/validation_abm_to_surrogate_*.pdf
    - figures/validation_abm_to_surrogate_ode/validation_abm_to_surrogate.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predict_surrogate import load_surrogate_model, predict_generator
from utils import BASE_DIR, get_all_model_names


# =========================
# Figure style parameters
# Edit here to tune visuals
# =========================
PLOT_STYLE = {
    # shared
    "grid_alpha": 0.3,

    # histogram figures
    "hist_fig_width": 6.0,
    "hist_fig_height": 4.0,
    "hist_title_fontsize": 23,
    "hist_axis_label_fontsize": 19,
    "hist_tick_fontsize": 18,

    # overlay figure
    "overlay_fig_width": 7.0,
    "overlay_fig_height": 4.5,
    "overlay_title_fontsize": 14,
    "overlay_axis_label_fontsize": 12,
    "overlay_tick_fontsize": 11,
    "overlay_legend_fontsize": 10,
    "overlay_linewidth_abm": 2.0,
    "overlay_linewidth_surrogate": 2.0,
}

STATE_ORDER = ["S", "Q", "L", "B"]

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Standard coefficient of determination:
        R^2 = 1 - SSE / SST
    Returns NaN if SST == 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = float(np.sum((y_pred - y_true) ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if np.isclose(sst, 0.0):
        return np.nan

    return 1.0 - sse / sst


def _pooled_state_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Aggregate R^2 across all SQLB states using state-wise means in SST.

    y_true, y_pred: shape (T, 4)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = float(np.sum((y_pred - y_true) ** 2))

    state_means = np.mean(y_true, axis=0, keepdims=True)  # shape (1, 4)
    sst = float(np.sum((y_true - state_means) ** 2))

    if np.isclose(sst, 0.0):
        return np.nan

    return 1.0 - sse / sst

def _load_states_df(states_dir: Path, model_name: str) -> pd.DataFrame:
    csv_path = states_dir / f"{model_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"State CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"t", "ratio_S", "ratio_Q", "ratio_L", "ratio_B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")

    df = df.sort_values("t").reset_index(drop=True)
    return df


def _load_settings_row(settings_path: Path, model_name: str) -> pd.Series:
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.csv not found: {settings_path}")

    df = pd.read_csv(settings_path)
    if "name" not in df.columns:
        raise ValueError("settings.csv must contain a 'name' column.")

    match = df.loc[df["name"].astype(str) == str(model_name)]
    if match.empty:
        raise ValueError(f"Model name '{model_name}' not found in settings.csv")

    return match.iloc[0]


def _simulate_ode_fractions(
    Q: np.ndarray,
    x0: np.ndarray,
    T: int,
    *,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Forward simulate fractions with explicit Euler:
        x_{t+1} = x_t + dt * (x_t @ Q)
    """
    x = np.zeros((T, 4), dtype=float)
    x[0] = x0

    for t in range(T - 1):
        x[t + 1] = x[t] + dt * (x[t] @ Q)

        # Numerical safety
        x[t + 1] = np.clip(x[t + 1], 0.0, 1.0)
        s = x[t + 1].sum()
        if s > 0:
            x[t + 1] /= s

    return x


def _extract_abm_array(df: pd.DataFrame) -> np.ndarray:
    return df[["ratio_S", "ratio_Q", "ratio_L", "ratio_B"]].to_numpy(dtype=float)


def _compute_state_metrics(
    abm: np.ndarray,
    surrogate: np.ndarray,
    *,
    min_adoption_for_composition: float = 1e-3,
) -> dict[str, float]:
    diff = surrogate - abm
    abs_diff = np.abs(diff)
    sq_diff = diff**2

    out: dict[str, float] = {}

    # Per-state metrics over full trajectory
    for i, state in enumerate(STATE_ORDER):
        out[f"rmse_{state}"] = float(np.sqrt(np.mean(sq_diff[:, i])))
        out[f"mae_{state}"] = float(np.mean(abs_diff[:, i]))
        out[f"final_abs_error_{state}"] = float(abs_diff[-1, i])

    # Aggregate trajectory metrics
    out["trajectory_rmse"] = float(np.sqrt(np.mean(sq_diff)))
    out["trajectory_mae"] = float(np.mean(abs_diff))
    out["max_abs_error"] = float(np.max(abs_diff))

    # Adoption metrics
    abm_adoption = abm[:, 1] + abm[:, 2]
    surrogate_adoption = surrogate[:, 1] + surrogate[:, 2]

    out["rmse_adoption"] = float(
        np.sqrt(np.mean((surrogate_adoption - abm_adoption) ** 2))
    )
    out["mae_adoption"] = float(
        np.mean(np.abs(surrogate_adoption - abm_adoption))
    )
    out["final_abm_adoption"] = float(abm_adoption[-1])
    out["final_surrogate_adoption"] = float(surrogate_adoption[-1])
    out["final_abs_error_adoption"] = float(
        abs(surrogate_adoption[-1] - abm_adoption[-1])
    )

    # Composition metrics at final time
    abm_final_adoption = float(abm_adoption[-1])
    surrogate_final_adoption = float(surrogate_adoption[-1])

    if abm_final_adoption >= min_adoption_for_composition:
        abm_quiet_share = float(abm[-1, 1] / abm_final_adoption)
        abm_loud_share = float(abm[-1, 2] / abm_final_adoption)
    else:
        abm_quiet_share = np.nan
        abm_loud_share = np.nan

    if surrogate_final_adoption >= min_adoption_for_composition:
        surrogate_quiet_share = float(surrogate[-1, 1] / surrogate_final_adoption)
        surrogate_loud_share = float(surrogate[-1, 2] / surrogate_final_adoption)
    else:
        surrogate_quiet_share = np.nan
        surrogate_loud_share = np.nan

    out["final_abm_quiet_share"] = abm_quiet_share
    out["final_abm_loud_share"] = abm_loud_share
    out["final_surrogate_quiet_share"] = surrogate_quiet_share
    out["final_surrogate_loud_share"] = surrogate_loud_share

    out["final_abs_error_quiet_share"] = (
        float(abs(surrogate_quiet_share - abm_quiet_share))
        if np.isfinite(abm_quiet_share) and np.isfinite(surrogate_quiet_share)
        else np.nan
    )
    out["final_abs_error_loud_share"] = (
        float(abs(surrogate_loud_share - abm_loud_share))
        if np.isfinite(abm_loud_share) and np.isfinite(surrogate_loud_share)
        else np.nan
    )

    return out


def evaluate_single_model(
    base_dir: Path,
    model_name: str,
    *,
    surrogate_model,
    min_adoption_for_composition: float = 1e-3,
    simulation_dt: float = 1.0,
) -> dict[str, float]:
    states_dir = base_dir / "states"
    settings_path = base_dir / "settings.csv"

    df = _load_states_df(states_dir, model_name)
    settings_row = _load_settings_row(settings_path, model_name)

    Q_surrogate, rates_surrogate = predict_generator(
        surrogate_model,
        teams_num=float(settings_row["teams_num"]),
        teams_size=float(settings_row["teams_size"]),
        agents_average_initial_opinion=float(
            settings_row["agents_average_initial_opinion"]
        ),
        technology_success_rate=float(settings_row["technology_success_rate"]),
    )

    abm = _extract_abm_array(df)
    T = len(df)
    x0 = abm[0].copy()

    surrogate = _simulate_ode_fractions(
        Q_surrogate,
        x0,
        T,
        dt=simulation_dt,
    )

    row: dict[str, float] = {
        "model_name": model_name,
        "T": int(T),
        "dt": float(simulation_dt),
        "teams_num": float(settings_row["teams_num"]),
        "teams_size": float(settings_row["teams_size"]),
        "agents_average_initial_opinion": float(
            settings_row["agents_average_initial_opinion"]
        ),
        "technology_success_rate": float(settings_row["technology_success_rate"]),
    }

    row.update(
        _compute_state_metrics(
            abm,
            surrogate,
            min_adoption_for_composition=min_adoption_for_composition,
        )
    )

    return row


def evaluate_all_models(
    base_dir: Path,
    *,
    surrogate_path: str | Path | None = None,
    min_adoption_for_composition: float = 1e-3,
    simulation_dt: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    model_names = get_all_model_names(base_dir)
    surrogate_model = load_surrogate_model(
        Path(surrogate_path) if surrogate_path is not None else None
    )

    rows: list[dict[str, float]] = []

    for i, model_name in enumerate(model_names, start=1):
        try:
            row = evaluate_single_model(
                base_dir,
                model_name,
                surrogate_model=surrogate_model,
                min_adoption_for_composition=min_adoption_for_composition,
                simulation_dt=simulation_dt,
            )
            rows.append(row)
            if verbose:
                print(f"[{i}/{len(model_names)}] validated {model_name}")
        except Exception as e:
            print(f"[{i}/{len(model_names)}] skipped {model_name}: {e}")

    if not rows:
        raise RuntimeError("No models were successfully validated.")

    return pd.DataFrame(rows)


def print_summary_table(results_df: pd.DataFrame) -> None:
    summary_cols = [
        "trajectory_rmse",
        "trajectory_mae",
        "rmse_adoption",
        "mae_adoption",
        "final_abs_error_adoption",
        "final_abs_error_quiet_share",
        "final_abs_error_loud_share",
        "max_abs_error",
    ]
    available = [c for c in summary_cols if c in results_df.columns]

    print("\nValidation summary statistics:")
    print(results_df[available].describe().round(6))


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_error_histograms(
    results_df: pd.DataFrame,
    *,
    save_dir: Path,
    style: dict = PLOT_STYLE,
    show: bool = False,
) -> None:
    metrics = [
        ("trajectory_rmse", "Trajectory RMSE"),
        #("trajectory_rmse", ""),
        ("final_abs_error_adoption", "Final adoption absolute error"),
        ("final_abs_error_quiet_share", "Final quiet-share absolute error"),
        ("final_abs_error_loud_share", "Final loud-share absolute error"),
    ]

    for col, title in metrics:
        vals = results_df[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue

        weights = np.ones_like(vals, dtype=float) / len(vals)

        fig = plt.figure(
            figsize=(style["hist_fig_width"], style["hist_fig_height"])
        )
        ax = plt.gca()

        ax.hist(vals, bins=30, weights=weights)
        ax.set_xlabel(title, fontsize=style["hist_axis_label_fontsize"])
        ax.set_ylabel("Share of scenarios", fontsize=style["hist_axis_label_fontsize"])
        #ax.set_title(title, fontsize=style["hist_title_fontsize"])
        ax.set_title(
            "Surrogate-ABM Trajectory Error Distribution",
            fontsize=style["hist_title_fontsize"],
            pad=20
        )
        ax.grid(True, alpha=style["grid_alpha"])
        ax.tick_params(axis="both", labelsize=style["hist_tick_fontsize"])

        if col == "trajectory_rmse":
            ax.set_yticks([0.00, 0.05, 0.10, 0.15])
            ax.set_yticklabels(["0.0", "5%", "10%", "15%"])
            ax.set_ylim(0.0, 0.15)

        save_path = save_dir / f"validation_abm_to_surrogate_hist_{col}.pdf"
        _save_fig(fig, save_path)

        if show:
            plt.show()


def plot_selected_overlay(
    base_dir: Path,
    model_name: str,
    *,
    surrogate_path: str | Path | None = None,
    simulation_dt: float = 1.0,
    save_dir: Path,
    style: dict = PLOT_STYLE,
    show: bool = False,
) -> None:
    states_dir = base_dir / "states"
    settings_path = base_dir / "settings.csv"

    surrogate_model = load_surrogate_model(
        Path(surrogate_path) if surrogate_path is not None else None
    )

    df = _load_states_df(states_dir, model_name)
    settings_row = _load_settings_row(settings_path, model_name)

    Q_surrogate, rates_surrogate = predict_generator(
        surrogate_model,
        teams_num=float(settings_row["teams_num"]),
        teams_size=float(settings_row["teams_size"]),
        agents_average_initial_opinion=float(
            settings_row["agents_average_initial_opinion"]
        ),
        technology_success_rate=float(settings_row["technology_success_rate"]),
    )

    t = df["t"].to_numpy(dtype=float)
    abm = _extract_abm_array(df)
    surrogate = _simulate_ode_fractions(
        Q_surrogate,
        abm[0].copy(),
        len(df),
        dt=simulation_dt,
    )

    fig = plt.figure(
        figsize=(style["overlay_fig_width"], style["overlay_fig_height"])
    )
    ax = plt.gca()

    line_S = ax.plot(
        t, abm[:, 0],
        linewidth=style["overlay_linewidth_abm"],
        label="S (ABM)"
    )[0]
    line_Q = ax.plot(
        t, abm[:, 1],
        linewidth=style["overlay_linewidth_abm"],
        label="Q (ABM)"
    )[0]
    line_L = ax.plot(
        t, abm[:, 2],
        linewidth=style["overlay_linewidth_abm"],
        label="L (ABM)"
    )[0]
    line_B = ax.plot(
        t, abm[:, 3],
        linewidth=style["overlay_linewidth_abm"],
        label="B (ABM)"
    )[0]

    colors = {
        "S": line_S.get_color(),
        "Q": line_Q.get_color(),
        "L": line_L.get_color(),
        "B": line_B.get_color(),
    }

    ax.plot(
        t, surrogate[:, 0], "--",
        linewidth=style["overlay_linewidth_surrogate"],
        color=colors["S"], label="S (Surrogate)"
    )
    ax.plot(
        t, surrogate[:, 1], "--",
        linewidth=style["overlay_linewidth_surrogate"],
        color=colors["Q"], label="Q (Surrogate)"
    )
    ax.plot(
        t, surrogate[:, 2], "--",
        linewidth=style["overlay_linewidth_surrogate"],
        color=colors["L"], label="L (Surrogate)"
    )
    ax.plot(
        t, surrogate[:, 3], "--",
        linewidth=style["overlay_linewidth_surrogate"],
        color=colors["B"], label="B (Surrogate)"
    )

    ax.set_xlabel("Time step", fontsize=style["overlay_axis_label_fontsize"])
    ax.set_ylabel("State fraction", fontsize=style["overlay_axis_label_fontsize"])
    ax.set_title(
        f"ABM vs surrogate — {model_name}",
        fontsize=style["overlay_title_fontsize"],
    )
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["overlay_tick_fontsize"])
    ax.legend(ncol=2, fontsize=style["overlay_legend_fontsize"])

    save_path = save_dir / f"validation_abm_to_surrogate_overlay_{model_name}.pdf"
    _save_fig(fig, save_path)

    if show:
        plt.show()


def run_validation(
    *,
    base_dir: Path = BASE_DIR,
    surrogate_path: str | Path | None = None,
    results_csv_name: str = "validation_abm_to_surrogate.csv",
    min_adoption_for_composition: float = 1e-3,
    simulation_dt: float = 1.0,
    make_histograms: bool = True,
    make_overlay: bool = True,
    overlay_model_name: str = "2_7_9_1_3",
    show: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    output_dir = base_dir / "figures" / "validation_abm_to_surrogate_ode"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = evaluate_all_models(
        base_dir,
        surrogate_path=surrogate_path,
        min_adoption_for_composition=min_adoption_for_composition,
        simulation_dt=simulation_dt,
        verbose=verbose,
    )

    results_csv_path = output_dir / results_csv_name
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved per-model validation results to: {results_csv_path}")

    print_summary_table(results_df)

    if make_histograms:
        plot_error_histograms(
            results_df,
            save_dir=output_dir,
            style=PLOT_STYLE,
            show=show,
        )
        print("Saved histogram figures.")

    if make_overlay:
        plot_selected_overlay(
            base_dir,
            overlay_model_name,
            surrogate_path=surrogate_path,
            simulation_dt=simulation_dt,
            save_dir=output_dir,
            style=PLOT_STYLE,
            show=show,
        )
        print(f"Saved overlay figure for model: {overlay_model_name}")

    return results_df


if __name__ == "__main__":
    run_validation(
        base_dir=BASE_DIR,
        surrogate_path=None,  # or BASE_DIR / "models" / "your_surrogate.pkl"
        results_csv_name="validation_abm_to_surrogate.csv",
        min_adoption_for_composition=0.05,
        simulation_dt=1.0,
        make_histograms=True,
        make_overlay=True,
        overlay_model_name="2_7_9_1_3",
        show=False,
        verbose=True,
    )