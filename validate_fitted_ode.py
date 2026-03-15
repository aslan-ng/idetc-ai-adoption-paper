"""
Validate the ABM -> fitted ODE reduction across all model configurations.

This module compares the ABM SQLB state-fraction trajectories against the
corresponding fitted ODE trajectories for every model in the experiment set.

Validation outputs:
    1. Per-model trajectory metrics
    2. Summary statistics across all models
    3. Error histograms
    4. Parity plots for final state fractions and final adoption
    5. Optional example overlays for a few selected models

Definitions:
    - ABM reference trajectory:
          x_abm(t) = [S(t), Q(t), L(t), B(t)]
    - Fitted ODE trajectory:
          x_ode(t) obtained by forward simulation of the fitted CTMC generator Q
    - Final adoption:
          adoption(T) = Q(T) + L(T)

Expected files:
    - states/{model_name}.csv
    - odes/{model_name}.npz
    - settings.csv

Outputs:
    - figures/validation_abm_to_ode/validation_abm_to_ode_*.pdf
    - (Optional) validation_abm_to_ode.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import BASE_DIR, get_all_model_names


# =========================
# Figure style parameters
# Edit here to tune visuals
# =========================
PLOT_STYLE = {
    # general
    "grid_alpha": 0.3,

    # histogram figures
    "hist_fig_width": 6.0,
    "hist_fig_height": 4.0,
    "hist_title_fontsize": 14,
    "hist_axis_label_fontsize": 12,
    "hist_tick_fontsize": 11,

    # parity figures
    "parity_fig_width": 5.5,
    "parity_fig_height": 5.5,
    "parity_title_fontsize": 20,
    "parity_axis_label_fontsize": 18,
    "parity_tick_fontsize": 16,

    # parity scatter appearance
    "parity_marker_size": 10,
    "parity_marker_alpha": 0.20,
    "parity_identity_linewidth": 1.5,

    # example overlay figures
    "overlay_fig_width": 7.0,
    "overlay_fig_height": 4.5,
    "overlay_title_fontsize": 14,
    "overlay_axis_label_fontsize": 12,
    "overlay_tick_fontsize": 11,
    "overlay_legend_fontsize": 10,
    "overlay_linewidth_abm": 2.0,
    "overlay_linewidth_ode": 2.0,
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


def _load_Q_npz(odes_dir: Path, model_name: str) -> tuple[np.ndarray, float, list[str]]:
    npz_path = odes_dir / f"{model_name}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"ODE fit file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    Q = data["Q"].astype(float)
    dt = float(data["dt"][0]) if "dt" in data else 1.0
    state_order = (
        data["state_order"].tolist()
        if "state_order" in data
        else STATE_ORDER.copy()
    )
    return Q, dt, state_order


def _reorder_Q_to_sqlb(Q: np.ndarray, state_order: list[str]) -> np.ndarray:
    if list(state_order) == STATE_ORDER:
        return Q

    idx = {s: i for i, s in enumerate(state_order)}
    missing = [s for s in STATE_ORDER if s not in idx]
    if missing:
        raise ValueError(
            f"ODE file state_order missing states required for SQLB ordering: {missing}"
        )

    perm = [idx[s] for s in STATE_ORDER]
    return Q[np.ix_(perm, perm)]


def _simulate_ode_fractions(
    Q: np.ndarray,
    x0: np.ndarray,
    T: int,
    *,
    dt: float = 1.0,
) -> np.ndarray:
    """
    Forward simulate fractions with explicit Euler:
        x_{t+1} = x_t + dt * x_t Q
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
    ode: np.ndarray,
    *,
    min_adoption_for_composition: float = 1e-3,
) -> dict[str, float]:
    diff = ode - abm
    abs_diff = np.abs(diff)
    sq_diff = diff**2

    out: dict[str, float] = {}

    # Per-state metrics over full trajectory
    for i, state in enumerate(STATE_ORDER):
        out[f"rmse_{state}"] = float(np.sqrt(np.mean(sq_diff[:, i])))
        out[f"mae_{state}"] = float(np.mean(abs_diff[:, i]))
        out[f"final_abs_error_{state}"] = float(abs_diff[-1, i])
        out[f"r2_{state}"] = _r2_score(abm[:, i], ode[:, i])

    # Aggregate pooled R^2 across all states and time steps
    out["r2_pooled_states"] = _pooled_state_r2(abm, ode)

    # Aggregate trajectory metrics across all states and time steps
    out["trajectory_rmse"] = float(np.sqrt(np.mean(sq_diff)))
    out["trajectory_mae"] = float(np.mean(abs_diff))
    out["max_abs_error"] = float(np.max(abs_diff))

    # Adoption metrics
    abm_adoption = abm[:, 1] + abm[:, 2]
    ode_adoption = ode[:, 1] + ode[:, 2]

    out["rmse_adoption"] = float(np.sqrt(np.mean((ode_adoption - abm_adoption) ** 2)))
    out["mae_adoption"] = float(np.mean(np.abs(ode_adoption - abm_adoption)))
    out["r2_adoption"] = _r2_score(abm_adoption, ode_adoption)
    out["final_abm_adoption"] = float(abm_adoption[-1])
    out["final_ode_adoption"] = float(ode_adoption[-1])
    out["final_abs_error_adoption"] = float(abs(ode_adoption[-1] - abm_adoption[-1]))

    # Composition metrics at final time
    abm_final_adoption = float(abm_adoption[-1])
    ode_final_adoption = float(ode_adoption[-1])

    if abm_final_adoption >= min_adoption_for_composition:
        abm_quiet_share = float(abm[-1, 1] / abm_final_adoption)
        abm_loud_share = float(abm[-1, 2] / abm_final_adoption)
    else:
        abm_quiet_share = np.nan
        abm_loud_share = np.nan

    if ode_final_adoption >= min_adoption_for_composition:
        ode_quiet_share = float(ode[-1, 1] / ode_final_adoption)
        ode_loud_share = float(ode[-1, 2] / ode_final_adoption)
    else:
        ode_quiet_share = np.nan
        ode_loud_share = np.nan

    out["final_abm_quiet_share"] = abm_quiet_share
    out["final_abm_loud_share"] = abm_loud_share
    out["final_ode_quiet_share"] = ode_quiet_share
    out["final_ode_loud_share"] = ode_loud_share
    out["final_abs_error_quiet_share"] = (
        float(abs(ode_quiet_share - abm_quiet_share))
        if np.isfinite(abm_quiet_share) and np.isfinite(ode_quiet_share)
        else np.nan
    )
    out["final_abs_error_loud_share"] = (
        float(abs(ode_loud_share - abm_loud_share))
        if np.isfinite(abm_loud_share) and np.isfinite(ode_loud_share)
        else np.nan
    )

    return out


def evaluate_single_model(
    base_dir: Path,
    model_name: str,
    *,
    min_adoption_for_composition: float = 1e-3,
) -> dict[str, float]:
    states_dir = base_dir / "states"
    odes_dir = base_dir / "odes"

    df = _load_states_df(states_dir, model_name)
    Q, dt, state_order = _load_Q_npz(odes_dir, model_name)
    Q = _reorder_Q_to_sqlb(Q, state_order)

    abm = _extract_abm_array(df)
    T = len(df)
    x0 = abm[0].copy()
    ode = _simulate_ode_fractions(Q, x0, T, dt=dt)

    row: dict[str, float] = {"model_name": model_name, "T": int(T), "dt": float(dt)}
    row.update(
        _compute_state_metrics(
            abm,
            ode,
            min_adoption_for_composition=min_adoption_for_composition,
        )
    )
    return row


def evaluate_all_models(
    base_dir: Path,
    *,
    min_adoption_for_composition: float = 1e-3,
    verbose: bool = True,
) -> pd.DataFrame:
    model_names = get_all_model_names(base_dir)
    rows: list[dict[str, float]] = []

    for i, model_name in enumerate(model_names, start=1):
        try:
            row = evaluate_single_model(
                base_dir,
                model_name,
                min_adoption_for_composition=min_adoption_for_composition,
            )
            rows.append(row)
            if verbose:
                print(f"[{i}/{len(model_names)}] validated {model_name}")
        except Exception as e:
            print(f"[{i}/{len(model_names)}] skipped {model_name}: {e}")

    if not rows:
        raise RuntimeError("No models were successfully validated.")

    return pd.DataFrame(rows)


def print_summary_table(
    results_df: pd.DataFrame,
    *,
    save_path: Path | None = None,
) -> pd.DataFrame:
    summary_cols = [
        "r2_pooled_states",
        "r2_S",
        "r2_Q",
        "r2_L",
        "r2_B",
        "r2_adoption",
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

    summary_df = results_df[available].describe().round(6)

    print("\nValidation summary statistics:")
    print(summary_df)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(save_path)
        print(f"Saved summary statistics to: {save_path}")

    return summary_df


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
        ("final_abs_error_adoption", "Final adoption absolute error"),
        ("final_abs_error_quiet_share", "Final quiet-share absolute error"),
        ("final_abs_error_loud_share", "Final loud-share absolute error"),
    ]

    for col, title in metrics:
        vals = results_df[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue

        fig = plt.figure(
            figsize=(style["hist_fig_width"], style["hist_fig_height"])
        )
        ax = plt.gca()

        ax.hist(vals, bins=30)
        ax.set_xlabel(col, fontsize=style["hist_axis_label_fontsize"])
        ax.set_ylabel("Count", fontsize=style["hist_axis_label_fontsize"])
        ax.set_title(title, fontsize=style["hist_title_fontsize"])
        ax.grid(True, alpha=style["grid_alpha"])
        ax.tick_params(axis="both", labelsize=style["hist_tick_fontsize"])

        save_path = save_dir / f"validation_abm_to_ode_hist_{col}.pdf"
        _save_fig(fig, save_path)

        if show:
            plt.show()

def _parity_plot(
    x: np.ndarray,
    y: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    save_path: Path,
    style: dict = PLOT_STYLE,
    show: bool = False,
) -> None:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        return

    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))

    # Pearson correlation
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else np.nan
    print(f"{title}: Pearson correlation = {corr:.6f}")

    fig = plt.figure(
        figsize=(style["parity_fig_width"], style["parity_fig_height"])
    )
    ax = plt.gca()

    ax.scatter(
        x,
        y,
        s=style["parity_marker_size"],
        alpha=style["parity_marker_alpha"],
    )
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle="--",
        linewidth=style["parity_identity_linewidth"],
    )

    ax.set_xlabel(xlabel, fontsize=style["parity_axis_label_fontsize"])
    ax.set_ylabel(ylabel, fontsize=style["parity_axis_label_fontsize"])
    ax.set_title(title, fontsize=style["parity_title_fontsize"])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, alpha=style["grid_alpha"])
    ax.tick_params(axis="both", labelsize=style["parity_tick_fontsize"])

    # Show correlation on figure
    '''
    ax.text(
        0.05,
        0.95,
        f"r = {corr:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    '''

    _save_fig(fig, save_path)

    if show:
        plt.show()

def plot_parity_figures(
    results_df: pd.DataFrame,
    *,
    save_dir: Path,
    style: dict = PLOT_STYLE,
    show: bool = False,
) -> None:
    parity_specs = [
        (
            "final_abm_adoption",
            "final_ode_adoption",
            "ABM final adoption",
            "Fitted CTMC/ODE final adoption",
            "Parity plot: final adoption",
            save_dir / "validation_abm_to_ode_parity_adoption.pdf",
        ),
        # (
        #     "final_abm_quiet_share",
        #     "final_ode_quiet_share",
        #     "ABM final quiet share",
        #     "ODE final quiet share",
        #     "Parity plot: final quiet share",
        #     save_dir / "validation_abm_to_ode_parity_quiet_share.pdf",
        # ),
        # (
        #     "final_abm_loud_share",
        #     "final_ode_loud_share",
        #     "ABM final loud share",
        #     "ODE final loud share",
        #     "Parity plot: final loud share",
        #     save_dir / "validation_abm_to_ode_parity_loud_share.pdf",
        # ),
    ]

    for xcol, ycol, xlabel, ylabel, title, save_path in parity_specs:
        _parity_plot(
            results_df[xcol].to_numpy(dtype=float),
            results_df[ycol].to_numpy(dtype=float),
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            save_path=save_path,
            style=style,
            show=show,
        )

def plot_example_overlays(
    base_dir: Path,
    model_names: list[str],
    *,
    save_dir: Path,
    style: dict = PLOT_STYLE,
    show: bool = False,
) -> None:
    for model_name in model_names:
        df = _load_states_df(base_dir / "states", model_name)
        Q, dt, state_order = _load_Q_npz(base_dir / "odes", model_name)
        Q = _reorder_Q_to_sqlb(Q, state_order)

        t = df["t"].to_numpy(dtype=float)
        abm = _extract_abm_array(df)
        ode = _simulate_ode_fractions(Q, abm[0].copy(), len(df), dt=dt)

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
            t, ode[:, 0], "--",
            linewidth=style["overlay_linewidth_ode"],
            color=colors["S"], label="S (ODE)"
        )
        ax.plot(
            t, ode[:, 1], "--",
            linewidth=style["overlay_linewidth_ode"],
            color=colors["Q"], label="Q (ODE)"
        )
        ax.plot(
            t, ode[:, 2], "--",
            linewidth=style["overlay_linewidth_ode"],
            color=colors["L"], label="L (ODE)"
        )
        ax.plot(
            t, ode[:, 3], "--",
            linewidth=style["overlay_linewidth_ode"],
            color=colors["B"], label="B (ODE)"
        )

        ax.set_xlabel("Time step (t)", fontsize=style["overlay_axis_label_fontsize"])
        ax.set_ylabel("Agent ratio", fontsize=style["overlay_axis_label_fontsize"])
        ax.set_title(
            f"ABM vs fitted ODE — {model_name}",
            fontsize=style["overlay_title_fontsize"],
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, alpha=style["grid_alpha"])
        ax.tick_params(axis="both", labelsize=style["overlay_tick_fontsize"])
        ax.legend(ncol=2, fontsize=style["overlay_legend_fontsize"])

        save_path = save_dir / f"validation_abm_to_ode_overlay_{model_name}.pdf"
        _save_fig(fig, save_path)

        if show:
            plt.show()


def pick_example_models(
    results_df: pd.DataFrame,
    *,
    n_best: int = 1,
    n_median: int = 1,
    n_worst: int = 1,
) -> list[str]:
    """
    Pick example models based on trajectory RMSE:
        - best fit(s)
        - median fit(s)
        - worst fit(s)
    """
    df = results_df.sort_values("trajectory_rmse").reset_index(drop=True)
    chosen: list[str] = []

    if len(df) == 0:
        return chosen

    if n_best > 0:
        chosen.extend(df.head(n_best)["model_name"].astype(str).tolist())

    if n_median > 0:
        mid = len(df) // 2
        half = n_median // 2
        start = max(0, mid - half)
        end = min(len(df), start + n_median)
        chosen.extend(df.iloc[start:end]["model_name"].astype(str).tolist())

    if n_worst > 0:
        chosen.extend(df.tail(n_worst)["model_name"].astype(str).tolist())

    # preserve order, remove duplicates
    seen = set()
    unique = []
    for x in chosen:
        if x not in seen:
            unique.append(x)
            seen.add(x)
    return unique


def run_validation(
    *,
    base_dir: Path = BASE_DIR,
    results_csv_name: str | None = None,
    min_adoption_for_composition: float = 1e-3,
    make_histograms: bool = True,
    make_parity_plots: bool = True,
    make_example_overlays: bool = True,
    n_best_examples: int = 1,
    n_median_examples: int = 1,
    n_worst_examples: int = 1,
    show: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    figures_dir = base_dir / "figures" / "validation_abm_to_fitted_ode"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results_df = evaluate_all_models(
        base_dir,
        min_adoption_for_composition=min_adoption_for_composition,
        verbose=verbose,
    )

    if results_csv_name is not None:
        results_csv_path = figures_dir / results_csv_name
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nSaved per-model validation results to: {results_csv_path}")

    print_summary_table(
        results_df,
        save_path=figures_dir / "validation_abm_to_fitted_ode_summary.csv",
    )

    if make_histograms:
        plot_error_histograms(results_df, save_dir=figures_dir, style=PLOT_STYLE, show=show)
        print("Saved histogram figures.")

    if make_parity_plots:
        plot_parity_figures(results_df, save_dir=figures_dir, style=PLOT_STYLE, show=show)
        print("Saved parity figures.")

    if make_example_overlays:
        example_models = pick_example_models(
            results_df,
            n_best=n_best_examples,
            n_median=n_median_examples,
            n_worst=n_worst_examples,
        )
        if example_models:
            plot_example_overlays(
                base_dir,
                example_models,
                save_dir=figures_dir,
                style=PLOT_STYLE,
                show=show,
            )
            print(f"Saved overlay figures for examples: {example_models}")

    return results_df


if __name__ == "__main__":
    run_validation(
        base_dir=BASE_DIR,
        #results_csv_name="validation_abm_to_ode.csv",
        min_adoption_for_composition=0.05,
        make_histograms=True,
        make_parity_plots=True,
        make_example_overlays=False,
        n_best_examples=1,
        n_median_examples=1,
        n_worst_examples=1,
        show=False,
        verbose=True,
    )