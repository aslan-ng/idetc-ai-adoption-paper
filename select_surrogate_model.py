from __future__ import annotations

from itertools import combinations_with_replacement
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import BASE_DIR


STATE_ORDER = ["S", "Q", "L", "B"]
BASE_INPUT_COLUMNS = [
    "teams_num",
    "teams_size",
    "agents_average_initial_opinion",
    "technology_success_rate",
]


def offdiag_transition_names(state_order: List[str] = STATE_ORDER) -> List[str]:
    names = []
    for i, si in enumerate(state_order):
        for j, sj in enumerate(state_order):
            if i != j:
                names.append(f"{si}->{sj}")
    return names


def load_one_ode_result(npz_path: Path) -> Dict[str, float]:
    data = np.load(npz_path, allow_pickle=True)
    transitions = data["transitions"].tolist()
    rates = data["rates"].astype(float).tolist()
    return {str(k): float(v) for k, v in zip(transitions, rates)}


def load_training_dataframe(base_dir: Path) -> pd.DataFrame:
    settings_path = base_dir / "settings.csv"
    odes_dir = base_dir / "odes"

    if not settings_path.exists():
        raise FileNotFoundError(f"Missing settings.csv: {settings_path}")
    if not odes_dir.exists():
        raise FileNotFoundError(f"Missing odes directory: {odes_dir}")

    settings_df = pd.read_csv(settings_path)

    required_cols = [
        "name",
        "teams_num",
        "teams_size",
        "agents_average_initial_opinion",
        "technology_success_rate",
    ]
    missing = set(required_cols) - set(settings_df.columns)
    if missing:
        raise ValueError(f"Missing settings columns: {sorted(missing)}")

    target_names = offdiag_transition_names()
    rows = []

    for _, row in settings_df.iterrows():
        model_name = str(row["name"])
        npz_path = odes_dir / f"{model_name}.npz"
        if not npz_path.exists():
            continue

        target_dict = load_one_ode_result(npz_path)

        out = {
            "name": model_name,
            "teams_num": float(row["teams_num"]),
            "teams_size": float(row["teams_size"]),
            "agents_average_initial_opinion": float(row["agents_average_initial_opinion"]),
            "technology_success_rate": float(row["technology_success_rate"]),
        }
        for t in target_names:
            out[t] = float(target_dict.get(t, 0.0))

        rows.append(out)

    if not rows:
        raise ValueError("No training rows found.")

    return pd.DataFrame(rows)


def generate_polynomial_powers(n_vars: int, max_degree: int) -> List[Tuple[int, ...]]:
    """
    Returns exponent tuples for all monomials with total degree <= max_degree.
    Includes bias term (0,0,...,0).
    """
    powers: List[Tuple[int, ...]] = [(0,) * n_vars]

    for deg in range(1, max_degree + 1):
        for combo in combinations_with_replacement(range(n_vars), deg):
            exponents = [0] * n_vars
            for idx in combo:
                exponents[idx] += 1
            powers.append(tuple(exponents))

    return powers


def power_name(exponents: Tuple[int, ...], base_names: List[str]) -> str:
    if sum(exponents) == 0:
        return "bias"

    parts = []
    for name, exp in zip(base_names, exponents):
        if exp == 0:
            continue
        if exp == 1:
            parts.append(name)
        else:
            parts.append(f"{name}^{exp}")
    return "*".join(parts)


def build_design_matrix(
    df: pd.DataFrame,
    degree: int,
    base_input_columns: List[str] = BASE_INPUT_COLUMNS,
) -> Tuple[np.ndarray, List[str]]:
    X_base = df[base_input_columns].to_numpy(dtype=float)  # (n,4)
    n_samples, n_vars = X_base.shape

    powers = generate_polynomial_powers(n_vars=n_vars, max_degree=degree)
    feature_names = [power_name(p, base_input_columns) for p in powers]

    X = np.ones((n_samples, len(powers)), dtype=float)
    for j, p in enumerate(powers):
        col = np.ones(n_samples, dtype=float)
        for var_idx, exp in enumerate(p):
            if exp != 0:
                col *= X_base[:, var_idx] ** exp
        X[:, j] = col

    return X, feature_names


def fit_ols(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def make_kfold_indices(n: int, k: int, seed: int = 42) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return [arr for arr in np.array_split(idx, k)]


def cross_validate_degree(
    training_df: pd.DataFrame,
    degree: int,
    k_folds: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    target_names = offdiag_transition_names()
    folds = make_kfold_indices(len(training_df), k_folds, seed=seed)

    fold_rmse_all = []
    fold_r2_all = []

    for fold_id in range(k_folds):
        test_idx = folds[fold_id]
        train_idx = np.concatenate([folds[i] for i in range(k_folds) if i != fold_id])

        train_df = training_df.iloc[train_idx].reset_index(drop=True)
        test_df = training_df.iloc[test_idx].reset_index(drop=True)

        X_train, feature_names = build_design_matrix(train_df, degree=degree)
        X_test, _ = build_design_matrix(test_df, degree=degree)

        Y_train = train_df[target_names].to_numpy(dtype=float)
        Y_test = test_df[target_names].to_numpy(dtype=float)

        beta = fit_ols(X_train, Y_train)
        Y_pred = X_test @ beta

        fold_rmse_all.append(rmse(Y_test, Y_pred))
        fold_r2_all.append(r2_score(Y_test.ravel(), Y_pred.ravel()))

    X_full, _ = build_design_matrix(training_df, degree=degree)
    Y_full = training_df[target_names].to_numpy(dtype=float)
    beta_full = fit_ols(X_full, Y_full)
    Y_hat_full = X_full @ beta_full

    return {
        "degree": degree,
        "n_features": X_full.shape[1],
        "feature_count_formula": comb(4 + degree, degree),
        "samples": len(training_df),
        "samples_per_feature": float(len(training_df) / X_full.shape[1]),
        "cv_rmse_mean": float(np.mean(fold_rmse_all)),
        "cv_rmse_std": float(np.std(fold_rmse_all)),
        "cv_r2_mean": float(np.mean(fold_r2_all)),
        "train_rmse": rmse(Y_full, Y_hat_full),
        "train_r2": r2_score(Y_full.ravel(), Y_hat_full.ravel()),
        "overfit_gap": float(np.mean(fold_rmse_all) - rmse(Y_full, Y_hat_full)),
    }


def compare_polynomial_degrees(
    base_dir: Path = BASE_DIR,
    max_degree: int = 8,
    k_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    training_df = load_training_dataframe(base_dir)

    results = []
    for degree in range(1, max_degree + 1):
        res = cross_validate_degree(
            training_df=training_df,
            degree=degree,
            k_folds=k_folds,
            seed=seed,
        )
        results.append(res)

    result_df = pd.DataFrame(results).sort_values("degree").reset_index(drop=True)
    return result_df

def plot_model_selection_results(result_df, base_dir):
    """
    Plot training vs cross-validation RMSE with CV error bars.
    Saves a single figure for the paper.
    """

    fig_dir = base_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    deg = result_df["degree"]

    plt.figure(figsize=(6,4))

    # Training curve
    plt.plot(
        deg,
        result_df["train_rmse"],
        marker="o",
        label="Train RMSE",
    )

    # CV curve with error bars
    plt.errorbar(
        deg,
        result_df["cv_rmse_mean"],
        yerr=result_df["cv_rmse_std"],
        marker="o",
        capsize=4,
        label="CV RMSE",
    )

    plt.xlabel("Polynomial Degree")
    plt.ylabel("RMSE")
    plt.title("Training vs Validation Error")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        fig_dir / "surrogate_model_selection.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


if __name__ == "__main__":

    result_df = compare_polynomial_degrees(
        base_dir=BASE_DIR,
        max_degree=8,
        k_folds=5,
        seed=42,
    )

    print("\nPolynomial surrogate model comparison")
    print(result_df.to_string(index=False))

    best_idx = result_df["cv_rmse_mean"].idxmin()
    best = result_df.loc[best_idx]

    print("\nBest model by CV RMSE:")
    print(
        f"degree={int(best['degree'])}, "
        f"n_features={int(best['n_features'])}, "
        f"cv_rmse_mean={best['cv_rmse_mean']:.6f}, "
        f"cv_r2_mean={best['cv_r2_mean']:.4f}"
    )

    # ---- generate figures ----
    plot_model_selection_results(result_df, BASE_DIR)

    print("\nFigures saved to:", BASE_DIR / "figures")
