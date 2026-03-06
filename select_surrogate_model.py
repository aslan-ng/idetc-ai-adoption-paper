from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import BASE_DIR


STATE_ORDER = ["S", "Q", "L", "B"]


# ============================================================
# Data loading
# ============================================================

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


# ============================================================
# Polynomial features
# ============================================================

BASE_INPUTS = [
    "teams_num",
    "teams_size",
    "agents_average_initial_opinion",
    "technology_success_rate",
]


def build_polynomial_feature_dict(row: pd.Series, degree: int) -> Dict[str, float]:
    """
    Polynomial features up to degree 4 for the 4 input variables.

    degree=1:
        bias + main effects

    degree=2:
        + squares + pairwise interactions

    degree=3:
        + cubes + square-linear interactions + triple interaction

    degree=4:
        + quartics + selected mixed quartic terms
    """
    tn = float(row["teams_num"])
    ts = float(row["teams_size"])
    op = float(row["agents_average_initial_opinion"])
    sr = float(row["technology_success_rate"])

    f = {
        "bias": 1.0,
        "teams_num": tn,
        "teams_size": ts,
        "initial_opinion": op,
        "success_rate": sr,
    }

    if degree >= 2:
        f.update({
            "teams_num_sq": tn**2,
            "teams_size_sq": ts**2,
            "initial_opinion_sq": op**2,
            "success_rate_sq": sr**2,

            "teams_num_x_teams_size": tn * ts,
            "teams_num_x_initial_opinion": tn * op,
            "teams_num_x_success_rate": tn * sr,
            "teams_size_x_initial_opinion": ts * op,
            "teams_size_x_success_rate": ts * sr,
            "initial_opinion_x_success_rate": op * sr,
        })

    if degree >= 3:
        f.update({
            "teams_num_cu": tn**3,
            "teams_size_cu": ts**3,
            "initial_opinion_cu": op**3,
            "success_rate_cu": sr**3,

            "teams_num_sq_x_teams_size": tn**2 * ts,
            "teams_num_sq_x_initial_opinion": tn**2 * op,
            "teams_num_sq_x_success_rate": tn**2 * sr,
            "teams_size_sq_x_teams_num": ts**2 * tn,
            "teams_size_sq_x_initial_opinion": ts**2 * op,
            "teams_size_sq_x_success_rate": ts**2 * sr,
            "initial_opinion_sq_x_teams_num": op**2 * tn,
            "initial_opinion_sq_x_teams_size": op**2 * ts,
            "initial_opinion_sq_x_success_rate": op**2 * sr,
            "success_rate_sq_x_teams_num": sr**2 * tn,
            "success_rate_sq_x_teams_size": sr**2 * ts,
            "success_rate_sq_x_initial_opinion": sr**2 * op,

            "teams_num_x_teams_size_x_initial_opinion": tn * ts * op,
            "teams_num_x_teams_size_x_success_rate": tn * ts * sr,
            "teams_num_x_initial_opinion_x_success_rate": tn * op * sr,
            "teams_size_x_initial_opinion_x_success_rate": ts * op * sr,
        })

    if degree >= 4:
        f.update({
            "teams_num_qt": tn**4,
            "teams_size_qt": ts**4,
            "initial_opinion_qt": op**4,
            "success_rate_qt": sr**4,

            "teams_num_sq_x_teams_size_sq": tn**2 * ts**2,
            "teams_num_sq_x_initial_opinion_sq": tn**2 * op**2,
            "teams_num_sq_x_success_rate_sq": tn**2 * sr**2,
            "teams_size_sq_x_initial_opinion_sq": ts**2 * op**2,
            "teams_size_sq_x_success_rate_sq": ts**2 * sr**2,
            "initial_opinion_sq_x_success_rate_sq": op**2 * sr**2,

            "teams_num_sq_x_teams_size_x_initial_opinion": tn**2 * ts * op,
            "teams_num_sq_x_teams_size_x_success_rate": tn**2 * ts * sr,
            "teams_num_x_teams_size_sq_x_initial_opinion": tn * ts**2 * op,
            "teams_num_x_teams_size_sq_x_success_rate": tn * ts**2 * sr,
            "teams_num_x_initial_opinion_sq_x_success_rate": tn * op**2 * sr,
            "teams_size_x_initial_opinion_sq_x_success_rate": ts * op**2 * sr,
            "teams_num_x_initial_opinion_x_success_rate_sq": tn * op * sr**2,
            "teams_size_x_initial_opinion_x_success_rate_sq": ts * op * sr**2,
        })

    return f


def build_design_matrix(df: pd.DataFrame, degree: int) -> Tuple[np.ndarray, List[str]]:
    feature_dicts = [build_polynomial_feature_dict(row, degree) for _, row in df.iterrows()]
    feature_names = list(feature_dicts[0].keys())
    X = np.array([[fd[name] for name in feature_names] for fd in feature_dicts], dtype=float)
    return X, feature_names


# ============================================================
# Regression + metrics
# ============================================================

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

    X_full, feature_names = build_design_matrix(training_df, degree=degree)
    Y_full = training_df[target_names].to_numpy(dtype=float)
    beta_full = fit_ols(X_full, Y_full)
    Y_hat_full = X_full @ beta_full

    return {
        "degree": degree,
        "n_features": X_full.shape[1],
        "cv_rmse_mean": float(np.mean(fold_rmse_all)),
        "cv_rmse_std": float(np.std(fold_rmse_all)),
        "cv_r2_mean": float(np.mean(fold_r2_all)),
        "train_rmse": rmse(Y_full, Y_hat_full),
        "train_r2": r2_score(Y_full.ravel(), Y_hat_full.ravel()),
    }


def compare_polynomial_degrees(
    base_dir: Path = BASE_DIR,
    degrees: List[int] = [1, 2, 3, 4],
    k_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    training_df = load_training_dataframe(base_dir)

    results = []
    for degree in degrees:
        res = cross_validate_degree(
            training_df=training_df,
            degree=degree,
            k_folds=k_folds,
            seed=seed,
        )
        results.append(res)

    result_df = pd.DataFrame(results).sort_values("cv_rmse_mean").reset_index(drop=True)
    return result_df


if __name__ == "__main__":
    result_df = compare_polynomial_degrees(
        base_dir=BASE_DIR,
        degrees=[1, 2, 3, 4],
        k_folds=5,
        seed=42,
    )

    print("\nPolynomial surrogate model comparison")
    print(result_df.to_string(index=False))

    best = result_df.iloc[0]
    print("\nBest model by CV RMSE:")
    print(
        f"degree={int(best['degree'])}, "
        f"n_features={int(best['n_features'])}, "
        f"cv_rmse_mean={best['cv_rmse_mean']:.6f}, "
        f"cv_r2_mean={best['cv_r2_mean']:.4f}"
    )