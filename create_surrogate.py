from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import BASE_DIR


STATE_ORDER = ["S", "Q", "L", "B"]
POLY_DEGREE = 6

BASE_INPUT_COLUMNS = [
    "teams_num",
    "teams_size",
    "agents_average_initial_opinion",
    "technology_success_rate",
]


# ============================================================
# Polynomial features (degree 6) + scaling
# ============================================================

def generate_polynomial_powers(n_vars: int, degree: int):
    powers = [(0,) * n_vars]

    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            exps = [0] * n_vars
            for idx in combo:
                exps[idx] += 1
            powers.append(tuple(exps))

    return powers


def power_name(exponents, base_names):
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


def fit_input_scaler(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit standardization parameters on the base inputs.
    """
    X_base = df[BASE_INPUT_COLUMNS].to_numpy(dtype=float)
    mu = X_base.mean(axis=0)
    sigma = X_base.std(axis=0, ddof=0)
    sigma = np.where(sigma == 0.0, 1.0, sigma)
    return mu, sigma


def transform_inputs(
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Standardize base inputs using supplied mu and sigma.
    """
    X_base = df[BASE_INPUT_COLUMNS].to_numpy(dtype=float)
    return (X_base - mu) / sigma


def build_design_matrix(
    df: pd.DataFrame,
    mu: np.ndarray,
    sigma: np.ndarray,
):
    X_base = transform_inputs(df, mu, sigma)
    n_samples, n_vars = X_base.shape

    powers = generate_polynomial_powers(n_vars, POLY_DEGREE)
    feature_names = [power_name(p, BASE_INPUT_COLUMNS) for p in powers]

    X = np.ones((n_samples, len(powers)), dtype=float)

    for j, p in enumerate(powers):
        col = np.ones(n_samples, dtype=float)

        for var_idx, exp in enumerate(p):
            if exp != 0:
                col *= X_base[:, var_idx] ** exp

        X[:, j] = col

    return X, feature_names


# ============================================================
# Target loading
# ============================================================

def offdiag_transition_names():
    names = []
    for i, si in enumerate(STATE_ORDER):
        for j, sj in enumerate(STATE_ORDER):
            if i != j:
                names.append(f"{si}->{sj}")
    return names


def load_one_ode_result(npz_path: Path):

    data = np.load(npz_path, allow_pickle=True)

    transitions = data["transitions"].tolist()
    rates = data["rates"].astype(float).tolist()

    return {str(k): float(v) for k, v in zip(transitions, rates)}


def load_training_dataframe(base_dir: Path):

    settings_path = base_dir / "settings.csv"
    odes_dir = base_dir / "odes"

    if not settings_path.exists():
        raise FileNotFoundError(f"Missing settings.csv: {settings_path}")
    if not odes_dir.exists():
        raise FileNotFoundError(f"Missing odes directory: {odes_dir}")

    settings_df = pd.read_csv(settings_path)

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
            out[t] = target_dict.get(t, 0.0)

        rows.append(out)

    if not rows:
        raise ValueError("No training rows found.")

    return pd.DataFrame(rows)


# ============================================================
# Linear regression
# ============================================================

@dataclass
class LinearRateModel:

    degree: int
    feature_names: List[str]
    transition_names: List[str]
    beta: np.ndarray

    mu: np.ndarray
    sigma: np.ndarray

    train_rmse_by_target: Dict[str, float]
    train_r2_by_target: Dict[str, float]

    overall_train_rmse: float
    overall_train_r2: float


def fit_surrogate(training_df: pd.DataFrame) -> LinearRateModel:

    mu, sigma = fit_input_scaler(training_df)
    X, feature_names = build_design_matrix(training_df, mu, sigma)

    transition_names = offdiag_transition_names()
    Y = training_df[transition_names].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Y_hat = X @ beta

    rmse_by_target = {}
    r2_by_target = {}

    for k, name in enumerate(transition_names):

        y = Y[:, k]
        yhat = Y_hat[:, k]

        rmse = np.sqrt(np.mean((y - yhat) ** 2))

        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rmse_by_target[name] = float(rmse)
        r2_by_target[name] = float(r2)

    overall_rmse = float(np.sqrt(np.mean((Y - Y_hat) ** 2)))

    ss_res_all = np.sum((Y - Y_hat) ** 2)
    ss_tot_all = np.sum((Y - np.mean(Y)) ** 2)
    overall_r2 = float(1 - ss_res_all / ss_tot_all) if ss_tot_all > 0 else np.nan

    return LinearRateModel(
        degree=POLY_DEGREE,
        feature_names=feature_names,
        transition_names=transition_names,
        beta=beta,
        mu=mu,
        sigma=sigma,
        train_rmse_by_target=rmse_by_target,
        train_r2_by_target=r2_by_target,
        overall_train_rmse=overall_rmse,
        overall_train_r2=overall_r2,
    )


# ============================================================
# Save surrogate
# ============================================================

def save_surrogate_model(base_dir: Path, model: LinearRateModel):

    out_dir = base_dir / "surrogates"
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "ode_rate_surrogate.npz"

    np.savez_compressed(
        path,
        degree=model.degree,
        feature_names=np.array(model.feature_names, dtype="U"),
        transition_names=np.array(model.transition_names, dtype="U"),
        beta=model.beta,
        mu=model.mu,
        sigma=model.sigma,
        overall_train_rmse=model.overall_train_rmse,
        overall_train_r2=model.overall_train_r2,
        rmse_keys=np.array(list(model.train_rmse_by_target.keys()), dtype="U"),
        rmse_vals=np.array(list(model.train_rmse_by_target.values()), dtype=float),
        r2_keys=np.array(list(model.train_r2_by_target.keys()), dtype="U"),
        r2_vals=np.array(list(model.train_r2_by_target.values()), dtype=float),
    )

    return path


# ============================================================
# Main
# ============================================================

def train_and_save_surrogate(base_dir: Path = BASE_DIR):

    training_df = load_training_dataframe(base_dir)
    model = fit_surrogate(training_df)

    print(f"\nPolynomial surrogate (degree = {POLY_DEGREE})")
    print("samples:", len(training_df))
    print("features:", len(model.feature_names))
    print("overall train RMSE:", model.overall_train_rmse)
    print("overall train R2:", model.overall_train_r2)

    path = save_surrogate_model(base_dir, model)

    print("\nSaved surrogate model to:", path)

    return path


if __name__ == "__main__":
    train_and_save_surrogate()