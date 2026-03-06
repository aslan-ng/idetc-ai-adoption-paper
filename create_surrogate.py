# surrogate_ode.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import BASE_DIR


STATE_ORDER = ["S", "Q", "L", "B"]


# ============================================================
# Feature engineering
# ============================================================

FEATURE_COLUMNS = [
    "teams_num",
    "teams_size",
    "agents_average_initial_opinion",
    "technology_success_rate",
]


def build_feature_vector(row: pd.Series) -> Dict[str, float]:
    """
    Create linear-regression features from one row of settings.
    Includes main effects + pairwise interactions + squared terms.

    Still linear in coefficients, but more expressive than pure linear.
    """
    tn = float(row["teams_num"])
    ts = float(row["teams_size"])
    op = float(row["agents_average_initial_opinion"])
    sr = float(row["technology_success_rate"])

    feats = {
        "bias": 1.0,
        "teams_num": tn,
        "teams_size": ts,
        "initial_opinion": op,
        "success_rate": sr,

        "teams_num_sq": tn ** 2,
        "teams_size_sq": ts ** 2,
        "initial_opinion_sq": op ** 2,
        "success_rate_sq": sr ** 2,

        "teams_num_x_teams_size": tn * ts,
        "teams_num_x_initial_opinion": tn * op,
        "teams_num_x_success_rate": tn * sr,
        "teams_size_x_initial_opinion": ts * op,
        "teams_size_x_success_rate": ts * sr,
        "initial_opinion_x_success_rate": op * sr,
    }
    return feats


def build_design_matrix(settings_df: pd.DataFrame) -> tuple[np.ndarray, List[str]]:
    feature_dicts = [build_feature_vector(row) for _, row in settings_df.iterrows()]
    feature_names = list(feature_dicts[0].keys())

    X = np.array(
        [[fd[name] for name in feature_names] for fd in feature_dicts],
        dtype=float,
    )
    return X, feature_names


# ============================================================
# Target loading
# ============================================================

def offdiag_transition_names(state_order: List[str] = STATE_ORDER) -> List[str]:
    names = []
    for i, si in enumerate(state_order):
        for j, sj in enumerate(state_order):
            if i != j:
                names.append(f"{si}->{sj}")
    return names


def load_one_ode_result(npz_path: Path) -> Dict[str, float]:
    """
    Load one saved ODE fit artifact and return off-diagonal rates as a dict.
    """
    data = np.load(npz_path, allow_pickle=True)

    transitions = data["transitions"].tolist()
    rates = data["rates"].astype(float).tolist()

    return {str(k): float(v) for k, v in zip(transitions, rates)}


def load_training_dataframe(base_dir: Path) -> pd.DataFrame:
    """
    Merge settings.csv with saved ode results by model_name.
    Returns one training row per model with inputs + targets.
    """
    settings_path = base_dir / "settings.csv"
    odes_dir = base_dir / "odes"

    if not settings_path.exists():
        raise FileNotFoundError(f"Missing settings.csv: {settings_path}")
    if not odes_dir.exists():
        raise FileNotFoundError(f"Missing odes directory: {odes_dir}")

    settings_df = pd.read_csv(settings_path)

    if "name" not in settings_df.columns:
        raise ValueError("settings.csv must contain a 'name' column.")

    required_input_cols = [
        "name",
        "teams_num",
        "teams_size",
        "agents_average_initial_opinion",
        "technology_success_rate",
    ]
    missing = set(required_input_cols) - set(settings_df.columns)
    if missing:
        raise ValueError(f"Missing settings columns: {sorted(missing)}")

    rows = []
    expected_targets = offdiag_transition_names()

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

        for tname in expected_targets:
            out[tname] = float(target_dict.get(tname, 0.0))

        rows.append(out)

    if not rows:
        raise ValueError("No training rows found. Make sure .npz ODE files exist.")

    return pd.DataFrame(rows)


# ============================================================
# Linear regression (dependency-free)
# ============================================================

@dataclass
class LinearRateModel:
    feature_names: List[str]
    transition_names: List[str]
    beta: np.ndarray  # shape (n_features, n_targets)
    train_rmse_by_target: Dict[str, float]
    train_r2_by_target: Dict[str, float]


def fit_multioutput_linear_model(training_df: pd.DataFrame) -> LinearRateModel:
    """
    Fit one linear model for all off-diagonal rates simultaneously:
        Y = X B

    Uses ordinary least squares via np.linalg.lstsq.
    """
    X, feature_names = build_design_matrix(training_df)
    transition_names = offdiag_transition_names()

    Y = training_df[transition_names].to_numpy(dtype=float)

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    Y_hat = X @ beta

    rmse_by_target: Dict[str, float] = {}
    r2_by_target: Dict[str, float] = {}

    for k, tname in enumerate(transition_names):
        y = Y[:, k]
        yhat = Y_hat[:, k]
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rmse_by_target[tname] = rmse
        r2_by_target[tname] = r2

    return LinearRateModel(
        feature_names=feature_names,
        transition_names=transition_names,
        beta=beta,
        train_rmse_by_target=rmse_by_target,
        train_r2_by_target=r2_by_target,
    )


# ============================================================
# Prediction utilities
# ============================================================

def _features_from_inputs(
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    feature_names: List[str],
) -> np.ndarray:
    row = pd.Series({
        "teams_num": teams_num,
        "teams_size": teams_size,
        "agents_average_initial_opinion": agents_average_initial_opinion,
        "technology_success_rate": technology_success_rate,
    })
    feat_dict = build_feature_vector(row)
    return np.array([feat_dict[name] for name in feature_names], dtype=float)


def predict_rates(
    model: LinearRateModel,
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    clip_nonnegative: bool = True,
) -> Dict[str, float]:
    """
    Predict off-diagonal rates for one input setting.
    """
    x = _features_from_inputs(
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
        feature_names=model.feature_names,
    )

    y = x @ model.beta

    if clip_nonnegative:
        y = np.maximum(y, 0.0)

    return {
        tname: float(val)
        for tname, val in zip(model.transition_names, y)
    }


def rates_to_generator(
    rate_dict: Dict[str, float],
    state_order: List[str] = STATE_ORDER,
) -> np.ndarray:
    """
    Convert off-diagonal rates into a valid CTMC generator matrix Q.
    """
    idx = {s: i for i, s in enumerate(state_order)}
    Q = np.zeros((len(state_order), len(state_order)), dtype=float)

    for key, val in rate_dict.items():
        src, dst = key.split("->")
        i = idx[src]
        j = idx[dst]
        if i == j:
            continue
        Q[i, j] = max(0.0, float(val))

    np.fill_diagonal(Q, -np.sum(Q, axis=1))
    return Q


def predict_generator(
    model: LinearRateModel,
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
) -> tuple[np.ndarray, Dict[str, float]]:
    rate_dict = predict_rates(
        model,
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
        clip_nonnegative=True,
    )
    Q = rates_to_generator(rate_dict)
    return Q, rate_dict


# ============================================================
# Save / load surrogate
# ============================================================

def save_surrogate_model(base_dir: Path, model: LinearRateModel) -> Path:
    out_dir = base_dir / "surrogates"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "ode_rate_surrogate.npz"

    np.savez_compressed(
        out_path,
        feature_names=np.array(model.feature_names, dtype="U"),
        transition_names=np.array(model.transition_names, dtype="U"),
        beta=model.beta,
        rmse_keys=np.array(list(model.train_rmse_by_target.keys()), dtype="U"),
        rmse_vals=np.array(list(model.train_rmse_by_target.values()), dtype=float),
        r2_keys=np.array(list(model.train_r2_by_target.keys()), dtype="U"),
        r2_vals=np.array(list(model.train_r2_by_target.values()), dtype=float),
    )
    return out_path


def load_surrogate_model(npz_path: Path) -> LinearRateModel:
    data = np.load(npz_path, allow_pickle=True)

    feature_names = data["feature_names"].tolist()
    transition_names = data["transition_names"].tolist()
    beta = data["beta"]

    rmse_keys = data["rmse_keys"].tolist()
    rmse_vals = data["rmse_vals"].astype(float).tolist()
    r2_keys = data["r2_keys"].tolist()
    r2_vals = data["r2_vals"].astype(float).tolist()

    return LinearRateModel(
        feature_names=[str(x) for x in feature_names],
        transition_names=[str(x) for x in transition_names],
        beta=np.array(beta, dtype=float),
        train_rmse_by_target={str(k): float(v) for k, v in zip(rmse_keys, rmse_vals)},
        train_r2_by_target={str(k): float(v) for k, v in zip(r2_keys, r2_vals)},
    )


# ============================================================
# Reporting
# ============================================================

def print_fit_summary(model: LinearRateModel) -> None:
    print("\nSurrogate fit summary")
    print("-" * 60)
    for tname in model.transition_names:
        rmse = model.train_rmse_by_target[tname]
        r2 = model.train_r2_by_target[tname]
        print(f"{tname:>4s} | RMSE = {rmse:.6f} | R^2 = {r2:.4f}")
    print("-" * 60)


# ============================================================
# Main workflow
# ============================================================

def train_and_save_surrogate(base_dir: Path = BASE_DIR) -> Path:
    training_df = load_training_dataframe(base_dir)
    model = fit_multioutput_linear_model(training_df)
    print_fit_summary(model)
    out_path = save_surrogate_model(base_dir, model)
    print(f"\nSaved surrogate model to: {out_path}")
    return out_path


if __name__ == "__main__":
    surrogate_path = train_and_save_surrogate(BASE_DIR)

    # Example usage
    model = load_surrogate_model(surrogate_path)

    Q_pred, rates_pred = predict_generator(
        model,
        teams_num=10,
        teams_size=10,
        agents_average_initial_opinion=0.2,
        technology_success_rate=0.8,
    )

    print("\nPredicted off-diagonal rates:")
    for k, v in rates_pred.items():
        print(f"{k}: {v:.6f}")

    print("\nPredicted generator Q:")
    print(Q_pred)