"""
Predict reduced-order CTMC dynamics from the trained surrogate model.

What this file does:
1. Loads the saved polynomial surrogate (`surrogates/ode_rate_surrogate.npz`).
2. Rebuilds the same standardized polynomial feature vector used in training.
3. Predicts off-diagonal SQLB transition rates and converts them to a valid
   generator matrix Q.
4. Computes steady-state SQLB ratios from Q.

How it is used in the pipeline:
- Called by analysis/plot scripts to map scenario inputs
  (`teams_num`, `teams_size`, `agents_average_initial_opinion`,
  `technology_success_rate`) to predicted ODE behavior.
- Provides the core prediction interface used for threshold plots, composition
  plots, and surrogate steady-state validation in the IDETC study.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import BASE_DIR


STATE_ORDER = ["S", "Q", "L", "B"]
BASE_INPUT_COLUMNS = [
    "teams_num",
    "teams_size",
    "agents_average_initial_opinion",
    "technology_success_rate",
]


class LinearRateModel:
    """
    Container for the saved surrogate model that maps:
        (teams_num, teams_size, initial_opinion, success_rate)
    to:
        off-diagonal CTMC transition rates.
    """

    def __init__(
        self,
        degree: int,
        feature_names: List[str],
        transition_names: List[str],
        beta: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        train_rmse_by_target: Dict[str, float],
        train_r2_by_target: Dict[str, float],
        overall_train_rmse: float,
        overall_train_r2: float,
    ) -> None:
        self.degree = degree
        self.feature_names = feature_names
        self.transition_names = transition_names
        self.beta = beta
        self.mu = mu
        self.sigma = sigma
        self.train_rmse_by_target = train_rmse_by_target
        self.train_r2_by_target = train_r2_by_target
        self.overall_train_rmse = overall_train_rmse
        self.overall_train_r2 = overall_train_r2


def load_surrogate_model(npz_path: Path | None = None) -> LinearRateModel:
    """
    Load the saved surrogate model produced by create_surrogate.py.

    Expected file:
        BASE_DIR / "surrogates" / "ode_rate_surrogate.npz"
    """
    if npz_path is None:
        npz_path = BASE_DIR / "surrogates" / "ode_rate_surrogate.npz"

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Surrogate file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    feature_names = [str(x) for x in data["feature_names"].tolist()]
    transition_names = [str(x) for x in data["transition_names"].tolist()]
    beta = np.array(data["beta"], dtype=float)
    mu = np.array(data["mu"], dtype=float)
    sigma = np.array(data["sigma"], dtype=float)

    rmse_keys = [str(x) for x in data["rmse_keys"].tolist()]
    rmse_vals = [float(x) for x in data["rmse_vals"].astype(float).tolist()]
    r2_keys = [str(x) for x in data["r2_keys"].tolist()]
    r2_vals = [float(x) for x in data["r2_vals"].astype(float).tolist()]

    degree = int(np.array(data["degree"]).item())
    overall_train_rmse = float(np.array(data["overall_train_rmse"]).item())
    overall_train_r2 = float(np.array(data["overall_train_r2"]).item())

    return LinearRateModel(
        degree=degree,
        feature_names=feature_names,
        transition_names=transition_names,
        beta=beta,
        mu=mu,
        sigma=sigma,
        train_rmse_by_target={k: v for k, v in zip(rmse_keys, rmse_vals)},
        train_r2_by_target={k: v for k, v in zip(r2_keys, r2_vals)},
        overall_train_rmse=overall_train_rmse,
        overall_train_r2=overall_train_r2,
    )


def generate_polynomial_powers(n_vars: int, degree: int) -> List[Tuple[int, ...]]:
    powers = [(0,) * n_vars]

    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_vars), d):
            exps = [0] * n_vars
            for idx in combo:
                exps[idx] += 1
            powers.append(tuple(exps))

    return powers


def _standardize_inputs(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    x = np.array(
        [
            float(teams_num),
            float(teams_size),
            float(agents_average_initial_opinion),
            float(technology_success_rate),
        ],
        dtype=float,
    )
    return (x - mu) / sigma


def _build_feature_vector(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    model: LinearRateModel,
) -> np.ndarray:
    """
    Must match the feature engineering used in create_surrogate.py:
    - standardize inputs using saved mu/sigma
    - build full polynomial basis up to saved degree
    """
    x_scaled = _standardize_inputs(
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
        mu=model.mu,
        sigma=model.sigma,
    )

    powers = generate_polynomial_powers(
        n_vars=len(BASE_INPUT_COLUMNS),
        degree=model.degree,
    )

    x_features = np.ones(len(powers), dtype=float)

    for j, p in enumerate(powers):
        value = 1.0
        for var_idx, exp in enumerate(p):
            if exp != 0:
                value *= x_scaled[var_idx] ** exp
        x_features[j] = value

    if len(x_features) != len(model.feature_names):
        raise ValueError(
            "Feature length mismatch. The saved surrogate model and prediction "
            "feature builder are inconsistent."
        )

    return x_features


def predict_offdiag_rates(
    model: LinearRateModel,
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    clip_nonnegative: bool = True,
) -> Dict[str, float]:
    """
    Predict off-diagonal transition rates:
        S->Q, S->L, ..., B->L
    """
    x = _build_feature_vector(
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
        model=model,
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
    Convert off-diagonal rates to a valid CTMC generator matrix Q.
    """
    idx = {s: i for i, s in enumerate(state_order)}
    n = len(state_order)

    Q = np.zeros((n, n), dtype=float)

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
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Predict generator matrix Q and its off-diagonal rate dictionary.
    """
    rate_dict = predict_offdiag_rates(
        model,
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
        clip_nonnegative=True,
    )
    Q = rates_to_generator(rate_dict, STATE_ORDER)
    return Q, rate_dict


def stationary_distribution_from_generator(
    Q: np.ndarray,
    state_order: List[str] = STATE_ORDER,
) -> Dict[str, float]:
    """
    Compute stationary distribution pi satisfying:
        pi Q = 0
        sum(pi) = 1

    Returns:
        {"S": ..., "Q": ..., "L": ..., "B": ...}
    """
    n = Q.shape[0]
    if Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be square")

    A = Q.T.copy()
    A[-1, :] = 1.0

    b = np.zeros(n, dtype=float)
    b[-1] = 1.0

    pi, *_ = np.linalg.lstsq(A, b, rcond=None)

    pi = np.maximum(pi, 0.0)
    total = pi.sum()
    if total <= 0:
        raise ValueError("Failed to compute stationary distribution.")
    pi = pi / total

    return {
        state: float(value)
        for state, value in zip(state_order, pi)
    }


def predict_steady_state_ratios(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    surrogate_path: Path | None = None,
) -> Dict[str, float]:
    """
    Main end-to-end function.

    Inputs:
        teams_num
        teams_size
        agents_average_initial_opinion
        technology_success_rate

    Returns:
        steady-state ratios of S, Q, L, B
    """
    model = load_surrogate_model(surrogate_path)

    Q, _ = predict_generator(
        model,
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
    )

    return stationary_distribution_from_generator(Q, STATE_ORDER)


def predict_full_surrogate_output(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    surrogate_path: Path | None = None,
) -> Dict[str, object]:
    """
    Optional convenience function.

    Returns:
        {
            "rates": {...},
            "Q": np.ndarray,
            "steady_state": {...}
        }
    """
    model = load_surrogate_model(surrogate_path)

    Q, rates = predict_generator(
        model,
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
    )
    steady_state = stationary_distribution_from_generator(Q, STATE_ORDER)

    return {
        "degree": model.degree,
        "overall_train_rmse": model.overall_train_rmse,
        "overall_train_r2": model.overall_train_r2,
        "rates": rates,
        "Q": Q,
        "steady_state": steady_state,
    }


if __name__ == "__main__":
    result = predict_full_surrogate_output(
        teams_num=10,
        teams_size=10,
        agents_average_initial_opinion=0.2,
        technology_success_rate=0.7,
    )

    print(f"\nLoaded polynomial surrogate (degree = {result['degree']})")
    print(f"overall train RMSE = {result['overall_train_rmse']:.6f}")
    print(f"overall train R2   = {result['overall_train_r2']:.6f}")

    print("\nPredicted off-diagonal rates:")
    for k, v in result["rates"].items():
        print(f"{k}: {v:.6f}")

    print("\nPredicted generator Q:")
    print(result["Q"])

    print("\nPredicted steady-state ratios:")
    for k, v in result["steady_state"].items():
        print(f"{k}: {v:.6f}")
