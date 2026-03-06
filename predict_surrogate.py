# predict_surrogate.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils import BASE_DIR


STATE_ORDER = ["S", "Q", "L", "B"]


class LinearRateModel:
    """
    Container for the saved surrogate model that maps:
        (teams_num, teams_size, initial_opinion, success_rate)
    to:
        off-diagonal CTMC transition rates.
    """

    def __init__(
        self,
        feature_names: List[str],
        transition_names: List[str],
        beta: np.ndarray,
        train_rmse_by_target: Dict[str, float],
        train_r2_by_target: Dict[str, float],
    ) -> None:
        self.feature_names = feature_names
        self.transition_names = transition_names
        self.beta = beta
        self.train_rmse_by_target = train_rmse_by_target
        self.train_r2_by_target = train_r2_by_target


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

    rmse_keys = [str(x) for x in data["rmse_keys"].tolist()]
    rmse_vals = [float(x) for x in data["rmse_vals"].astype(float).tolist()]
    r2_keys = [str(x) for x in data["r2_keys"].tolist()]
    r2_vals = [float(x) for x in data["r2_vals"].astype(float).tolist()]

    return LinearRateModel(
        feature_names=feature_names,
        transition_names=transition_names,
        beta=beta,
        train_rmse_by_target={k: v for k, v in zip(rmse_keys, rmse_vals)},
        train_r2_by_target={k: v for k, v in zip(r2_keys, r2_vals)},
    )


def _build_feature_dict(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
) -> Dict[str, float]:
    """
    Must match the feature engineering used in create_surrogate.py.
    """
    tn = float(teams_num)
    ts = float(teams_size)
    op = float(agents_average_initial_opinion)
    sr = float(technology_success_rate)

    return {
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


def _build_feature_vector(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    feature_names: List[str],
) -> np.ndarray:
    feat_dict = _build_feature_dict(
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
    )
    return np.array([feat_dict[name] for name in feature_names], dtype=float)


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

    # Solve:
    #   Q^T * pi^T = 0
    # with final row replaced by normalization sum(pi)=1
    A = Q.T.copy()
    A[-1, :] = 1.0

    b = np.zeros(n, dtype=float)
    b[-1] = 1.0

    pi, *_ = np.linalg.lstsq(A, b, rcond=None)

    # numerical cleanup
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
    Main end-to-end function you want.

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
        "rates": rates,
        "Q": Q,
        "steady_state": steady_state,
    }


if __name__ == "__main__":
    result = predict_full_surrogate_output(
        teams_num=10,
        teams_size=10,
        agents_average_initial_opinion=0.2,
        technology_success_rate=0.8,
    )

    print("\nPredicted off-diagonal rates:")
    for k, v in result["rates"].items():
        print(f"{k}: {v:.6f}")

    print("\nPredicted generator Q:")
    print(result["Q"])

    print("\nPredicted steady-state ratios:")
    for k, v in result["steady_state"].items():
        print(f"{k}: {v:.6f}")