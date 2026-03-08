from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from ctmc_steady_state import compute_long_run_state
from predict_surrogate import (
    STATE_ORDER,
    load_surrogate_model,
    predict_generator,
)


DEFAULT_X0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
# Replace this if your ABM uses a different standard initial composition.


def predict_surrogate_steady_state(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    initial_state: np.ndarray | None = None,
    surrogate_path: Path | None = None,
) -> Dict[str, float]:
    """
    Predict long-run state ratios from the ODE-parameter surrogate.

    This computes:
        x_inf = lim_{t->infinity} x0 expm(Q t)

    where Q is predicted by the surrogate and x0 is the supplied initial state.
    """
    model = load_surrogate_model(surrogate_path)

    Q, _ = predict_generator(
        model,
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
    )

    if initial_state is None:
        x0 = DEFAULT_X0.copy()
    else:
        x0 = np.asarray(initial_state, dtype=float).copy()
        x0 = np.clip(x0, 0.0, None)
        s = x0.sum()
        if s <= 0:
            raise ValueError("initial_state must have positive sum")
        x0 /= s

    x_inf = compute_long_run_state(Q, x0)

    return {
        "steady_S": float(x_inf[0]),
        "steady_Q": float(x_inf[1]),
        "steady_L": float(x_inf[2]),
        "steady_B": float(x_inf[3]),
        "steady_adoption": float(x_inf[1] + x_inf[2]),
    }


def predict_surrogate_steady_state_with_details(
    *,
    teams_num: float,
    teams_size: float,
    agents_average_initial_opinion: float,
    technology_success_rate: float,
    initial_state: np.ndarray | None = None,
    surrogate_path: Path | None = None,
) -> Dict[str, object]:
    model = load_surrogate_model(surrogate_path)

    Q, rates = predict_generator(
        model,
        teams_num=teams_num,
        teams_size=teams_size,
        agents_average_initial_opinion=agents_average_initial_opinion,
        technology_success_rate=technology_success_rate,
    )

    if initial_state is None:
        x0 = DEFAULT_X0.copy()
    else:
        x0 = np.asarray(initial_state, dtype=float).copy()
        x0 = np.clip(x0, 0.0, None)
        s = x0.sum()
        if s <= 0:
            raise ValueError("initial_state must have positive sum")
        x0 /= s

    x_inf = compute_long_run_state(Q, x0)

    steady_state = {
        "steady_S": float(x_inf[0]),
        "steady_Q": float(x_inf[1]),
        "steady_L": float(x_inf[2]),
        "steady_B": float(x_inf[3]),
        "steady_adoption": float(x_inf[1] + x_inf[2]),
    }

    return {
        "Q": Q,
        "rates": rates,
        "initial_state": x0,
        "steady_state": steady_state,
    }


if __name__ == "__main__":
    out = predict_surrogate_steady_state_with_details(
        teams_num=10,
        teams_size=10,
        agents_average_initial_opinion=0.2,
        technology_success_rate=0.7,
        initial_state=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    print("\nPredicted generator Q:")
    print(out["Q"])

    print("\nPredicted steady-state ratios:")
    for k, v in out["steady_state"].items():
        print(f"{k}: {v:.6f}")