"""
Compute long-run ODE state ratios from fitted CTMC generators.

For each model:
    1. Load fitted generator Q from: odes/{model_name}.npz
    2. Load initial state x0 from the first row of: states/{model_name}.csv
    3. Compute the asymptotic limit
            x_inf = lim_{t->infinity} x0 expm(Q t)
       using the reusable CTMC steady-state utilities
    4. Save long-run state ratios to:
            steady_states.csv

Output columns:
    model_name, steady_S, steady_Q, steady_L, steady_B
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ctmc_steady_state import compute_long_run_state
from utils import BASE_DIR, get_all_model_names


STATE_ORDER = ["S", "Q", "L", "B"]


def load_Q(model_name: str, base_dir: Path = BASE_DIR) -> np.ndarray:
    path = base_dir / "odes" / f"{model_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing ODE file: {path}")

    data = np.load(path)
    Q = np.asarray(data["Q"], dtype=float)

    if Q.shape != (4, 4):
        raise ValueError(f"{model_name}: expected Q shape (4,4), got {Q.shape}")

    return Q


def load_initial_state(model_name: str, base_dir: Path = BASE_DIR) -> np.ndarray:
    path = base_dir / "states" / f"{model_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing state file: {path}")

    df = pd.read_csv(path)

    required = ["ratio_S", "ratio_Q", "ratio_L", "ratio_B"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{model_name}: missing columns in states CSV: {missing}")

    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)

    x0 = np.array(
        [
            df.loc[0, "ratio_S"],
            df.loc[0, "ratio_Q"],
            df.loc[0, "ratio_L"],
            df.loc[0, "ratio_B"],
        ],
        dtype=float,
    )

    x0 = np.clip(x0, 0.0, None)
    s = x0.sum()
    if s <= 0:
        raise ValueError(f"{model_name}: initial state sums to zero")

    return x0 / s


def build_steady_state_table(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    rows = []

    for model_name in get_all_model_names():
        Q = load_Q(model_name, base_dir=base_dir)
        x0 = load_initial_state(model_name, base_dir=base_dir)
        x_inf = compute_long_run_state(Q, x0)

        rows.append(
            {
                "model_name": model_name,
                "steady_S": float(x_inf[0]),
                "steady_Q": float(x_inf[1]),
                "steady_L": float(x_inf[2]),
                "steady_B": float(x_inf[3]),
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("model_name").reset_index(drop=True)


def save_steady_states(base_dir: Path = BASE_DIR) -> Path:
    df = build_steady_state_table(base_dir=base_dir)
    out_path = base_dir / "steady_states.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    save_steady_states(BASE_DIR)