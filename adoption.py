"""
Compute ODE-predicted final adoption for each experimental model.

This script uses the fitted continuous-time Markov generator (Q matrix)
obtained from ODE reduction of the SQLB state dynamics to estimate
organizational adoption at the end of the simulation horizon.

For each model:
    1. Load fitted generator Q from: odes/{model_name}.npz
    2. Load initial state distribution from: states/{model_name}.csv
    3. Simulate the ODE system:
            dx/dt = x Q
       using explicit Euler integration over T steps
    4. Compute final adoption:
            A_final = Q(T) + L(T)
    5. Merge with model design variables from settings.csv:
            - agents_average_initial_opinion
            - technology_success_rate
    6. Save aggregated results to:
            final_adoption.csv

Interpretation:
    - A_final represents the ODE-predicted fraction of agents
      in adoption states (Q + L) at time horizon T.
    - This provides a reduced-order surrogate of the ABM
      suitable for tradeoff analysis and decision frameworks.

Assumptions:
    - The fitted generator Q is valid (rows sum to zero).
    - Time step dt matches the original simulation step.
    - Explicit Euler integration is numerically stable over T.
    - All models share a comparable time horizon.

Output:
    final_adoption.csv with columns:
        name,
        agents_average_initial_opinion,
        technology_success_rate,
        final_adoption
"""

import numpy as np
import pandas as pd
from pathlib import Path

from utils import BASE_DIR, get_all_model_names, load_settings


def ode_final_adoption(base_dir: Path, model_name: str, T: int | None = None) -> float:
    # load Q
    data = np.load(base_dir / "odes" / f"{model_name}.npz", allow_pickle=False)
    Q = data["Q"].astype(float)
    dt = float(data["dt"][0]) if "dt" in data else 1.0

    # load initial x0 (and horizon from states)
    df = pd.read_csv(base_dir / "states" / f"{model_name}.csv").sort_values("t").reset_index(drop=True)
    x0 = np.array(
        [df.loc[0, "ratio_S"], df.loc[0, "ratio_Q"], df.loc[0, "ratio_L"], df.loc[0, "ratio_B"]],
        dtype=float,
    )

    if T is None:
        T = len(df) - 1  # number of steps to advance

    x = x0.copy()
    for _ in range(T):
        x = x + dt * (x @ Q)

        # numerical safety (Euler can drift)
        x = np.clip(x, 0.0, 1.0)
        s = float(x.sum())
        if s > 0:
            x = x / s

    return float(x[1] + x[2])  # Q + L


def build_final_adoption_table(base_dir: Path, *, T: int | None = None) -> pd.DataFrame:
    settings = pd.DataFrame(load_settings())

    # make sure numeric
    settings["agents_average_initial_opinion"] = pd.to_numeric(settings["agents_average_initial_opinion"], errors="raise")
    settings["technology_success_rate"] = pd.to_numeric(settings["technology_success_rate"], errors="raise")

    # index by model name for fast lookup
    settings = settings.set_index("name", drop=False)

    rows = []
    model_names = get_all_model_names()

    for model_name in model_names:
        if model_name not in settings.index:
            # Skip if no settings row matches this model name
            continue

        try:
            final_adoption = ode_final_adoption(base_dir, model_name, T=T)
        except FileNotFoundError:
            # Skip if missing states CSV or odes NPZ
            continue

        r = settings.loc[model_name]
        rows.append(
            {
                "name": model_name,
                "final_adoption": float(final_adoption),
            }
        )

    df_out = pd.DataFrame(rows)
    if df_out.empty:
        raise ValueError("No rows produced. Check that model names match settings.csv and that states/ and odes/ files exist.")

    return df_out.sort_values(["name"]).reset_index(drop=True)


def save_final_adoption_csv(base_dir: Path, df: pd.DataFrame) -> Path:
    out_path = base_dir / "final_adoption.csv"
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    df = build_final_adoption_table(BASE_DIR, T=None)  # or set T=100 explicitly
    out_path = save_final_adoption_csv(BASE_DIR, df)
    print(f"Saved: {out_path}")
    print(df.head())