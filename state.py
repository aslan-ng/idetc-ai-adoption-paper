"""
Extract SQLB state time series from simulated agent-based models.

This script reconstructs the discrete organizational state trajectory
(S, Q, L, B) from each stored ABM simulation and converts it into
fractional state time series suitable for ODE reduction and tradeoff analysis.

For each model:

    1. Load the saved trustdynamics model object.
    2. For every time step t and every agent:
           - Retrieve agent opinion.
           - Compute inbound trust mass.
           - Compute "loudness" = opinion × inbound_trust.
           - Classify agent into SQLB state using hierarchical logic:
                B: opinion < technology_use_cutoff_opinion
                S: technology_use_cutoff_opinion ≤ opinion < 0
                Q: opinion ≥ 0 and loudness < LOUDNESS_BOUND
                L: opinion ≥ 0 and loudness ≥ LOUDNESS_BOUND
    3. Aggregate counts across agents.
    4. Convert counts to fractions.
    5. Save per-step ratios to:
           states/{model_name}.csv

Output format:
    CSV with columns:
        t,
        ratio_S,
        ratio_Q,
        ratio_L,
        ratio_B

Properties:
    - Fractions sum to 1 at each timestep (by construction).
    - Time horizon equals `steps` defined in config.
    - LOUDNESS_BOUND controls activation threshold for advocacy (L vs Q).
    - technology_use_cutoff_opinion controls belief threshold (B vs S).

Role in the modeling pipeline:
    ABM → state fractions (this file)
        → ODE generator fitting
        → final adoption estimation
        → tradeoff contour / decision framework analysis

This module defines the empirical bridge between agent-level dynamics
and reduced-order organizational state dynamics.
"""

import os
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import trustdynamics as td

from config import steps, BASE_DIR, technology_use_cutoff_opinion
from utils import get_all_model_names, get_model

LOUDNESS_BOUND = 0.5


def sqlb_state(opinion: float, loudness: float, *, loudness_bound: float) -> str:
    # ---- Belief layer ----
    if opinion < technology_use_cutoff_opinion:
        return "B"
    if opinion < 0.0:
        return "S"

    # ---- Activation layer (only for advocates) ----
    if loudness < loudness_bound:
        return "Q"
    return "L"


def get_inbound_trust(model: td.Model, agent: int | str, history_index: int) -> float:
    neighbors = list(model.organization.agents_connected_to(agent))
    inbound = 0.0
    for j in neighbors:
        inbound += float(
            model.organization.get_agent_trust(j, agent, history_index=history_index)
        )
    return inbound


def classify_agent(model: td.Model, agent: int | str, t: int, loudness_bound: float) -> str:
    opinion = float(model.organization.get_agent_opinion(agent, t))
    inbound_trust = get_inbound_trust(model, agent, t)
    loudness = opinion * inbound_trust
    return sqlb_state(opinion, loudness, loudness_bound=loudness_bound)


def process_one_model(model_name: str) -> tuple[str, str]:
    states_dir = BASE_DIR / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    out_path = states_dir / f"{model_name}.csv"
    if out_path.exists():
        return model_name, "skipped (exists)"

    model = get_model(model_name)
    agents = list(model.organization.all_agent_ids)
    n_agents = len(agents)
    if n_agents == 0:
        return model_name, "skipped (no agents)"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["t", "ratio_S", "ratio_Q", "ratio_L", "ratio_B"],
        )
        writer.writeheader()

        for t in range(steps):
            counts = {"S": 0, "Q": 0, "L": 0, "B": 0}
            for agent in agents:
                st = classify_agent(model, agent, t, loudness_bound=LOUDNESS_BOUND)
                counts[st] += 1

            writer.writerow(
                {
                    "t": t,
                    "ratio_S": counts["S"] / n_agents,
                    "ratio_Q": counts["Q"] / n_agents,
                    "ratio_L": counts["L"] / n_agents,
                    "ratio_B": counts["B"] / n_agents,
                }
            )

    return model_name, "wrote"


def main():
    model_names = get_all_model_names()

    max_workers = max(1, (os.cpu_count() or 2) - 2)

    total = len(model_names)
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one_model, name) for name in model_names]
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            print(f"[{done}/{total}] {name}: {status}")


if __name__ == "__main__":
    main()