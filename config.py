"""
Generate the full experimental design for SQLB simulation runs.

This script defines all global simulation parameters and constructs
the Cartesian product of:

    - random seed (replication group)
    - agents_average_initial_opinion
    - technology_success_rate

Each unique combination becomes one simulation configuration,
saved as a row in settings.csv.

Design structure:
    - seeds_num controls stochastic replications.
    - agents_average_initial_opinion_list spans the initial
      mean sentiment toward the technology.
    - technology_success_rate_list spans objective AI accuracy.
    - Each (seed, opinion, accuracy) triplet defines one model.

Naming convention:
    name = "{seed_index}_{opinion_index}_{accuracy_index}"

    This ensures:
        - deterministic mapping between settings and model outputs
        - compatibility with downstream ODE fitting and tradeoff analysis

Output:
    settings.csv with columns:
        name
        group                  (seed group identifier)
        seed
        agents_average_initial_opinion
        technology_success_rate

Experimental size:
    total_runs = seeds_num × len(opinion_list) × len(success_rate_list)

This file defines the parameter grid that later enables:
    - ODE reduction and k estimation
    - final adoption computation
    - tradeoff contour analysis
    - decision framework construction
"""

import csv
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


# Steps
steps = 100


# Master seed
master_seed = 42
rng = np.random.default_rng(master_seed)

# Seeds
seeds_num = 3
seeds_list = rng.integers(low=0, high=2**32 - 1, size=seeds_num).tolist()


# Agents
def calculate_agents_num(teams_num, teams_size):
    return teams_num * teams_size

def calculate_agents_connection_probability(
    teams_size: int,
    target_density: float = 0.29,
    empirical_mean_group_size: float = 6.16,
) -> float:
    """
    Within-team ER edge probability p_AA.

    Matches empirical work-group density (mean density=0.29 at mean group size=6.16),
    converts to target expected within-team degree:
        k_intra = density * (mean_group_size - 1)
    then maps to your team size n:
        p_AA = k_intra / (n - 1)

    Returns p_AA clipped to [0, 1].
    """
    n = int(teams_size)
    if n <= 1:
        return 0.0

    k_intra = float(target_density) * (float(empirical_mean_group_size) - 1.0)
    p = k_intra / (n - 1.0)
    return float(np.clip(p, 0.0, 1.0))

agents_average_initial_opinion_list = [
    round(x, 1)
    for x in np.linspace(-1, 1, 21)
]
technology_use_cutoff_opinion = -0.2


# Teams
def calculate_teams_connection_probability(
    teams_num: int,
    avg_team_degree: float = 5.0,
) -> float:
    """
    Team-level ER edge probability p_TT.

    Organizational studies show that teams typically maintain a small number
    of meaningful cross-team collaboration ties (≈4–6). To keep the expected
    number of cross-team connections per team constant as the organization
    grows, we convert the expected team degree into an ER probability:

        p_TT = k_team / (T - 1)

    where:
        k_team = expected number of cross-team links per team
        T      = number of teams in the organization

    Returns p_TT clipped to [0, 1].
    """
    T = int(teams_num)
    if T <= 1:
        return 0.0

    p = avg_team_degree / (T - 1.0)
    return float(np.clip(p, 0.0, 1.0))

teams_size_list = [5, 10, 15, 20]
teams_num_list = [1, 10, 20, 30]


# Technology
technology_success_rate_list = [
    round(x, 1)
    for x in np.linspace(0, 1, 11)
]


if __name__ == "__main__":
    settings: list[dict] = []
    for s, seed in enumerate(seeds_list):
        for o, agents_average_initial_opinion in enumerate(agents_average_initial_opinion_list):
            for t, technology_success_rate in enumerate(technology_success_rate_list):
                for tn, teams_num in enumerate(teams_num_list):
                    for ts, teams_size in enumerate(teams_size_list):
                        setting = {
                            "name": f"{s}_{o}_{t}_{tn}_{ts}",
                            "seed": seed,
                            "agents_num": calculate_agents_num(teams_num=teams_num, teams_size=teams_size),
                            "teams_num": teams_num,
                            "teams_size": teams_size,
                            "agents_connection_probability": calculate_agents_connection_probability(teams_size=teams_size),
                            "teams_connection_probability": calculate_teams_connection_probability(teams_num=teams_num),
                            "agents_average_initial_opinion": agents_average_initial_opinion,
                            "technology_success_rate": technology_success_rate,
                        }
                        settings.append(setting)

    fieldnames = settings[0].keys()

    settings_path = BASE_DIR / "settings.csv"
    with open(settings_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(settings)

    print("Total runs:", len(settings))