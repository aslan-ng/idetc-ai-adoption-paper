import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import trustdynamics as td

from utils import BASE_DIR, load_settings
from config import steps, technology_use_cutoff_opinion

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_one(setting: dict) -> tuple[str, str]:
    name = str(setting["name"])
    path = MODELS_DIR / f"{name}.json"
    if path.exists():
        return name, "skipped (exists)"

    seed = int(setting["seed"])

    # Build org PER RUN (no shared / global org)
    organization = td.organization.generate_random_organization_structure(
        teams_num=int(setting["teams_num"]),
        agents_num=int(setting["agents_num"]),
        agents_connection_probability_inside_team=float(setting["agents_connection_probability"]),
        teams_connection_probability=float(setting["teams_connection_probability"]),
        technology_use_cutoff_opinion=float(technology_use_cutoff_opinion),
        seed=seed,  # <- tie topology to setting seed (or use a separate topology_seed field)
    )

    # Seed-dependent initialization (opinions etc.)
    organization.initialize(
        agents_average_initial_opinion=float(setting["agents_average_initial_opinion"]),
        seed=seed,
    )

    technology = td.Technology(
        success_rate=float(setting["technology_success_rate"]),
        seed=seed,
    )

    model = td.Model(organization, technology)
    model.run(steps=int(steps), show_progress=False)
    model.save(path=path)

    return name, "saved"


def main() -> None:
    settings = load_settings()

    max_workers = max(1, (os.cpu_count() or 2) - 2)
    total = len(settings)
    done = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one, st) for st in settings]
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            print(f"[{done}/{total}] {name}: {status}")


if __name__ == "__main__":
    main()