"""
Useful stuff for the rest of pipeline.
"""

import csv
from pathlib import Path
import trustdynamics as td

BASE_DIR = Path(__file__).resolve().parent


def get_all_model_names(teams_num=None) -> list[str]:
    with open(BASE_DIR / "settings.csv" , newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row["name"] for row in reader]
    
#models_names = get_all_model_names(teams_num=1)
#print(len(models_names))

def get_model(name: str) -> td.Model:
    path = BASE_DIR / "models"
    path = path / f"{name}.json"
    model = td.Model.load(path)
    return model

#model = get_model("models_1", '0_0_0')

def load_settings():
    settings_path = BASE_DIR / "settings.csv"
    with settings_path.open(mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        settings = [dict(row) for row in reader]
    return settings