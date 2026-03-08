# validate_surrogate_steady_state.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from steady_state_surrogate_ode import predict_surrogate_steady_state
from utils import BASE_DIR


STATE_COLUMNS = ["steady_S", "steady_Q", "steady_L", "steady_B"]
INPUT_COLUMNS = [
    "teams_num",
    "teams_size",
    "agents_average_initial_opinion",
    "technology_success_rate",
]

VALIDATION_DIR = BASE_DIR / "surrogates" / "steady_state_validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def load_initial_state(model_name: str, base_dir: Path = BASE_DIR) -> np.ndarray:
    """
    Load x0 from the first row of states/<model_name>.csv.
    """
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
        raise ValueError(f"{model_name}: invalid initial state with nonpositive sum")

    return x0 / s


def composition_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def adoption_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a_true = y_true[:, 1] + y_true[:, 2]
    a_pred = y_pred[:, 1] + y_pred[:, 2]
    return float(np.sqrt(np.mean((a_true - a_pred) ** 2)))


def mean_kl_divergence(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = np.clip(y_true, eps, None)
    y_pred = np.clip(y_pred, eps, None)

    y_true = y_true / y_true.sum(axis=1, keepdims=True)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

    kl = np.sum(y_true * np.log(y_true / y_pred), axis=1)
    return float(np.mean(kl))


def mean_state_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = [r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])]
    return float(np.mean(scores))


def max_abs_sum_error(y_pred: np.ndarray) -> float:
    return float(np.max(np.abs(y_pred.sum(axis=1) - 1.0)))


def make_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main(base_dir: Path = BASE_DIR) -> None:
    settings = pd.read_csv(base_dir / "settings.csv")
    truth = pd.read_csv(base_dir / "steady_states.csv")

    df = settings.merge(
        truth,
        left_on="name",
        right_on="model_name",
        how="inner",
        validate="one_to_one",
    ).copy()

    for col in INPUT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    for col in STATE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    rows = []

    for _, row in df.iterrows():
        model_name = str(row["name"])
        x0 = load_initial_state(model_name, base_dir=base_dir)

        pred = predict_surrogate_steady_state(
            teams_num=float(row["teams_num"]),
            teams_size=float(row["teams_size"]),
            agents_average_initial_opinion=float(row["agents_average_initial_opinion"]),
            technology_success_rate=float(row["technology_success_rate"]),
            initial_state=x0,
        )

        rows.append(
            {
                "model_name": model_name,
                "true_steady_S": float(row["steady_S"]),
                "true_steady_Q": float(row["steady_Q"]),
                "true_steady_L": float(row["steady_L"]),
                "true_steady_B": float(row["steady_B"]),
                "pred_steady_S": float(pred["steady_S"]),
                "pred_steady_Q": float(pred["steady_Q"]),
                "pred_steady_L": float(pred["steady_L"]),
                "pred_steady_B": float(pred["steady_B"]),
                "true_adoption": float(row["steady_Q"] + row["steady_L"]),
                "pred_adoption": float(pred["steady_Q"] + pred["steady_L"]),
            }
        )

    results = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)

    y_true = results[
        ["true_steady_S", "true_steady_Q", "true_steady_L", "true_steady_B"]
    ].to_numpy(dtype=float)

    y_pred = results[
        ["pred_steady_S", "pred_steady_Q", "pred_steady_L", "pred_steady_B"]
    ].to_numpy(dtype=float)

    summary = {
        "comp_rmse": composition_rmse(y_true, y_pred),
        "adoption_rmse": adoption_rmse(y_true, y_pred),
        "mean_kl": mean_kl_divergence(y_true, y_pred),
        "mean_r2": mean_state_r2(y_true, y_pred),
        "max_sum_error": max_abs_sum_error(y_pred),
        "n_models": len(results),
    }
    summary_df = pd.DataFrame([summary])

    # Save CSV outputs
    results_path = VALIDATION_DIR / "steady_state_predictions.csv"
    summary_path = VALIDATION_DIR / "steady_state_validation_summary.csv"

    results.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # Save parity plots
    for state in ["S", "Q", "L", "B"]:
        make_parity_plot(
            results[f"true_steady_{state}"].to_numpy(dtype=float),
            results[f"pred_steady_{state}"].to_numpy(dtype=float),
            title=f"Parity plot: steady_{state} (surrogate ODE vs fitted ODE)",
            out_path=VALIDATION_DIR / f"parity_steady_{state}.png",
        )

    make_parity_plot(
        results["true_adoption"].to_numpy(dtype=float),
        results["pred_adoption"].to_numpy(dtype=float),
        title="Parity plot: steady adoption (Q + L) (surrogate ODE vs fitted ODE)",
        out_path=VALIDATION_DIR / "parity_adoption.png",
    )

    print("\nValidation summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved predictions to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved parity plots to: {VALIDATION_DIR}")


if __name__ == "__main__":
    main()