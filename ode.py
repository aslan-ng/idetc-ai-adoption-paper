"""
Fit continuous-time SQLB transition rates (k values) from simulated state-fraction time series.

This module performs an ODE/CTMC reduction of the agent-based SQLB dynamics by estimating
a continuous-time Markov chain (CTMC) generator matrix Q from the per-step fractions of agents
in each state:

    x(t) = [S(t), Q(t), L(t), B(t)] ,  with  sum(x(t)) = 1

We assume the reduced-order dynamics follow:

    dx/dt = x Q

where:
    - off-diagonal entries Q[i,j] = k_{i->j} are nonnegative transition rates (hazards)
    - diagonal entries are implied by conservation of probability mass:
            Q[i,i] = -sum_{j != i} Q[i,j]
      so each row sums to zero.

Inputs:
    - states/{model_name}.csv
      containing per-step fractions:
          ratio_S, ratio_Q, ratio_L, ratio_B

Allowed transition structure:
    The generator is constrained using ALLOWED_STATE_CHANGES. Disallowed transitions are
    forced to zero; only allowed off-diagonal rates are fit.

Estimation method:
    - Uses the finite-difference approximation:
          (x_{t+1} - x_t)/dt ≈ x_t Q
    - Fits allowed off-diagonal rates via nonnegative least squares (NNLS) implemented
      with projected gradient descent (dependency-free).
    - Enforces generator validity by setting diagonals to negative row sums.
    - Reports fit quality as RMSE between observed derivatives and model-predicted derivatives.

Outputs:
    For each model, saves a single compressed artifact:
        odes/{model_name}.npz
    containing:
        - Q (4x4 generator matrix)
        - dt (time step used in fitting)
        - rmse (fit error)
        - state_order (e.g., ["S","Q","L","B"])
        - transitions, rates (human-readable mapping of k_{i->j})
        - csv_path (source states file)

Batch mode:
    fit_all_models(BASE_DIR) loops over all available models and produces one .npz per model,
    enabling downstream:
        - ODE vs ABM comparison plots
        - final adoption computation from ODE surrogate
        - tradeoff contour / decision-framework analysis
"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, List, Iterable
import numpy as np
import pandas as pd

from utils import BASE_DIR, get_all_model_names


ALLOWED_STATE_CHANGES = {
    "S": ["S", "Q", "L", "B"],
    "Q": ["S", "Q", "L", "B"],
    "L": ["S", "Q", "L", "B"],
    "B": ["S", "Q", "L", "B"],
}
STATE_ORDER = list(ALLOWED_STATE_CHANGES.keys())

@dataclass(frozen=True)
class KFitResult:
    model_name: str
    state_order: List[str]
    dt: float
    Q: np.ndarray                 # generator, shape (4,4), rows sum to 0
    k_dict: Dict[str, float]      # keys like "S->Q"
    residual_rmse: float
    csv_path: Path


def _load_state_fractions(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path)
    required = {"ratio_S", "ratio_Q", "ratio_L", "ratio_B"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path.name}: {sorted(missing)}")

    X = np.column_stack([df["ratio_S"], df["ratio_Q"], df["ratio_L"], df["ratio_B"]]).astype(float)
    X = np.clip(X, 0.0, 1.0)

    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X = X / row_sums
    return X


def _solve_nnls_ls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Small, dependency-free NNLS via projected gradient descent.
    """
    x = np.zeros(A.shape[1], dtype=float)
    AtA = A.T @ A
    Atb = A.T @ b

    L = float(np.max(np.linalg.eigvalsh(AtA))) if A.shape[1] > 0 else 0.0
    if L <= 0:
        return x
    step = 1.0 / L

    for _ in range(8000):
        grad = AtA @ x - Atb
        x_new = np.maximum(x - step * grad, 0.0)
        if np.linalg.norm(x_new - x) < 1e-11:
            x = x_new
            break
        x = x_new
    return x


def _allowed_offdiag_pairs(
    state_order: List[str],
    allowed_state_changes: Dict[str, Iterable[str]],
) -> List[Tuple[int, int]]:
    """
    Returns list of (i,j) for allowed *off-diagonal* transitions i->j.
    Self transitions are not parameters in a CTMC generator (they are implied by row sums).
    """
    idx = {s: k for k, s in enumerate(state_order)}
    pairs: List[Tuple[int, int]] = []
    for si in state_order:
        for sj in allowed_state_changes.get(si, []):
            if sj == si:
                continue
            if sj not in idx:
                raise ValueError(f"Unknown state in allowed_state_changes: {sj}")
            pairs.append((idx[si], idx[sj]))
    # remove duplicates while preserving order
    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def fit_k_from_states(
    base_dir: str | Path,
    model_name: str,
    *,
    dt: float = 1.0,
    allowed_state_changes: Dict[str, Iterable[str]] = ALLOWED_STATE_CHANGES,
) -> KFitResult:
    """
    Estimate CTMC generator rates k_{i->j} from per-step state fractions,
    respecting allowed_state_changes (disallowed rates forced to 0).

    Reads:
        base_dir/states/{model_name}.csv
    """
    base_dir = Path(base_dir)
    csv_path = base_dir / "states" / f"{model_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"State CSV not found: {csv_path}")

    X = _load_state_fractions(csv_path)  # (T,4)
    if X.shape[0] < 3:
        raise ValueError(f"Need at least 3 timesteps; got T={X.shape[0]}")

    # Regression: (X_{t+1}-X_t)/dt = X_t Q
    Y = (X[1:] - X[:-1]) / float(dt)   # (T-1,4)
    X0 = X[:-1]                        # (T-1,4)

    # Only fit allowed off-diagonal pairs
    param_pairs = _allowed_offdiag_pairs(STATE_ORDER, allowed_state_changes)
    p = len(param_pairs)
    if p == 0:
        raise ValueError("No allowed off-diagonal transitions to fit.")

    Tm1 = X0.shape[0]
    A = np.zeros((Tm1 * 4, p), dtype=float)
    b = Y.reshape(-1)

    # Same structure as before:
    # b[t,j] = sum_{i!=j} X0[t,i]*q_{i->j} - X0[t,j]*sum_{m!=j} q_{j->m}
    for t in range(Tm1):
        for j in range(4):
            row = t * 4 + j
            for idx_p, (i, k) in enumerate(param_pairs):
                if k == j:
                    A[row, idx_p] += X0[t, i]
                if i == j:
                    A[row, idx_p] -= X0[t, j]

    q_off = _solve_nnls_ls(A, b)  # nonnegative, shape (p,)

    Q = np.zeros((4, 4), dtype=float)
    for val, (i, j) in zip(q_off, param_pairs):
        Q[i, j] = float(val)

    # Diagonal enforces row-sum=0; disallowed off-diagonals remain 0
    np.fill_diagonal(Q, -np.sum(Q, axis=1))

    # Fit quality
    Y_hat = X0 @ Q
    resid = Y - Y_hat
    rmse = float(np.sqrt(np.mean(resid**2)))

    # Dict of *all* off-diagonals (including zeros for disallowed)
    k_dict: Dict[str, float] = {}
    for i, si in enumerate(STATE_ORDER):
        for j, sj in enumerate(STATE_ORDER):
            if i != j:
                k_dict[f"{si}->{sj}"] = float(Q[i, j])

    return KFitResult(
        model_name=model_name,
        state_order=STATE_ORDER.copy(),
        dt=float(dt),
        Q=Q,
        k_dict=k_dict,
        residual_rmse=rmse,
        csv_path=csv_path,
    )

#res = fit_k_from_states(BASE_DIR, "0_9_8", dt=1.0)
#print(res.k_dict)
#print(res.Q)
#print("RMSE:", res.residual_rmse)

def save_k_result_single_file(base_dir: Path, result: KFitResult) -> Path:
    """
    Save one model's ODE fit results to a single file:

        base_dir/odes/{model_name}.npz
    """
    odes_dir = base_dir / "odes"
    odes_dir.mkdir(parents=True, exist_ok=True)

    out_path = odes_dir / f"{result.model_name}.npz"

    # Stable ordering for transitions
    transitions = []
    rates = []
    for i, si in enumerate(result.state_order):
        for j, sj in enumerate(result.state_order):
            if i != j:
                transitions.append(f"{si}->{sj}")
                rates.append(float(result.Q[i, j]))  # same as k_dict

    np.savez_compressed(
        out_path,
        Q=result.Q,
        dt=np.array([result.dt], dtype=float),
        rmse=np.array([result.residual_rmse], dtype=float),
        state_order=np.array(result.state_order, dtype="U"),
        transitions=np.array(transitions, dtype="U"),
        rates=np.array(rates, dtype=float),
        csv_path=np.array([str(result.csv_path)], dtype="U"),
    )
    return out_path

def _fit_one_model_job(args: tuple[str, str, float]) -> tuple[str, str]:
    """
    Worker job. args = (base_dir_str, model_name, dt)
    Returns: (model_name, status)
    """
    base_dir_str, model_name, dt = args
    base_dir = Path(base_dir_str)

    odes_dir = base_dir / "odes"
    odes_dir.mkdir(parents=True, exist_ok=True)

    out_path = odes_dir / f"{model_name}.npz"

    # ---- Skip if already computed ----
    if out_path.exists():
        return model_name, "skipped (exists)"

    # ---- Optional: lock to avoid double-work if multiple processes/scripts run ----
    lock_path = out_path.with_suffix(".npz.lock")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return model_name, "skipped (locked)"

    try:
        # check again after acquiring lock
        if out_path.exists():
            return model_name, "skipped (exists)"

        res = fit_k_from_states(base_dir, model_name, dt=dt)
        save_k_result_single_file(base_dir, res)
        return model_name, f"saved (RMSE={res.residual_rmse:.6f})"

    except Exception as e:
        return model_name, f"failed ({e})"

    finally:
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            pass


def fit_all_models_parallel(base_dir: Path, dt: float = 1.0) -> None:
    base_dir = Path(base_dir)

    # Ensure output dir exists up-front (also created in workers)
    (base_dir / "odes").mkdir(parents=True, exist_ok=True)

    model_names = get_all_model_names()
    total = len(model_names)
    if total == 0:
        print("No models found.")
        return

    max_workers = max(1, (os.cpu_count() or 2) - 2)

    jobs = [(str(base_dir), name, float(dt)) for name in model_names]

    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_fit_one_model_job, job) for job in jobs]
        for fut in as_completed(futures):
            name, status = fut.result()
            done += 1
            print(f"[{done}/{total}] {name}: {status}")


if __name__ == "__main__":
    fit_all_models_parallel(BASE_DIR, dt=1.0)