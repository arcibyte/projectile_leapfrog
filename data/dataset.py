import numpy as np
import pandas as pd
from simulation.leapfrog import leapfrog
from config import (
    V0_MIN, V0_MAX,
    THETA_MIN, THETA_MAX,
    K_MIN, K_MAX,
    DT, RANDOM_SEED,
    DATASET_PATH,
)


FEATURES = ["x_n", "y_n", "vx_n", "vy_n", "t_n", "k_over_m"]
TARGETS  = ["x_next", "y_next"]


def generate_dataset(N: int = 10_000, dt: float = DT,
                     k_fixed: float | None = None,
                     verbose: bool = True) -> pd.DataFrame:
    
    rng = np.random.default_rng(RANDOM_SEED)

    v0s    = rng.uniform(V0_MIN,    V0_MAX,    N)
    thetas = rng.uniform(THETA_MIN, THETA_MAX, N)
    ks     = (np.full(N, k_fixed) if k_fixed is not None
              else rng.uniform(K_MIN, K_MAX, N))

    records = []

    for i in range(N):
        xs, ys, vxs, vys, ts = leapfrog(v0s[i], thetas[i], ks[i], dt)

        if len(xs) < 3:
            continue

        for n in range(len(xs) - 1):
            records.append({
                "x_n":      xs[n],
                "y_n":      ys[n],
                "vx_n":     vxs[n],
                "vy_n":     vys[n],
                "t_n":      ts[n],
                "k_over_m": ks[i],
                "x_next":   xs[n + 1],
                "y_next":   ys[n + 1],
            })

        if verbose and (i + 1) % 1000 == 0:
            print(f"trayectorias: {i+1}/{N}")

    df = pd.DataFrame(records)

    if verbose:
        print(f"\n dataset generado: {len(df):,} muestras de {N} trayectorias")

    return df


def save_dataset(df: pd.DataFrame, path: str = DATASET_PATH) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"dataset guardado en: {path}  {df.shape}")


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"dataset cargado en:{path}  {df.shape}")
    return df