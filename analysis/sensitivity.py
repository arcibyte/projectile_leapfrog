import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from data.dataset import generate_dataset, FEATURES, TARGETS
from config import DT_VALUES, K_VALUES, N_SMALL, RANDOM_SEED


def _quick_train(df: pd.DataFrame,
                 n_estimators: int = 50) -> tuple:
    X = df[FEATURES].values
    y = df[TARGETS].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=15,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return y_te, y_pred


def sweep_dt(dt_values: list = DT_VALUES,
             N: int = N_SMALL) -> pd.DataFrame:
    rows = []
    for dt in dt_values:
        print(f"  Δt = {dt:.3f} s …", end=" ", flush=True)
        df = generate_dataset(N=N, dt=dt, verbose=False)
        y_te, y_pred = _quick_train(df)
        mae_x = mean_absolute_error(y_te[:, 0], y_pred[:, 0])
        mae_y = mean_absolute_error(y_te[:, 1], y_pred[:, 1])
        rows.append({"dt": dt, "mae_x": mae_x, "mae_y": mae_y})
        print(f"MAE_x={mae_x:.5f}  MAE_y={mae_y:.5f}")

    return pd.DataFrame(rows)


def sweep_k(k_values: list = K_VALUES,
            N: int = N_SMALL) -> pd.DataFrame:
    rows = []
    for k in k_values:
        print(f"  k/m = {k:.3f} …", end=" ", flush=True)
        df = generate_dataset(N=N, k_fixed=k, verbose=False)
        y_te, y_pred = _quick_train(df)
        mae_x = mean_absolute_error(y_te[:, 0], y_pred[:, 0])
        mae_y = mean_absolute_error(y_te[:, 1], y_pred[:, 1])
        rows.append({"k_over_m": k, "mae_x": mae_x, "mae_y": mae_y})
        print(f"MAE_x={mae_x:.5f}  MAE_y={mae_y:.5f}")

    return pd.DataFrame(rows)


def run_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:

    print("sensibilidad Δt")
    df_dt = sweep_dt()

    print("\n ensibilidad k/m")
    df_k  = sweep_k()

    return df_dt, df_k

def analizar_sensibilidad(model=None):
    import matplotlib.pyplot as plt
    from config import FIGURE_PATH, OUTPUT_DIR
    import os

    df_dt, df_k = run_sensitivity()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(df_dt['dt'], df_dt['mae_x'], 'o-', label='MAE X (Alcance)', color='crimson')
    ax1.plot(df_dt['dt'], df_dt['mae_y'], 's-', label='MAE Y (Altura)', color='darkorange')
    ax1.set_title('sensibilidad ante resolucion temporal ($\Delta t$)')
    ax1.set_xlabel('$\Delta t$ (segundos)')
    ax1.set_ylabel('MAE (metros)')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    ax2.plot(df_k['k_over_m'], df_k['mae_x'], 'o-', label='MAE X (Alcance)', color='royalblue')
    ax2.plot(df_k['k_over_m'], df_k['mae_y'], 's-', label='MAE Y (Altura)', color='teal')
    ax2.set_title('sensibilidad ante coeficiente de arrastre ($k/m$)')
    ax2.set_xlabel('coeficiente $k/m$ ($m^{-1}$)')
    ax2.set_ylabel('MAE (metros)')
    ax2.grid(True, linestyle=':')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300)
    plt.close()

    return df_dt, df_k