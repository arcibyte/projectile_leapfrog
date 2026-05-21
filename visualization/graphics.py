import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import r2_score

from simulation.leapfrog import leapfrog
from ml.rollout import rollout
from data.dataset import FEATURES

C1, C2, C3 = "#1f77b4", "#ff7f0e", "#2ca02c"
OUTDIR = "outputs/graficas"


def _guardar(nombre: str) -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, nombre)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ {nombre}")


def grafica_trayectorias() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    casos = [
        (20, 45, 0.005, "v₀=20, θ=45°, k/m=0.005"),
        (15, 30, 0.02,  "v₀=15, θ=30°, k/m=0.02"),
        (25, 60, 0.04,  "v₀=25, θ=60°, k/m=0.04"),
    ]
    for (v0, th, k, lb), color in zip(casos, [C1, C2, C3]):
        xs, ys, *_ = leapfrog(v0, th, k)
        ax.plot(xs, ys, color=color, lw=2, label=lb)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("A · Trayectorias Leapfrog (ejemplos)")
    ax.legend(fontsize=9)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    _guardar("A_trayectorias.png")


def grafica_importancia(modelo) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    imp = modelo.feature_importances_
    idx = np.argsort(imp)[::-1]
    ax.barh([FEATURES[i] for i in idx], imp[idx],
            color=C1, edgecolor="white")
    ax.set_title("B · Importancia de características")
    ax.set_xlabel("Importancia (Gini)")
    ax.invert_yaxis()
    _guardar("B_importancia.png")


def grafica_pred_vs_real(y_test, y_pred) -> None:
    sample = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)

    # C — x
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test[sample, 0], y_pred[sample, 0],
               alpha=0.3, s=4, color=C1)
    lims = [y_test[:, 0].min(), y_test[:, 0].max()]
    ax.plot(lims, lims, "r--", lw=1.5)
    r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
    ax.set_xlabel("Real x_{n+1} (m)"); ax.set_ylabel("Pred x_{n+1} (m)")
    ax.set_title(f"C · Pred vs Real — x  (R²={r2_x:.5f})")
    _guardar("C_pred_vs_real_x.png")

    # D — y
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test[sample, 1], y_pred[sample, 1],
               alpha=0.3, s=4, color=C2)
    lims = [y_test[:, 1].min(), y_test[:, 1].max()]
    ax.plot(lims, lims, "r--", lw=1.5)
    r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
    ax.set_xlabel("Real y_{n+1} (m)"); ax.set_ylabel("Pred y_{n+1} (m)")
    ax.set_title(f"D · Pred vs Real — y  (R²={r2_y:.5f})")
    _guardar("D_pred_vs_real_y.png")


def grafica_residuos(y_test, y_pred) -> None:
    """E · Distribucion de residuos."""
    sample = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(y_test[sample, 0] - y_pred[sample, 0], bins=60,
            alpha=0.6, color=C1, label="Residuo x", density=True)
    ax.hist(y_test[sample, 1] - y_pred[sample, 1], bins=60,
            alpha=0.6, color=C2, label="Residuo y", density=True)
    ax.set_xlabel("Error (m)"); ax.set_ylabel("Densidad")
    ax.set_title("E · Distribución de residuos")
    ax.legend(fontsize=9)
    _guardar("E_residuos.png")


def grafica_sensibilidad(df_dt, df_k) -> None:
    # F — Δt
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df_dt["dt"], df_dt["mae_x"], "o-", color=C1, label="MAE x")
    ax.plot(df_dt["dt"], df_dt["mae_y"], "s-", color=C2, label="MAE y")
    ax.set_xlabel("Δt (s)"); ax.set_ylabel("MAE (m)")
    ax.set_title("F · Sensibilidad — Δt")
    ax.set_xscale("log"); ax.legend()
    _guardar("F_sensibilidad_dt.png")

    # G — k/m
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(df_k["k_over_m"], df_k["mae_x"], "o-", color=C1, label="MAE x")
    ax.plot(df_k["k_over_m"], df_k["mae_y"], "s-", color=C2, label="MAE y")
    ax.set_xlabel("k/m (m⁻¹)"); ax.set_ylabel("MAE (m)")
    ax.set_title("G · Sensibilidad — k/m")
    ax.legend()
    _guardar("G_sensibilidad_k.png")


def grafica_rollout(modelo) -> None:
    xs_p, ys_p, xs_t, ys_t = rollout(modelo, 18.0, 40.0, 0.015)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(xs_t, ys_t, color=C1, lw=2, label="Leapfrog (real)")
    ax.plot(xs_p, ys_p, "--", color=C3, lw=2, label="RF Rollout (pred)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("H · Rollout predicho vs simulado\n(v₀=18, θ=40°, k/m=0.015)")
    ax.legend(fontsize=9); ax.set_ylim(bottom=0)
    _guardar("H_rollout.png")


def generar_graficas(modelo, X_test, y_test, y_pred,
                     df_dt=None, df_k=None) -> None:
    
    print("\n[VIZ] generando graficas individuales")
    grafica_trayectorias()
    grafica_importancia(modelo)
    grafica_pred_vs_real(y_test, y_pred)
    grafica_residuos(y_test, y_pred)
    grafica_rollout(modelo)

    if df_dt is not None and df_k is not None:
        grafica_sensibilidad(df_dt, df_k)
    else:
        print("  — F y G omitidas (no hay datos de sensibilidad)")
