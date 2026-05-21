import os
import config
from simulation.leapfrog import leapfrog
from data.dataset import generate_dataset, save_dataset, load_dataset
from ml.model import train, save_model
from ml.rollout import rollout_error
from visualization.graphics import generar_graficas


def pipeline_principal():
    print("iniciando Pipeline de Mecanica + ML")

    print("cargando configuracion basica")
    dt = config.DT

    print("ejecutando simulacion de prueba con integrador numerico Leapfrog")
    v0_test    = (config.V0_MIN + config.V0_MAX) / 2
    theta_test = (config.THETA_MIN + config.THETA_MAX) / 2
    k_test     = (config.K_MIN + config.K_MAX) / 2

    xs, ys, vxs, vys, ts = leapfrog(v0_test, theta_test, k_test, dt=dt)
    print(f"trayectoria de prueba completada")
    print(f"alcance maximo simulado: {xs[-1]:.4f} m")
    print(f"tiempo de vuelo simulado: {ts[-1]:.4f} s")

    # dataset
    print("\nverificando dataset")
    if os.path.exists(config.DATASET_PATH):
        print("dataset encontrado, cargando desde CSV")
        dataset = load_dataset()
    else:
        print("dataset no encontrado, generando")
        dataset = generate_dataset()
        save_dataset(dataset)

    print("\n entrenando el modelo predictor")
    modelo_entrenado, X_test, y_test, y_pred = train(dataset)
    save_model(modelo_entrenado)

    # Rollout
    print("\n ejecutando rollout iterativo con el modelo")
    errores = rollout_error(modelo_entrenado, v0_test, theta_test, k_test, dt=dt)
    print("errores acumulados en la prueba:")
    print(f"MAE en X : {errores['mae_x']:.6f} m")
    print(f"MAE en Y : {errores['mae_y']:.6f} m")
    print(f"Máx Err X: {errores['max_err_x']:.6f} m")
    print(f"Máx Err Y: {errores['max_err_y']:.6f} m")


    generar_graficas(modelo_entrenado, X_test, y_test, y_pred)

    print(f"graficas outputs/graficas/")
    print(f"modelo   {config.MODEL_PATH}")
    print(f"dataset  {config.DATASET_PATH}")


if __name__ == "__main__":
    pipeline_principal()
