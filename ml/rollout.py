import numpy as np
from sklearn.ensemble import RandomForestRegressor
from simulation.leapfrog import leapfrog


def rollout(model: RandomForestRegressor,
            v0: float, theta_deg: float, k_over_m: float,
            dt: float = 0.005) -> tuple:
    
    # trayectoria de referencia (Leapfrog)
    xs_true, ys_true, vxs, vys, ts = leapfrog(v0, theta_deg, k_over_m, dt)

    # Rollout con el modelo
    x_pred, y_pred = [xs_true[0]], [ys_true[0]]
    state = [xs_true[0], ys_true[0], vxs[0], vys[0], ts[0], k_over_m]

    for i in range(1, len(xs_true)):
        pred = model.predict([state])[0]
        x_pred.append(pred[0])
        y_pred.append(pred[1])

        if pred[1] < 0:
            break

        # actualizar estado: posicion predicha + velocidades del Leapfrog
        idx = min(i, len(vxs) - 1)
        state = [pred[0], pred[1], vxs[idx], vys[idx], ts[idx], k_over_m]

    return (np.array(x_pred), np.array(y_pred),
            xs_true,          ys_true)


def rollout_error(model: RandomForestRegressor,
                  v0: float, theta_deg: float,
                  k_over_m: float, dt: float = 0.005) -> dict:
    xs_p, ys_p, xs_t, ys_t = rollout(model, v0, theta_deg, k_over_m, dt)
    n = min(len(xs_p), len(xs_t))

    return {
        "mae_x":    np.mean(np.abs(xs_p[:n] - xs_t[:n])),
        "mae_y":    np.mean(np.abs(ys_p[:n] - ys_t[:n])),
        "max_err_x": np.max(np.abs(xs_p[:n] - xs_t[:n])),
        "max_err_y": np.max(np.abs(ys_p[:n] - ys_t[:n])),
    }