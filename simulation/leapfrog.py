
#integrador numerico Leapfrog (Stormer-Verlet) para proyectil con arrastre
import numpy as np
from config import GRAVITY, MAX_STEPS


def acceleration(vx: float, vy: float, k_over_m: float):
    v  = np.sqrt(vx**2 + vy**2)
    ax = -k_over_m * v * vx
    ay = -GRAVITY - k_over_m * v * vy
    return ax, ay


def leapfrog(v0: float, theta_deg: float, k_over_m: float,
             dt: float = 0.005) -> tuple:

    theta = np.radians(theta_deg)
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x, y, t = 0.0, 0.0, 0.0

    xs, ys, vxs, vys, ts = [x], [y], [vx], [vy], [t]

    # medio paso inicial de velocidad (kick)
    ax, ay = acceleration(vx, vy, k_over_m)
    vx_half = vx + ax * dt / 2
    vy_half = vy + ay * dt / 2

    for _ in range(MAX_STEPS):
        x += vx_half * dt
        y += vy_half * dt
        t += dt

        if y < 0:
            break
        ax, ay = acceleration(vx_half, vy_half, k_over_m)

        vx = vx_half + ax * dt / 2
        vy = vy_half + ay * dt / 2

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        ts.append(t)

        vx_half = vx + ax * dt / 2
        vy_half = vy + ay * dt / 2

    return (np.array(xs), np.array(ys),
            np.array(vxs), np.array(vys),
            np.array(ts))