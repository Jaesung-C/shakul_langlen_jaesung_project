from numba import njit
import numpy as np

@njit
def solve_system_euler(beta, epsilon, dt, tmax, V0, V1, VM2, k2, VM3, kx, ky, kz, VM5, k5, kd, V4, k, kf, m, n, p):
    Nn = round(tmax / dt)
    
    # Initial conditions
    x = np.ones(Nn + 1)
    y = np.ones(Nn + 1)
    z = np.ones(Nn + 1)
    
    # Euler discretization
    for i in range(Nn):
        V2 = VM2 * x[i]**2 / (k2**2 + x[i]**2)
        V3 = VM3 * x[i]**m / (kx**m + x[i]**m) * y[i]**2 / (ky**2 + y[i]**2) * z[i]**4 / (kz**4 + z[i]**4)
        V5 = VM5 * z[i]**p / (k5**p + z[i]**p) * x[i]**n / (kd**n + x[i]**n)
        
        x[i + 1] = x[i] + (V0 + V1 * beta - k * x[i] - V2 + kf * y[i] + V3) * dt
        y[i + 1] = y[i] + (V2 - kf * y[i] - V3) * dt
        z[i + 1] = z[i] + (V4 * beta - epsilon * z[i] - V5) * dt
        
    return x, y, z

@njit
def solve_system_rk4(beta, epsilon, dt, tmax, V0, V1, VM2, k2, VM3, kx, ky, kz, VM5, k5, kd, V4, k, kf, m, n, p):
    Nn = round(tmax / dt)
    
    # Initial conditions
    x = np.ones(Nn + 1)
    y = np.ones(Nn + 1)
    z = np.ones(Nn + 1)

    def f(x, y, z, beta):
        V2 = VM2 * x**2 / (k2**2 + x**2)
        V3 = VM3 * x**m / (kx**m + x**m) * y**2 / (ky**2 + y**2) * z**4 / (kz**4 + z**4)
        V5 = VM5 * z**p / (k5**p + z**p) * x**n / (kd**n + x**n)
        fx = V0 + V1 * beta - k * x - V2 + kf * y + V3
        fy = V2 - kf * y - V3
        fz = V4 * beta - epsilon * z - V5
        return fx, fy, fz

    for i in range(Nn):
        # Runge-Kutta 4th order method
        kx1, ky1, kz1 = f(x[i], y[i], z[i], beta)
        kx2, ky2, kz2 = f(x[i] + 0.5 * kx1 * dt, y[i] + 0.5 * ky1 * dt, z[i] + 0.5 * kz1 * dt, beta)
        kx3, ky3, kz3 = f(x[i] + 0.5 * kx2 * dt, y[i] + 0.5 * ky2 * dt, z[i] + 0.5 * kz2 * dt, beta)
        kx4, ky4, kz4 = f(x[i] + kx3 * dt, y[i] + ky3 * dt, z[i] + kz3 * dt, beta)
        
        x[i + 1] = x[i] + (kx1 + 2 * kx2 + 2 * kx3 + kx4) * dt / 6
        y[i + 1] = y[i] + (ky1 + 2 * ky2 + 2 * ky3 + ky4) * dt / 6
        z[i + 1] = z[i] + (kz1 + 2 * kz2 + 2 * kz3 + kz4) * dt / 6
        
    return x, y, z