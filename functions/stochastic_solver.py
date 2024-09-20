# functions/solve_system.py
from numba import njit
import numpy as np

@njit
def solve_system(beta, epsilon, V, time_step):
    V0, V1, V4 = 2, 2, 2.5
    kf, k = 1, 10
    VM2, k2 = 6, 0.1
    VM3, m = 20, 4
    kx, ky, kz = 0.3, 0.2, 0.1
    VM5, k5 = 30, 1
    kd, p, n = 0.6, 1, 2
    tmax = 200
    Nn = round(tmax / time_step)

    x, y, z = np.ones(Nn + 1), np.ones(Nn + 1), np.ones(Nn + 1)
    for i in range(Nn):
        # For efficient memory management
        xi = [np.random.normal(0, np.sqrt(time_step)) for _ in range(12)]
        
        V2 = VM2 * x[i]**2 / (k2**2 + x[i]**2)
        V3 = VM3 * x[i]**m / (kx**m + x[i]**m) * y[i]**2 / (ky**2 + y[i]**2) * z[i]**4 / (kz**4 + z[i]**4)
        V5 = VM5 * z[i]**p / (k5**p + z[i]**p) * x[i]**n / (kd**n + x[i]**n)
        x[i + 1] = (x[i] + (V0 + V1 * beta - k * x[i] - V2 + kf * y[i] + V3) * time_step +
                    1 / np.sqrt(V) * (np.sqrt(V0) * xi[0] + np.sqrt(V1 * beta) * xi[1] - np.sqrt(V2) * xi[2] + 
                                      np.sqrt(V3) * xi[3] + np.sqrt(kf * y[i]) * xi[4] - np.sqrt(k) * xi[5]))
        y[i + 1] = (y[i] + (V2 - kf * y[i] - V3) * time_step +
                    1 / np.sqrt(V) * (np.sqrt(V2) * xi[6] - np.sqrt(V3) * xi[7] - np.sqrt(kf * y[i]) * xi[8]))
        z[i + 1] = (z[i] + (V4 * beta - epsilon * z[i] - V5) * time_step +
                    1 / np.sqrt(V) * (np.sqrt(V4 * beta) * xi[9] - np.sqrt(V5) * xi[10] - np.sqrt(epsilon * z[i]) * xi[11]))
    return x, y, z

@njit
def solve_system_srk(beta, epsilon, dt, tmax, V0, V1, VM2, k2, VM3, kx, ky, kz, VM5, k5, kd, V4, k, kf, m, n, p, V, seed):
    Nn = round(tmax / dt)
    np.random.seed(seed)
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

    def g(x, y, z, beta):
        V2 = VM2 * x**2 / (k2**2 + x**2)
        V3 = VM3 * x**m / (kx**m + x**m) * y**2 / (ky**2 + y**2) * z**4 / (kz**4 + z**4)
        V5 = VM5 * z**p / (k5**p + z**p) * x**n / (kd**n + x**n)
        gx = np.sqrt(V0 + V1 * beta + V2 + V3 + kf * y + k * x) / np.sqrt(V)
        gy = np.sqrt(V2 + V3 + kf * y) / np.sqrt(V)
        gz = np.sqrt(V4 * beta + V5 + epsilon * z) / np.sqrt(V)
        return gx, gy, gz

    for i in range(Nn):
        # Generate new random numbers at each step
        dW = np.array([np.random.normal(0, np.sqrt(dt)) for _ in range(12)])
        
        fx, fy, fz = f(x[i], y[i], z[i], beta)
        gx, gy, gz = g(x[i], y[i], z[i], beta)
        
        x[i + 1] = x[i] + fx * dt + gx * np.sum(dW[:6])
        y[i + 1] = y[i] + fy * dt + gy * np.sum(dW[6:9])
        z[i + 1] = z[i] + fz * dt + gz * np.sum(dW[9:])

    return x, y, z


@njit
def solve_system_seuler(beta, epsilon, dt, tmax, V0, V1, VM2, k2, VM3, kx, ky, kz, VM5, k5, kd, V4, k, kf, m, n, p, V, seed):
    Nn = round(tmax / dt)
    np.random.seed(seed)
    # Initial conditions
    x = np.ones(Nn + 1)
    y = np.ones(Nn + 1)
    z = np.ones(Nn + 1)

    # Euler discretization
    for i in range(Nn):
        # Generate new random numbers at each step
        dW = np.array([np.random.normal(0, np.sqrt(dt)) for _ in range(12)])
        
        V2 = VM2 * x[i]**2 / (k2**2 + x[i]**2)
        V3 = VM3 * x[i]**m / (kx**m + x[i]**m) * y[i]**2 / (ky**2 + y[i]**2) * z[i]**4 / (kz**4 + z[i]**4)
        V5 = VM5 * z[i]**p / (k5**p + z[i]**p) * x[i]**n / (kd**n + x[i]**n)

        x[i + 1] = (x[i] + (V0 + V1 * beta - k * x[i] - V2 + kf * y[i] + V3) * dt +
                    1 / np.sqrt(V) * (np.sqrt(V0) * dW[0] + np.sqrt(V1 * beta) * dW[1] - np.sqrt(V2) * dW[2] + 
                                      np.sqrt(V3) * dW[3] + np.sqrt(kf * y[i]) * dW[4] - np.sqrt(k) * dW[5]))
        y[i + 1] = (y[i] + (V2 - kf * y[i] - V3) * dt +
                    1 / np.sqrt(V) * (np.sqrt(V2) * dW[6] - np.sqrt(V3) * dW[7] - np.sqrt(kf * y[i]) * dW[8]))
        z[i + 1] = (z[i] + (V4 * beta - epsilon * z[i] - V5) * dt +
                    1 / np.sqrt(V) * (np.sqrt(V4 * beta) * dW[9] - np.sqrt(V5) * dW[10] - np.sqrt(epsilon * z[i]) * dW[11]))
    return x, y, z


@njit
def solve_system_heun(beta, epsilon, dt, tmax, V0, V1, VM2, k2, VM3, kx, ky, kz, VM5, k5, kd, V4, k, kf, m, n, p, V):
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

    def g(x, y, z, beta):
        V2 = VM2 * x**2 / (k2**2 + x**2)
        V3 = VM3 * x**m / (kx**m + x**m) * y**2 / (ky**2 + y**2) * z**4 / (kz**4 + z**4)
        V5 = VM5 * z**p / (k5**p + z**p) * x**n / (kd**n + x**n)
        gx = np.sqrt(V0 + V1 * beta + V2 + V3 + kf * y + k * x) / np.sqrt(V)
        gy = np.sqrt(V2 + V3 + kf * y) / np.sqrt(V)
        gz = np.sqrt(V4 * beta + V5 + epsilon * z) / np.sqrt(V)
        return gx, gy, gz

    for i in range(Nn):
        # Generate new random numbers at each step
        dW = np.array([np.random.normal(0, np.sqrt(dt)) for _ in range(12)])
        
        fx, fy, fz = f(x[i], y[i], z[i], beta)
        gx, gy, gz = g(x[i], y[i], z[i], beta)

        x_pred = x[i] + fx * dt + gx * np.sum(dW[:6])
        y_pred = y[i] + fy * dt + gy * np.sum(dW[6:9])
        z_pred = z[i] + fz * dt + gz * np.sum(dW[9:])

        fx_pred, fy_pred, fz_pred = f(x_pred, y_pred, z_pred, beta)
        gx_pred, gy_pred, gz_pred = g(x_pred, y_pred, z_pred, beta)

        x[i + 1] = x[i] + 0.5 * (fx + fx_pred) * dt + 0.5 * (gx + gx_pred) * np.sum(dW[:6])
        y[i + 1] = y[i] + 0.5 * (fy + fy_pred) * dt + 0.5 * (gy + gy_pred) * np.sum(dW[6:9])
        z[i + 1] = z[i] + 0.5 * (fz + fz_pred) * dt + 0.5 * (gz + gz_pred) * np.sum(dW[9:])

    return x, y, z


# @njit
# def solve_system_srk(beta, epsilon, dt, tmax, V0, V1, VM2, k2, VM3, kx, ky, kz, VM5, k5, kd, V4, k, kf, m, n, p, V, seed):
#     Nn = round(tmax / dt)
#     np.random.seed(seed)
#     # Random variates
#     xi = [np.random.normal(0, np.sqrt(dt), Nn) for _ in range(12)]
#     # Initial conditions
#     x = np.ones(Nn + 1)
#     y = np.ones(Nn + 1)
#     z = np.ones(Nn + 1)

#     def f(x, y, z, beta):
#         V2 = VM2 * x**2 / (k2**2 + x**2)
#         V3 = VM3 * x**m / (kx**m + x**m) * y**2 / (ky**2 + y**2) * z**4 / (kz**4 + z**4)
#         V5 = VM5 * z**p / (k5**p + z**p) * x**n / (kd**n + x**n)
#         fx = V0 + V1 * beta - k * x - V2 + kf * y + V3
#         fy = V2 - kf * y - V3
#         fz = V4 * beta - epsilon * z - V5
#         return fx, fy, fz

#     def g(x, y, z, beta):
#         V2 = VM2 * x**2 / (k2**2 + x**2)
#         V3 = VM3 * x**m / (kx**m + x**m) * y**2 / (ky**2 + y**2) * z**4 / (kz**4 + z**4)
#         V5 = VM5 * z**p / (k5**p + z**p) * x**n / (kd**n + x**n)
#         gx = np.sqrt(V0 + V1 * beta + V2 + V3 + kf * y + k * x) / np.sqrt(V)
#         gy = np.sqrt(V2 + V3 + kf * y) / np.sqrt(V)
#         gz = np.sqrt(V4 * beta + V5 + epsilon * z) / np.sqrt(V)
#         return gx, gy, gz

#     for i in range(Nn):
#         fx, fy, fz = f(x[i], y[i], z[i], beta)
#         gx, gy, gz = g(x[i], y[i], z[i], beta)
        
#         dW = np.array([xi[j][i] for j in range(12)])
        
#         x[i + 1] = x[i] + fx * dt + gx * np.sum(dW[:6])
#         y[i + 1] = y[i] + fy * dt + gy * np.sum(dW[6:9])
#         z[i + 1] = z[i] + fz * dt + gz * np.sum(dW[9:])

#     return x, y, z


# @njit
# def solve_system_seuler(beta, epsilon, dt, tmax, V0, V1, VM2, k2, VM3, kx, ky, kz, VM5, k5, kd, V4, k, kf, m, n, p, V, seed):
#     Nn = round(tmax / dt)
#     np.random.seed(seed)
#     # Random variates
#     xi = [np.random.normal(0, np.sqrt(dt), Nn) for _ in range(12)]
#     # Initial conditions
#     x = np.ones(Nn + 1)
#     y = np.ones(Nn + 1)
#     z = np.ones(Nn + 1)
#     # Euler discretization
#     for i in range(Nn):
#         V2 = VM2 * x[i]**2 / (k2**2 + x[i]**2)
#         V3 = VM3 * x[i]**m / (kx**m + x[i]**m) * y[i]**2 / (ky**2 + y[i]**2) * z[i]**4 / (kz**4 + z[i]**4)
#         V5 = VM5 * z[i]**p / (k5**p + z[i]**p) * x[i]**n / (kd**n + x[i]**n)
#         x[i + 1] = (x[i] + (V0 + V1 * beta - k * x[i] - V2 + kf * y[i] + V3) * dt + 1 / np.sqrt(V) * (np.sqrt(V0) * xi[0][i] + np.sqrt(V1 * beta) * xi[1][i] - np.sqrt(V2) * xi[2][i] + np.sqrt(V3) * xi[3][i] + np.sqrt(kf * y[i]) * xi[4][i] - np.sqrt(k) * xi[5][i]))
#         y[i + 1] = (y[i] + (V2 - kf * y[i] - V3) * dt + 1 / np.sqrt(V) * (np.sqrt(V2) * xi[6][i] - np.sqrt(V3) * xi[7][i] - np.sqrt(kf * y[i]) * xi[8][i]))
#         z[i + 1] = (z[i] + (V4 * beta - epsilon * z[i] - V5) * dt + 1 / np.sqrt(V) * (np.sqrt(V4 * beta) * xi[9][i] - np.sqrt(V5) * xi[10][i] - np.sqrt(epsilon * z[i]) * xi[11][i]))
#     return x, y, z

# @njit
# def solve_system_heun(beta, epsilon, dt, tmax, V0, V1, VM2, k2, VM3, kx, ky, kz, VM5, k5, kd, V4, k, kf, m, n, p, V):
#     Nn = round(tmax / dt)

#     # Random variates
#     xi = [np.random.normal(0, np.sqrt(dt), Nn) for _ in range(12)]
#     # Initial conditions
#     x = np.ones(Nn + 1)
#     y = np.ones(Nn + 1)
#     z = np.ones(Nn + 1)

#     def f(x, y, z, beta):
#         V2 = VM2 * x**2 / (k2**2 + x**2)
#         V3 = VM3 * x**m / (kx**m + x**m) * y**2 / (ky**2 + y**2) * z**4 / (kz**4 + z**4)
#         V5 = VM5 * z**p / (k5**p + z**p) * x**n / (kd**n + x**n)
#         fx = V0 + V1 * beta - k * x - V2 + kf * y + V3
#         fy = V2 - kf * y - V3
#         fz = V4 * beta - epsilon * z - V5
#         return fx, fy, fz

#     def g(x, y, z, beta):
#         V2 = VM2 * x**2 / (k2**2 + x**2)
#         V3 = VM3 * x**m / (kx**m + x**m) * y**2 / (ky**2 + y**2) * z**4 / (kz**4 + z**4)
#         V5 = VM5 * z**p / (k5**p + z**p) * x**n / (kd**n + x**n)
#         gx = np.sqrt(V0 + V1 * beta + V2 + V3 + kf * y + k * x) / np.sqrt(V)
#         gy = np.sqrt(V2 + V3 + kf * y) / np.sqrt(V)
#         gz = np.sqrt(V4 * beta + V5 + epsilon * z) / np.sqrt(V)
#         return gx, gy, gz

#     for i in range(Nn):
#         fx, fy, fz = f(x[i], y[i], z[i], beta)
#         gx, gy, gz = g(x[i], y[i], z[i], beta)
        
#         dW = np.array([xi[j][i] for j in range(12)])
        
#         x_pred = x[i] + fx * dt + gx * np.sum(dW[:6])
#         y_pred = y[i] + fy * dt + gy * np.sum(dW[6:9])
#         z_pred = z[i] + fz * dt + gz * np.sum(dW[9:])
        
#         fx_pred, fy_pred, fz_pred = f(x_pred, y_pred, z_pred, beta)
#         gx_pred, gy_pred, gz_pred = g(x_pred, y_pred, z_pred, beta)
        
#         x[i + 1] = x[i] + 0.5 * (fx + fx_pred) * dt + 0.5 * (gx + gx_pred) * np.sum(dW[:6])
#         y[i + 1] = y[i] + 0.5 * (fy + fy_pred) * dt + 0.5 * (gy + gy_pred) * np.sum(dW[6:9])
#         z[i + 1] = z[i] + 0.5 * (fz + fz_pred) * dt + 0.5 * (gz + gz_pred) * np.sum(dW[9:])

#     return x, y, z