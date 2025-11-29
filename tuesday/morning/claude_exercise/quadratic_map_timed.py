import numpy as np
from numba import jit
import time

@jit(nopython=True)
def quadratic_map(x):
    """Quadratic map function: g(x) = 4x(1-x)"""
    return 4.0 * x * (1.0 - x)

@jit(nopython=True)
def generate_function_values(num_points=1000):
    """Generate function values for plotting"""
    x_vals = np.linspace(0.0, 1.0, num_points + 1)
    g_vals = np.empty(num_points + 1)

    for i in range(num_points + 1):
        g_vals[i] = quadratic_map(x_vals[i])

    return x_vals, g_vals

@jit(nopython=True)
def generate_trajectory_values(x0, n):
    """Generate trajectory values"""
    trajectory = np.empty(n + 1)
    x = x0

    for t in range(n + 1):
        trajectory[t] = x
        x = quadratic_map(x)

    return trajectory

@jit(nopython=True)
def generate_histogram_values(x0, n):
    """Generate long trajectory for histogram"""
    histogram_data = np.empty(n + 1)
    x = x0

    for t in range(n + 1):
        histogram_data[t] = x
        x = quadratic_map(x)

    return histogram_data

@jit(nopython=True)
def generate_all_data():
    """Generate all data - this is what we time"""
    x_vals, g_vals = generate_function_values(1000)
    trajectory = generate_trajectory_values(0.3, 150)
    histogram_data = generate_histogram_values(0.3, 100000)
    return x_vals, g_vals, trajectory, histogram_data

if __name__ == '__main__':
    # Warm up JIT compilation (don't time this)
    print('Warming up JIT compilation...')
    _ = generate_all_data()

    # Now time the actual data generation
    print('Running timed benchmark...')

    start_time = time.perf_counter()
    x_vals, g_vals, trajectory, histogram_data = generate_all_data()
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    print(f'Numba data generation time: {elapsed:.6f} seconds')
