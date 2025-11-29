import numpy as np
from numba import jit

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

def save_function_data():
    """Save function data to CSV"""
    x_vals, g_vals = generate_function_values(1000)

    with open('quadratic_function_numba.csv', 'w') as f:
        f.write('x,g_x\n')
        for i in range(len(x_vals)):
            f.write(f'{x_vals[i]:.15f},{g_vals[i]:.15f}\n')

def save_trajectory_data(x0, n, filename):
    """Save trajectory data to CSV"""
    trajectory = generate_trajectory_values(x0, n)

    with open(filename, 'w') as f:
        f.write('t,x\n')
        for t in range(len(trajectory)):
            f.write(f'{t},{trajectory[t]:.15f}\n')

def save_histogram_data(x0, n, filename):
    """Save histogram data to CSV"""
    histogram_data = generate_histogram_values(x0, n)

    with open(filename, 'w') as f:
        f.write('x\n')
        for i in range(len(histogram_data)):
            f.write(f'{histogram_data[i]:.15f}\n')

def main():
    """Main function to generate all data files"""
    print('Generating quadratic map data with Numba...')

    # Generate function data for Figure 1
    save_function_data()
    print('Generated quadratic_function_numba.csv')

    # Generate trajectory for Figure 2 (x0 = 0.3, n = 150)
    save_trajectory_data(0.3, 150, 'trajectory_numba.csv')
    print('Generated trajectory_numba.csv')

    # Generate long trajectory for Figure 3 (x0 = 0.3, n = 100000)
    save_histogram_data(0.3, 100000, 'histogram_data_numba.csv')
    print('Generated histogram_data_numba.csv')

    print('All Numba data files generated successfully!')

if __name__ == '__main__':
    # Warm up JIT compilation
    _ = generate_function_values(10)
    _ = generate_trajectory_values(0.3, 10)
    _ = generate_histogram_values(0.3, 100)

    # Generate actual data
    main()
