import time
import numpy as np
from numba import jit

N_STEPS = 100000

@jit(nopython=True)
def generate_trajectory(n_steps, x0):
    """Generate trajectory using quadratic map with Numba JIT compilation"""
    trajectory = np.zeros(n_steps + 1)
    trajectory[0] = x0
    x_current = x0

    for t in range(1, n_steps + 1):
        x_current = 4.0 * x_current * (1.0 - x_current)
        trajectory[t] = x_current

    return trajectory

def main():
    # Start timer
    start = time.time()

    # Initial condition
    x0 = 0.3

    # Generate trajectory (JIT compiled)
    trajectory = generate_trajectory(N_STEPS, x0)

    # Write to file
    with open('figure3_data_numba.txt', 'w') as fp:
        fp.write('# x_t\n')
        for t in range(N_STEPS + 1):
            fp.write(f'{trajectory[t]:.10f}\n')

    # End timer
    elapsed_time = time.time() - start

    print(f'Figure 3 data written to figure3_data_numba.txt')
    print(f'Total observations: {N_STEPS + 1}')
    print(f'Execution time: {elapsed_time:.6f} seconds')

if __name__ == '__main__':
    main()
