import time
import numpy as np
from numba import jit

N_STEPS = 100000

@jit(nopython=True)
def generate_trajectory(trajectory, n_steps, x0):
    """Generate trajectory using quadratic map with Numba JIT compilation"""
    trajectory[0] = x0
    x_current = x0

    for t in range(1, n_steps + 1):
        x_current = 4.0 * x_current * (1.0 - x_current)
        trajectory[t] = x_current

# Warm up JIT compilation
warmup_array = np.zeros(11)
generate_trajectory(warmup_array, 10, 0.3)

def main():
    # Initialize array outside timer
    trajectory = np.zeros(N_STEPS + 1)
    x0 = 0.3

    # Start timer (after JIT warmup and array allocation)
    start = time.time()

    # Generate trajectory (JIT compiled)
    generate_trajectory(trajectory, N_STEPS, x0)

    # End timer
    elapsed_time = time.time() - start

    # Write to file
    with open('figure3_data_numba.txt', 'w') as fp:
        fp.write('# x_t\n')
        for t in range(N_STEPS + 1):
            fp.write(f'{trajectory[t]:.10f}\n')

    print(f'Execution time: {elapsed_time:.6f} seconds')

if __name__ == '__main__':
    main()
