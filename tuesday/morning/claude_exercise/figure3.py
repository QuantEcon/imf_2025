import time

N_STEPS = 100000

def main():
    # Initial condition
    x0 = 0.3
    trajectory = [0.0] * (N_STEPS + 1)
    trajectory[0] = x0
    x_current = x0

    # Start timer
    start = time.time()

    # Generate long time series using quadratic map g(x) = 4x(1-x)
    for t in range(1, N_STEPS + 1):
        x_current = 4.0 * x_current * (1.0 - x_current)
        trajectory[t] = x_current

    # End timer
    elapsed_time = time.time() - start

    # Write to file
    with open('figure3_data_py.txt', 'w') as fp:
        fp.write('# x_t\n')
        for t in range(N_STEPS + 1):
            fp.write(f'{trajectory[t]:.10f}\n')

    print(f'Figure 3 data written to figure3_data_py.txt')
    print(f'Total observations: {N_STEPS + 1}')
    print(f'Execution time: {elapsed_time:.6f} seconds')

if __name__ == '__main__':
    main()
