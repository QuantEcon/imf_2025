import numpy as np
import time

def g(x):
    """The quadratic map g(x) = 4x(1-x)"""
    return 4 * x * (1 - x)

def generate_time_series(x0, n):
    """Generate n observations from the quadratic map"""
    x = np.zeros(n)
    x[0] = x0
    for t in range(n - 1):
        x[t + 1] = g(x[t])
    return x

if __name__ == '__main__':
    x0 = 0.3
    n = 100001

    # Warm-up run
    _ = generate_time_series(x0, 1000)

    # Timed run
    start_time = time.time()
    x = generate_time_series(x0, n)
    end_time = time.time()

    elapsed = end_time - start_time
    print(f"Python: Generated {n} observations in {elapsed:.6f} seconds")
    print(f"Python: {n/elapsed:.0f} iterations per second")
