import numpy as np
import time
from numba import jit

# Original Python version
def g_python(x):
    """The quadratic map g(x) = 4x(1-x)"""
    return 4 * x * (1 - x)

def generate_time_series_python(x0, n):
    """Generate n observations from the quadratic map"""
    x = np.zeros(n)
    x[0] = x0
    for t in range(n - 1):
        x[t + 1] = g_python(x[t])
    return x

# Numba JIT-compiled version
@jit(nopython=True)
def g_numba(x):
    """The quadratic map g(x) = 4x(1-x)"""
    return 4 * x * (1 - x)

@jit(nopython=True)
def generate_time_series_numba(x0, n):
    """Generate n observations from the quadratic map"""
    x = np.zeros(n)
    x[0] = x0
    for t in range(n - 1):
        x[t + 1] = g_numba(x[t])
    return x

if __name__ == '__main__':
    x0 = 0.3
    n = 100001

    print("="*60)
    print("Python Performance Comparison: Standard vs Numba")
    print("="*60)
    print()

    # Benchmark standard Python
    print("Standard Python:")
    # Warm-up
    _ = generate_time_series_python(x0, 1000)

    times_python = []
    for i in range(5):
        start_time = time.time()
        x = generate_time_series_python(x0, n)
        end_time = time.time()
        elapsed = end_time - start_time
        times_python.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.3f} ms")

    avg_python = np.mean(times_python) * 1000
    print(f"  Average: {avg_python:.3f} ms")
    print()

    # Benchmark Numba (with compilation time shown separately)
    print("Numba JIT-compiled:")
    print("  Compiling... (first run includes compilation)")

    # First run includes compilation time
    start_compile = time.time()
    _ = generate_time_series_numba(x0, n)
    end_compile = time.time()
    compile_time = end_compile - start_compile
    print(f"  First run (with compilation): {compile_time*1000:.3f} ms")
    print()
    print("  Subsequent runs (compiled):")

    times_numba = []
    for i in range(5):
        start_time = time.time()
        x = generate_time_series_numba(x0, n)
        end_time = time.time()
        elapsed = end_time - start_time
        times_numba.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.3f} ms")

    avg_numba = np.mean(times_numba) * 1000
    print(f"  Average: {avg_numba:.3f} ms")
    print()

    # Summary
    speedup = avg_python / avg_numba
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Standard Python: {avg_python:.3f} ms ({avg_python*1000:.1f} μs)")
    print(f"Numba Python:    {avg_numba:.3f} ms ({avg_numba*1000:.1f} μs)")
    print(f"Speedup:         {speedup:.1f}x faster with Numba")
    print("="*60)
