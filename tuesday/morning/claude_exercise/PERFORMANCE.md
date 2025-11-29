# Performance Comparison: C vs Fortran vs Numba

## Implementation Details

All three implementations generate the same data for the quadratic map g(x) = 4x(1-x):
- Quadratic function values (1,001 points)
- Time series trajectory (151 points, x₀ = 0.3)
- Histogram data (100,001 points, x₀ = 0.3)

### Compilation/Execution:
- **C**: `gcc -O3 -o quadratic_map_c quadratic_map.c -lm`
- **Fortran**: `gfortran -O3 -o quadratic_map_fortran quadratic_map.f90`
- **Numba**: `python3 quadratic_map_numba.py` (JIT compilation with warmup)

## Performance Results

### Average Execution Time (5 runs each)

| Implementation | Average Time | Range | Relative Speed |
|---------------|--------------|-------|----------------|
| **C** 🥇       | ~0.020s      | 0.019-0.022s | 1.0x (baseline) |
| **Fortran** 🥈 | ~0.046s      | 0.044-0.047s | 2.3x slower |
| **Numba** 🥉   | ~0.891s      | 0.868-0.908s | 44.5x slower |

### Detailed Metrics

| Metric | C | Fortran | Numba |
|--------|---|---------|-------|
| User time | 0.01s | 0.03s | 0.81s |
| System time | 0.00s | 0.00s | 0.06s |
| Wall clock time | 0.01s | 0.04s | 0.88s |
| Memory (RSS) | 1,640 KB | 2,808 KB | 166,900 KB |

## Summary

**Winner: C is the clear performance champion** for this I/O-heavy task.

### Performance Rankings:
1. **C**: Fastest - 20ms
2. **Fortran**: 2.3x slower than C - 46ms
3. **Numba**: 44.5x slower than C - 891ms

### Key Observations:

1. **Speed**: C completes in ~20ms vs Fortran's ~46ms
2. **Memory**: C uses ~40% less memory (1.6 MB vs 2.7 MB)
3. **Consistency**: Both implementations are very consistent across runs
4. **Correctness**: Both produce identical numerical results (verified)

### Why is C faster here?

The performance difference is likely due to:
1. **I/O operations**: File writing is the dominant operation in this code. C's `fprintf` may be more efficient than Fortran's formatted I/O for these small writes.
2. **Runtime overhead**: Fortran's runtime system has more overhead for formatted output.
3. **Compiler differences**: gcc vs gfortran may have different optimization strategies.

### When might Fortran be faster?

For compute-intensive operations (less I/O, more math):
- Large matrix operations
- Scientific simulations with heavy numerical computation
- Code using BLAS/LAPACK libraries
- Vectorizable loops with array operations

In this exercise, the bottleneck is file I/O, not computation, which favors C.
