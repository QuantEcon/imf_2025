# Benchmark Results: C vs Fortran vs Numba

## Two Different Benchmarks

### Benchmark 1: Complete Program (with I/O)
Measures the total time including file I/O operations.

### Benchmark 2: Pure Computation (no I/O)
Measures ONLY the time to generate data in memory, excluding file writes.

---

## Results Summary

### Benchmark 1: Complete Program (with I/O)

| Implementation | Average Time | Relative Speed |
|---------------|--------------|----------------|
| **C** 🥇       | 20 ms       | 1.0x (baseline) |
| **Fortran** 🥈 | 46 ms       | 2.3x slower |
| **Numba** 🥉   | 891 ms      | 44.5x slower |

**Winner: C** - File I/O dominates, and C's `fprintf` is highly optimized.

---

### Benchmark 2: Pure Computation (no I/O)

| Implementation | Average Time | Relative Speed |
|---------------|--------------|----------------|
| **Numba** 🥇   | 0.470 ms    | 1.0x (baseline) |
| **C** 🥈       | 0.497 ms    | 1.06x slower |
| **Fortran** 🥉 | 0.512 ms    | 1.09x slower |

**Winner: Numba** - All three are essentially tied! Within ~9% of each other.

---

## Key Insights

### 1. I/O vs Computation Changes Everything

When **I/O dominates** (Benchmark 1):
- C is 44x faster than Numba
- Python's file I/O overhead is massive
- Low-level I/O operations matter

When **computation dominates** (Benchmark 2):
- All three are virtually identical
- JIT compilation works!
- NumPy/Numba compiles to competitive machine code

### 2. Performance Breakdown

**Benchmark 1 Times:**
- C: 20 ms total → 0.5 ms compute + 19.5 ms I/O
- Fortran: 46 ms total → 0.5 ms compute + 45.5 ms I/O
- Numba: 891 ms total → 0.5 ms compute + 890.5 ms I/O + startup

**I/O Overhead:**
- C I/O: 19.5 ms (98% of runtime)
- Fortran I/O: 45.5 ms (99% of runtime)
- Numba I/O: 890.5 ms (99.9% of runtime)

### 3. Pure Computation Performance

For the actual numerical work (generating 101,152 data points):

| Implementation | Time (μs) | Points/μs |
|---------------|-----------|-----------|
| Numba         | 470       | 215 |
| C             | 497       | 204 |
| Fortran       | 512       | 197 |

All three achieve similar throughput: ~200 points per microsecond.

---

## Conclusions

### When to Use Each Language:

**Use C when:**
- ✅ File I/O performance matters
- ✅ Minimal dependencies needed
- ✅ Standalone executables required
- ✅ Memory footprint is critical

**Use Fortran when:**
- ✅ Matrix operations with BLAS/LAPACK
- ✅ Legacy code integration
- ✅ Array-heavy scientific computing
- ✅ Fortran library ecosystem needed

**Use Numba when:**
- ✅ **Compute-intensive inner loops** (this is where it shines!)
- ✅ Python ecosystem integration (NumPy, SciPy, Pandas)
- ✅ Rapid prototyping and iteration
- ✅ I/O is minimal or handled separately
- ✅ Dynamic/interactive development

### The Big Takeaway

**For pure numerical computation, Numba matches C and Fortran!**

The massive 44x difference in Benchmark 1 was entirely due to:
1. Python startup time
2. Python's file I/O overhead
3. Library import time

When you isolate the actual numerical work, Numba's JIT compiler generates machine code that's just as fast as C and Fortran.

### Practical Advice

For scientific computing workloads:
- Use **Numba for compute kernels** (hot loops, numerical algorithms)
- Use **efficient I/O libraries** (HDF5, NumPy's binary formats, not CSV)
- Consider **separating computation from I/O** in your design
- Profile first - don't assume the bottleneck!

---

## Technical Details

### Data Generated
- Quadratic function values: 1,001 points
- Time series trajectory: 151 points
- Histogram data: 100,001 points
- **Total: 101,152 floating-point values**

### Compilation Flags
- C: `gcc -O3 -o quadratic_map_timed_c quadratic_map_timed.c -lm`
- Fortran: `gfortran -O3 -o quadratic_map_timed_fortran quadratic_map_timed.f90`
- Numba: JIT compilation with warmup run

### Timing Methods
- C: `clock_gettime(CLOCK_MONOTONIC)` for nanosecond precision
- Fortran: `cpu_time()` for CPU time
- Python: `time.perf_counter()` for high-resolution timing
