#!/bin/bash

echo "=========================================================="
echo "Performance Comparison: Pure Data Generation (No I/O)"
echo "=========================================================="
echo ""
echo "This benchmark measures ONLY the computational time to"
echo "generate the data, excluding all file I/O operations."
echo ""
echo "=========================================================="
echo ""

echo "Running C implementation 10 times..."
echo "-----------------------------------"
for i in {1..10}; do
    ./quadratic_map_timed_c
done

echo ""
echo "Running Fortran implementation 10 times..."
echo "-----------------------------------"
for i in {1..10}; do
    ./quadratic_map_timed_fortran
done

echo ""
echo "Running Numba/Python implementation 10 times..."
echo "-----------------------------------"
for i in {1..10}; do
    python3 quadratic_map_timed.py
done

echo ""
echo "=========================================================="
echo "Summary: Computing statistics..."
echo "=========================================================="
echo ""

echo "C average (excluding first run):"
./quadratic_map_timed_c > /dev/null  # warm up
for i in {1..5}; do
    ./quadratic_map_timed_c
done | awk '{sum+=$5; count++} END {printf "  Mean: %.6f seconds\n", sum/count}'

echo ""
echo "Fortran average (excluding first run):"
./quadratic_map_timed_fortran > /dev/null  # warm up
for i in {1..5}; do
    ./quadratic_map_timed_fortran
done | awk '{sum+=$5; count++} END {printf "  Mean: %.6f seconds\n", sum/count}'

echo ""
echo "Numba average (after JIT warmup):"
python3 quadratic_map_timed.py > /dev/null  # warm up
for i in {1..5}; do
    python3 quadratic_map_timed.py
done | grep "Numba data generation" | awk '{sum+=$5; count++} END {printf "  Mean: %.6f seconds\n", sum/count}'
