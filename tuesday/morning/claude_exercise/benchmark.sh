#!/bin/bash

echo "=================================================="
echo "Performance Comparison: C vs Fortran vs Numba"
echo "=================================================="
echo ""

echo "Running C implementation 5 times..."
echo "-----------------------------------"
c_total=0
for i in {1..5}; do
    echo "Run $i:"
    c_time=$( { time ./quadratic_map_c > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}')
    echo "  Time: $c_time"
done

echo ""
echo "Running Fortran implementation 5 times..."
echo "-----------------------------------"
f_total=0
for i in {1..5}; do
    echo "Run $i:"
    f_time=$( { time ./quadratic_map_fortran > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}')
    echo "  Time: $f_time"
done

echo ""
echo "Running Numba/Python implementation 5 times..."
echo "-----------------------------------"
n_total=0
for i in {1..5}; do
    echo "Run $i:"
    n_time=$( { time python3 quadratic_map_numba.py > /dev/null 2>&1; } 2>&1 | grep real | awk '{print $2}')
    echo "  Time: $n_time"
done

echo ""
echo "==================================="
echo "Using /usr/bin/time for detailed comparison..."
echo "==================================="
echo ""

echo "C implementation:"
/usr/bin/time -v ./quadratic_map_c 2>&1 | grep -E "(User time|System time|Elapsed|Maximum resident)"

echo ""
echo "Fortran implementation:"
/usr/bin/time -v ./quadratic_map_fortran 2>&1 | grep -E "(User time|System time|Elapsed|Maximum resident)"

echo ""
echo "Numba/Python implementation:"
/usr/bin/time -v python3 quadratic_map_numba.py 2>&1 | grep -E "(User time|System time|Elapsed|Maximum resident)"
