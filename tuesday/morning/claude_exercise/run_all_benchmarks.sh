#!/bin/bash

echo "=========================================="
echo "Benchmarking Time Series Generation"
echo "100,001 observations from quadratic map"
echo "=========================================="
echo ""

echo "Running C benchmark (5 runs)..."
for i in {1..5}; do
    ./benchmark_c
done
echo ""

echo "Running Fortran benchmark (5 runs)..."
for i in {1..5}; do
    ./benchmark_fortran
done
echo ""

echo "Running Python benchmark (5 runs)..."
for i in {1..5}; do
    python benchmark_python.py
done
