#!/bin/bash

echo "=========================================="
echo "C Compiler Optimization Flag Comparison"
echo "=========================================="
echo ""

# Test different optimization flags
flags=(
    "-O0"
    "-O1"
    "-O2"
    "-O3"
    "-O3 -march=native"
    "-O3 -march=native -mtune=native"
    "-O3 -march=native -mtune=native -ffast-math"
    "-O3 -march=native -mtune=native -ffast-math -funroll-loops"
    "-Ofast -march=native -mtune=native"
    "-Ofast -march=native -mtune=native -funroll-loops"
)

for flag in "${flags[@]}"; do
    echo "Compiling with: gcc $flag -o benchmark_c_opt benchmark_c.c -lm"
    gcc $flag -o benchmark_c_opt benchmark_c.c -lm 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "Running benchmark (5 runs):"
        for i in {1..5}; do
            ./benchmark_c_opt
        done
        echo ""
    else
        echo "Compilation failed!"
        echo ""
    fi
done

rm -f benchmark_c_opt
