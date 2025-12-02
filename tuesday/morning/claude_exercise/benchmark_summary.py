import matplotlib.pyplot as plt
import numpy as np

# Average times from the benchmarks (in milliseconds)
languages = ['Fortran', 'C', 'Python']
times_ms = [0.549, 0.682, 27.616]  # Average times in milliseconds
speedup = [times_ms[2] / t for t in times_ms]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Execution time
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars1 = ax1.bar(languages, times_ms, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Time (milliseconds)', fontsize=12)
ax1.set_title('Execution Time: 100,001 Iterations', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(times_ms) * 1.2)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, time in zip(bars1, times_ms):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.3f} ms',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 2: Speedup relative to Python
bars2 = ax2.bar(languages, speedup, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Speedup vs Python', fontsize=12)
ax2.set_title('Relative Performance', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(speedup) * 1.2)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Python baseline')
ax2.legend()

# Add value labels on bars
for bar, speed in zip(bars2, speedup):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{speed:.1f}x',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('benchmark_comparison.png', dpi=150)
print("Benchmark comparison saved as 'benchmark_comparison.png'")

# Print summary table
print("\n" + "="*60)
print("BENCHMARK SUMMARY: Quadratic Map Time Series (100,001 obs)")
print("="*60)
print(f"{'Language':<12} {'Time (ms)':<15} {'Speedup vs Python':<20}")
print("-"*60)
for lang, time, speed in zip(languages, times_ms, speedup):
    print(f"{lang:<12} {time:>8.3f} ms     {speed:>8.1f}x faster")
print("="*60)
print(f"\nFortran is {speedup[0]/speedup[1]:.2f}x faster than C")
print(f"Fortran is {speedup[0]:.1f}x faster than Python")
print(f"C is {speedup[1]:.1f}x faster than Python")
