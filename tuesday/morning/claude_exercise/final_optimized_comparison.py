import matplotlib.pyplot as plt
import numpy as np

# Best times for each language (in microseconds)
results = {
    'Python (no optimization)': 27616,
    'C (-O3)': 450,
    'C (-O3 -march=native -mtune=native -ffast-math -funroll-loops)': 418,
    'Fortran (-O2)': 441,
    'Fortran (-O3 -march=native -mtune=native)': 389,
}

# Organize data
languages = list(results.keys())
times_us = list(results.values())
times_ms = [t/1000 for t in times_us]

# Calculate speedups relative to Python
python_time = times_us[0]
speedups = [python_time / t for t in times_us]

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Execution time (log scale for better visualization)
colors = ['#2ca02c', '#ff7f0e', '#d62728', '#1f77b4', '#9467bd']
bars1 = ax1.barh(languages, times_us, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Time (microseconds, log scale)', fontsize=12)
ax1.set_xscale('log')
ax1.set_title('Execution Time: 100,001 Iterations\n(Optimized Compilation)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, time in zip(bars1, times_us):
    width = bar.get_width()
    ax1.text(width * 1.3, bar.get_y() + bar.get_height()/2,
            f'{time:.0f} μs',
            ha='left', va='center', fontsize=10, fontweight='bold')

# Plot 2: Speedup
bars2 = ax2.barh(languages, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Speedup vs Python', fontsize=12)
ax2.set_title('Relative Performance\n(Higher is Better)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.axvline(x=1, color='black', linestyle='--', linewidth=2, alpha=0.5)

# Add value labels
for bar, speed in zip(bars2, speedups):
    width = bar.get_width()
    ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{speed:.1f}x',
            ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('final_optimized_comparison.png', dpi=150, bbox_inches='tight')
print("Final comparison saved as 'final_optimized_comparison.png'\n")

# Print detailed summary
print("="*90)
print("FINAL OPTIMIZED PERFORMANCE COMPARISON")
print("="*90)
print(f"{'Configuration':<60} {'Time (μs)':<12} {'Speedup':<10}")
print("-"*90)
for lang, time, speed in zip(languages, times_us, speedups):
    print(f"{lang:<60} {time:>6.0f} μs    {speed:>5.1f}x")
print("="*90)

# Key insights
print("\nKEY FINDINGS:")
print("="*90)
print(f"1. Best Fortran: {results['Fortran (-O3 -march=native -mtune=native)']} μs")
print(f"   Flags: -O3 -march=native -mtune=native")
print(f"   Speedup: {speedups[4]:.1f}x faster than Python\n")

print(f"2. Best C: {results['C (-O3 -march=native -mtune=native -ffast-math -funroll-loops)']} μs")
print(f"   Flags: -O3 -march=native -mtune=native -ffast-math -funroll-loops")
print(f"   Speedup: {speedups[2]:.1f}x faster than Python\n")

print(f"3. Optimization Impact:")
c_improvement = (results['C (-O3)'] - results['C (-O3 -march=native -mtune=native -ffast-math -funroll-loops)']) / results['C (-O3)'] * 100
fortran_improvement = (results['Fortran (-O2)'] - results['Fortran (-O3 -march=native -mtune=native)']) / results['Fortran (-O2)'] * 100
print(f"   - C improved by {c_improvement:.1f}% with aggressive flags")
print(f"   - Fortran improved by {fortran_improvement:.1f}% with aggressive flags\n")

print(f"4. With best optimizations:")
print(f"   - Fortran is {speedups[4]/speedups[2]:.2f}x faster than C")
print(f"   - Both compiled languages are ~40-70x faster than Python")
print("="*90)

# Create a simpler comparison chart
fig2, ax = plt.subplots(figsize=(12, 6))

simple_labels = ['Fortran\n(optimized)', 'C\n(optimized)', 'Python']
simple_times = [389, 418, 27616]
simple_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

bars = ax.bar(simple_labels, simple_times, color=simple_colors, alpha=0.8,
              edgecolor='black', linewidth=2, width=0.6)

ax.set_ylabel('Time (microseconds, log scale)', fontsize=13)
ax.set_yscale('log')
ax.set_title('Best Performance: 100,001 Iterations of Quadratic Map',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, time in zip(bars, simple_times):
    height = bar.get_height()
    speedup_val = python_time / time
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.5,
            f'{time:.0f} μs\n({speedup_val:.1f}x)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add optimization details as text
opt_text = (
    "Fortran: -O3 -march=native -mtune=native\n"
    "C: -O3 -march=native -mtune=native -ffast-math -funroll-loops\n"
    "Python: Standard interpreter"
)
ax.text(0.02, 0.98, opt_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('simple_optimized_comparison.png', dpi=150, bbox_inches='tight')
print("\nSimple comparison saved as 'simple_optimized_comparison.png'")
