import matplotlib.pyplot as plt
import numpy as np

# All results (in microseconds)
results = {
    'Fortran\n(-O3 -march=native -mtune=native)': 389,
    'C\n(-O3 -march=native -mtune=native\n-ffast-math -funroll-loops)': 418,
    'Python + Numba\n(JIT compiled)': 267,
    'Python\n(standard interpreter)': 27616,
}

languages = list(results.keys())
times_us = list(results.values())

# Calculate speedups relative to standard Python
python_time = times_us[-1]
speedups = [python_time / t for t in times_us]

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Execution time (log scale)
ax1 = fig.add_subplot(gs[0, :])
colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#2ca02c']
bars1 = ax1.barh(languages, times_us, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Time (microseconds, log scale)', fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.set_title('Execution Time: 100,001 Iterations of Quadratic Map\n(All Optimizations Applied)',
              fontsize=15, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, axis='x')

for bar, time in zip(bars1, times_us):
    width = bar.get_width()
    ax1.text(width * 1.5, bar.get_y() + bar.get_height()/2,
            f'{time:.0f} μs',
            ha='left', va='center', fontsize=11, fontweight='bold')

# Plot 2: Speedup comparison
ax2 = fig.add_subplot(gs[1, 0])
bars2 = ax2.barh(languages, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Speedup vs Standard Python', fontsize=12, fontweight='bold')
ax2.set_title('Relative Performance', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.axvline(x=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Python baseline')
ax2.legend(fontsize=10)

for bar, speed in zip(bars2, speedups):
    width = bar.get_width()
    ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
            f'{speed:.1f}x',
            ha='left', va='center', fontsize=10, fontweight='bold')

# Plot 3: Simple bar chart (compiled languages only, linear scale)
ax3 = fig.add_subplot(gs[1, 1])
compiled_langs = ['Numba', 'Fortran', 'C']
compiled_times = [267, 389, 418]
compiled_colors = ['#9467bd', '#1f77b4', '#ff7f0e']
bars3 = ax3.bar(compiled_langs, compiled_times, color=compiled_colors, alpha=0.8,
                edgecolor='black', linewidth=1.5, width=0.6)
ax3.set_ylabel('Time (microseconds)', fontsize=12, fontweight='bold')
ax3.set_title('Compiled Code Performance\n(Linear Scale)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(compiled_times) * 1.3)

for bar, time in zip(bars3, compiled_times):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
            f'{time} μs',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Summary table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

table_data = [
    ['Language/Method', 'Time (μs)', 'Time (ms)', 'Speedup vs Python', 'Relative to Numba'],
    ['Fortran (optimized)', f'{389}', f'{389/1000:.3f}', f'{speedups[0]:.1f}x', f'{267/389:.2f}x slower'],
    ['C (optimized)', f'{418}', f'{418/1000:.3f}', f'{speedups[1]:.1f}x', f'{267/418:.2f}x slower'],
    ['Python + Numba', f'{267}', f'{267/1000:.3f}', f'{speedups[2]:.1f}x', '1.00x (fastest)'],
    ['Python (standard)', f'{27616}', f'{27616/1000:.3f}', '1.0x', f'{27616/267:.1f}x slower'],
]

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.15, 0.15, 0.20, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style the header row
for i in range(len(table_data[0])):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 3:  # Numba row (fastest)
            cell.set_facecolor('#E2EFDA')
        elif i % 2 == 0:
            cell.set_facecolor('#F2F2F2')

ax4.set_title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)

plt.savefig('final_comparison_with_numba.png', dpi=150, bbox_inches='tight')
print("Final comparison saved as 'final_comparison_with_numba.png'\n")

# Print detailed summary
print("="*90)
print("FINAL PERFORMANCE COMPARISON: Including Numba")
print("="*90)
print(f"{'Language/Method':<45} {'Time (μs)':<12} {'Speedup':<15}")
print("-"*90)

results_sorted = [
    ('Python + Numba (JIT)', 267, speedups[2]),
    ('Fortran (-O3 -march=native -mtune=native)', 389, speedups[0]),
    ('C (-O3 -march=native -mtune=native -ffast-math -funroll-loops)', 418, speedups[1]),
    ('Python (standard interpreter)', 27616, speedups[3]),
]

for lang, time, speed in results_sorted:
    marker = " ← FASTEST" if time == min([t for _, t, _ in results_sorted]) else ""
    print(f"{lang:<45} {time:>7.0f} μs    {speed:>6.1f}x{marker}")

print("="*90)

print("\nKEY FINDINGS:")
print("="*90)
print(f"1. Python + Numba is THE FASTEST: {267} μs")
print(f"   - 103.4x faster than standard Python")
print(f"   - 1.46x faster than Fortran")
print(f"   - 1.57x faster than C")
print()
print(f"2. Numba provides easy speedups:")
print(f"   - Just add @jit decorator to Python code")
print(f"   - Achieves compiled-language performance")
print(f"   - No need to write C/Fortran for many tasks")
print()
print(f"3. Language comparison (best optimizations):")
print(f"   - Python + Numba: {267} μs (FASTEST)")
print(f"   - Fortran: {389} μs (1.46x slower than Numba)")
print(f"   - C: {418} μs (1.57x slower than Numba)")
print(f"   - Python: {27616} μs (103.4x slower than Numba)")
print()
print(f"4. Practical implications:")
print(f"   - For pure Python users: Numba gives ~100x speedup")
print(f"   - Numba beats manually optimized C/Fortran here")
print(f"   - Best of both worlds: Python convenience + compiled speed")
print("="*90)

# Create a simpler visualization
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left plot: All languages (log scale)
simple_langs = ['Numba', 'Fortran', 'C', 'Python']
simple_times = [267, 389, 418, 27616]
simple_colors = ['#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c']

bars = ax1.bar(simple_langs, simple_times, color=simple_colors, alpha=0.8,
               edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('Time (microseconds, log scale)', fontsize=13, fontweight='bold')
ax1.set_yscale('log')
ax1.set_title('Performance: 100,001 Iterations\n(All Optimizations)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for bar, time in zip(bars, simple_times):
    height = bar.get_height()
    speedup_val = python_time / time
    ax1.text(bar.get_x() + bar.get_width()/2., height * 2,
            f'{time:.0f} μs\n({speedup_val:.1f}x)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Right plot: Compiled languages only (linear scale)
compiled_bars = ax2.bar(compiled_langs, compiled_times, color=compiled_colors[:3],
                        alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('Time (microseconds)', fontsize=13, fontweight='bold')
ax2.set_title('Compiled Code Comparison\n(Numba vs C vs Fortran)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0, 500)

for bar, time in zip(compiled_bars, compiled_times):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 15,
            f'{time} μs',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('simple_comparison_with_numba.png', dpi=150, bbox_inches='tight')
print("\nSimple comparison saved as 'simple_comparison_with_numba.png'")
