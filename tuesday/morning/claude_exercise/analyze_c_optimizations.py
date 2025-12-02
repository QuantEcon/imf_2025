import matplotlib.pyplot as plt
import numpy as np

# Optimization flags and their average times (in microseconds)
optimizations = [
    ('No optimization\n-O0', 920),
    ('Basic\n-O1', 629),
    ('Standard\n-O2', 504),
    ('Aggressive\n-O3', 450),
    ('-O3\n-march=native', 427),
    ('-O3 -march=native\n-mtune=native', 455),
    ('-O3 -march=native\n-mtune=native\n-ffast-math', 455),
    ('-O3 -march=native\n-mtune=native\n-ffast-math\n-funroll-loops', 418),
    ('-Ofast\n-march=native\n-mtune=native', 451),
    ('-Ofast\n-march=native\n-mtune=native\n-funroll-loops', 519),
]

labels = [opt[0] for opt in optimizations]
times_us = [opt[1] for opt in optimizations]

# Find best and baseline
best_idx = np.argmin(times_us)
baseline_time = times_us[0]
speedups = [baseline_time / t for t in times_us]

# Create bar plot
fig, ax = plt.subplots(figsize=(14, 8))
colors = ['#d62728' if i == 0 else '#2ca02c' if i == best_idx else '#1f77b4' for i in range(len(labels))]
bars = ax.barh(labels, times_us, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Time (microseconds)', fontsize=13)
ax.set_title('C Compiler Optimization Flags Performance Comparison\n100,001 iterations',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(0, max(times_us) * 1.15)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels on bars
for bar, time, speedup in zip(bars, times_us, speedups):
    width = bar.get_width()
    ax.text(width + 20, bar.get_y() + bar.get_height()/2,
            f'{time} μs ({speedup:.2f}x)',
            ha='left', va='center', fontsize=10, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', alpha=0.8, edgecolor='black', label='No optimization'),
    Patch(facecolor='#2ca02c', alpha=0.8, edgecolor='black', label='Best optimization'),
    Patch(facecolor='#1f77b4', alpha=0.8, edgecolor='black', label='Other optimizations')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plt.savefig('c_optimization_comparison.png', dpi=150, bbox_inches='tight')
print("C optimization comparison saved as 'c_optimization_comparison.png'\n")

# Print summary
print("="*80)
print("C COMPILER OPTIMIZATION FLAGS ANALYSIS")
print("="*80)
print(f"{'Flags':<50} {'Time (μs)':<12} {'Speedup':<10}")
print("-"*80)
for i, (label, time, speedup) in enumerate(zip(labels, times_us, speedups)):
    label_clean = label.replace('\n', ' ')
    marker = " ← FASTEST" if i == best_idx else " ← BASELINE" if i == 0 else ""
    print(f"{label_clean:<50} {time:>6} μs    {speedup:>5.2f}x{marker}")
print("="*80)

best_label = labels[best_idx].replace('\n', ' ')
improvement = (baseline_time - times_us[best_idx]) / baseline_time * 100
print(f"\nBest optimization: {best_label}")
print(f"Improvement over -O0: {improvement:.1f}% faster ({speedups[best_idx]:.2f}x speedup)")
print(f"Time reduction: {baseline_time} μs → {times_us[best_idx]} μs")

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)
print("1. -O3 -march=native gives best performance (427 μs)")
print("2. Adding more flags after -march=native shows diminishing returns")
print("3. -funroll-loops can help in some cases (418 μs)")
print("4. -Ofast with -funroll-loops is actually slower (519 μs)")
print("5. Overall speedup from -O0 to best: 2.2x faster")
print("="*80)
