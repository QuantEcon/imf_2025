import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic/logistic map
def g(x):
    """The quadratic map g(x) = 4x(1-x)"""
    return 4 * x * (1 - x)

# Figure 1: The quadratic map and 45 degree line
def figure_1():
    x = np.linspace(0, 1, 1000)
    y = g(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, linewidth=2, label='g(x) = 4x(1 − x)')
    plt.plot(x, x, 'k--', linewidth=1.5, label='45 degrees')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('g(x)', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.title('The quadratic map and 45 degree line', fontsize=13)
    plt.tight_layout()
    plt.savefig('figure_1_quadratic_map.png', dpi=150)
    print("Figure 1 saved as 'figure_1_quadratic_map.png'")

# Figure 2: Time series from the quadratic map when x0 = 0.3
def figure_2():
    x0 = 0.3
    n = 150

    # Generate the time series
    x = np.zeros(n)
    x[0] = x0
    for t in range(n - 1):
        x[t + 1] = g(x[t])

    plt.figure(figsize=(10, 6))
    plt.plot(range(n), x, 'o-', linewidth=1, markersize=3)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('$x_t$', fontsize=12)
    plt.xlim(0, 150)
    plt.ylim(0, 1)
    plt.title(f'Time series from the quadratic map when $x_0 = {x0}$', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure_2_time_series.png', dpi=150)
    print("Figure 2 saved as 'figure_2_time_series.png'")

# Figure 3: Histogram of (x0, ..., xn) when n = 100,000
def figure_3():
    x0 = 0.3
    n = 100001  # 100,001 to get 100,000 observations including x0

    # Generate the long time series
    x = np.zeros(n)
    x[0] = x0
    for t in range(n - 1):
        x[t + 1] = g(x[t])

    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=100, edgecolor='none', alpha=0.8, label='observations')
    plt.xlabel('state', fontsize=12)
    plt.ylabel('frequency', fontsize=12)
    plt.xlim(0, 1)
    plt.title(f'A histogram of $(x_0, ..., x_n)$ when $n = 100,000$', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure_3_histogram.png', dpi=150)
    print("Figure 3 saved as 'figure_3_histogram.png'")

if __name__ == '__main__':
    print("Generating Figure 1...")
    figure_1()
    print("\nGenerating Figure 2...")
    figure_2()
    print("\nGenerating Figure 3...")
    figure_3()
    print("\nAll figures generated successfully!")
    plt.show()
