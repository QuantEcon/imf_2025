import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Figure 1: The quadratic map and 45 degree line
def plot_figure1():
    data = pd.read_csv('data_figure1.csv')

    plt.figure(figsize=(8, 6))
    plt.plot(data['x'], data['g_x'], linewidth=2, label='g(x) = 4x(1 − x)')
    plt.plot(data['x'], data['line_45'], 'k--', linewidth=1.5, label='45 degrees')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('g(x)', fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.title('The quadratic map and 45 degree line', fontsize=13)
    plt.tight_layout()
    plt.savefig('c_figure_1_quadratic_map.png', dpi=150)
    print("Figure 1 saved as 'c_figure_1_quadratic_map.png'")

# Figure 2: Time series from the quadratic map when x0 = 0.3
def plot_figure2():
    data = pd.read_csv('data_figure2.csv')

    plt.figure(figsize=(10, 6))
    plt.plot(data['t'], data['x_t'], 'o-', linewidth=1, markersize=3)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('$x_t$', fontsize=12)
    plt.xlim(0, 150)
    plt.ylim(0, 1)
    plt.title('Time series from the quadratic map when $x_0 = 0.3$', fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('c_figure_2_time_series.png', dpi=150)
    print("Figure 2 saved as 'c_figure_2_time_series.png'")

# Figure 3: Histogram of (x0, ..., xn) when n = 100,000
def plot_figure3():
    data = pd.read_csv('data_figure3.csv')

    plt.figure(figsize=(10, 6))
    plt.hist(data['x_t'], bins=100, edgecolor='none', alpha=0.8, label='observations')
    plt.xlabel('state', fontsize=12)
    plt.ylabel('frequency', fontsize=12)
    plt.xlim(0, 1)
    plt.title('A histogram of $(x_0, ..., x_n)$ when $n = 100,000$', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('c_figure_3_histogram.png', dpi=150)
    print("Figure 3 saved as 'c_figure_3_histogram.png'")

if __name__ == '__main__':
    print("Plotting figures from C-generated data...\n")
    print("Generating Figure 1...")
    plot_figure1()
    print("\nGenerating Figure 2...")
    plot_figure2()
    print("\nGenerating Figure 3...")
    plot_figure3()
    print("\nAll figures generated successfully!")
    plt.show()
