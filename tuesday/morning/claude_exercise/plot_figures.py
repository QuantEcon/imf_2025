import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')

def plot_figure1():
    """Figure 1: The quadratic map and 45 degree line"""
    # Read the function data
    df = pd.read_csv('quadratic_function.csv')

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the quadratic map
    ax.plot(df['x'], df['g_x'], linewidth=2, label=r'$g(x) = 4x(1-x)$')

    # Plot the 45 degree line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='45 degrees')

    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('g(x)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure1_quadratic_map.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure1_quadratic_map.png', dpi=300, bbox_inches='tight')
    print("Generated Figure 1: figure1_quadratic_map.pdf")
    plt.show()
    plt.close()

def plot_figure2():
    """Figure 2: Time series from the quadratic map when x0 = 0.3"""
    # Read the trajectory data
    df = pd.read_csv('trajectory.csv')

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the time series with markers
    ax.plot(df['t'], df['x'], 'o-', linewidth=1, markersize=4)

    ax.set_xlabel('t', fontsize=14)
    ax.set_ylabel(r'$x_t$', fontsize=14)
    ax.set_title(r'Time series from the quadratic map when $x_0 = 0.3$', fontsize=14)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figure2_time_series.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure2_time_series.png', dpi=300, bbox_inches='tight')
    print("Generated Figure 2: figure2_time_series.pdf")
    plt.show()
    plt.close()

def plot_figure3():
    """Figure 3: Histogram of (x0, ..., xn) when n = 100,000"""
    # Read the histogram data
    df = pd.read_csv('histogram_data.csv')

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create histogram with appropriate bins
    n_bins = 100
    ax.hist(df['x'], bins=n_bins, edgecolor='white', linewidth=0.5, label='observations')

    ax.set_xlabel('state', fontsize=14)
    ax.set_ylabel('frequency', fontsize=14)
    ax.set_title(r'A histogram of $(x_0, \ldots, x_n)$ when $n = 100,000$', fontsize=14)
    ax.set_xlim(0, 1)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figure3_histogram.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('figure3_histogram.png', dpi=300, bbox_inches='tight')
    print("Generated Figure 3: figure3_histogram.pdf")
    plt.show()
    plt.close()

if __name__ == "__main__":
    print("Generating figures from C-generated data...\n")

    plot_figure1()
    plot_figure2()
    plot_figure3()

    print("\nAll figures generated successfully!")
    print("PDF files: figure1_quadratic_map.pdf, figure2_time_series.pdf, figure3_histogram.pdf")
    print("PNG files: figure1_quadratic_map.png, figure2_time_series.png, figure3_histogram.png")
