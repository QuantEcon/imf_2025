import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data points
n_points = 50
x = np.linspace(0, 10, n_points)

# True underlying function: f(x) = a * sin(b * x) + c
true_a, true_b, true_c = 3.0, 0.8, 2.0
y_true = true_a * np.sin(true_b * x) + true_c

# Add noise to create scatter
noise = np.random.normal(0, 0.5, n_points)
y_data = y_true + noise

# Define two nonlinear functions with different parameters
# f_θ: Better fit (close to true parameters)
theta = {'a': 2.9, 'b': 0.78, 'c': 2.1}
f_theta = theta['a'] * np.sin(theta['b'] * x) + theta['c']

# f_θ': Worse fit (far from true parameters)
theta_prime = {'a': 2.0, 'b': 1.2, 'c': 1.5}
f_theta_prime = theta_prime['a'] * np.sin(theta_prime['b'] * x) + theta_prime['c']

# Calculate residuals (sum of squared errors)
sse_theta = np.sum((y_data - f_theta)**2)
sse_theta_prime = np.sum((y_data - f_theta_prime)**2)


def base_plot():

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data points
    ax.scatter(x, y_data, c='black', s=50, alpha=0.6, zorder=3)

    # Formatting
    ax.set_xlabel('$x$', fontsize=16)
    ax.set_ylabel('$y$', fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def create_plot_0():

    ax = base_plot()

    # Plot the two fitted functions
    x_smooth = np.linspace(0, 10, 200)
    f_theta_smooth = theta['a'] * np.sin(theta['b'] * x_smooth) + theta['c']
    f_theta_prime_smooth = theta_prime['a'] * np.sin(theta_prime['b'] * x_smooth) + theta_prime['c']

    line1, = ax.plot(x_smooth, f_theta_smooth, linewidth=2.5, zorder=2)
    line2, = ax.plot(x_smooth, f_theta_prime_smooth, linewidth=2.5, zorder=1)

    # Add annotations with arrows pointing to the curves
    # Annotate f_θ (better fit) - bottom left
    ax.annotate(f'$\\ell(\\theta_1)$={sse_theta:.2f}',
                xy=(2.5, f_theta_smooth[25]), xytext=(0.5, 0.5),
                fontsize=16, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=line1.get_color(), alpha=0.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=line1.get_color()))

    # Annotate f_θ' (worse fit) - y=5
    ax.annotate(f'$\\ell(\\theta_0)$={sse_theta_prime:.2f}',
                xy=(7.0, f_theta_prime_smooth[140]), xytext=(5.5, 5.0),
                fontsize=16, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=line2.get_color(), alpha=0.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=line2.get_color()))

    plt.tight_layout()
    plt.savefig('nonlinear_fitting.pdf', bbox_inches='tight')
    plt.show()


def create_plot_1():

    ax = base_plot()
    # Plot the two fitted functions
    x_smooth = np.linspace(0, 10, 200)
    f_theta_smooth = theta['a'] * np.sin(theta['b'] * x_smooth) + theta['c']

    line1, = ax.plot(x_smooth, f_theta_smooth, linewidth=2.5, zorder=2)

    # Add annotations with arrows pointing to the curves
    # Annotate f_θ (better fit) - bottom left
    ax.annotate(f'$f$',
                xy=(2.5, f_theta_smooth[25]), xytext=(0.5, 0.5),
                fontsize=16, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=line1.get_color(), alpha=0.2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=line1.get_color()))

    # Add annotation pointing to a data point (x_i, y_i)
    # Choose a point near the middle of the figure for good visibility
    idx = 15
    ax.annotate(f'$(x_i, y_i)$',
                xy=(x[idx], y_data[idx]), xytext=(x[idx] + 1.5, y_data[idx] + 1.0),
                fontsize=16, ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

    plt.tight_layout()
    plt.savefig('nonlinear_fitting_1.pdf', bbox_inches='tight')
    plt.show()


create_plot_0()
create_plot_1()
