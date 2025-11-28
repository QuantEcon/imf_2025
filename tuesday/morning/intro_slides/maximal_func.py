import numpy as np
import matplotlib.pyplot as plt

# Define the interval
x = np.linspace(-2, 4, 1000)

# Define several functions
f1 = x**2                   # Quadratic function
f2 = np.sin(x) + 2          # Sine function shifted up
f3 = np.exp(x/3)            # Exponential function
f4 = np.log(x + 3)          # Logarithmic function
f5 = np.maximum(f1, np.maximum(f2, np.maximum(f3, f4))) + 0.5  # Function pointwise greater than others

# Create the plot using the fig, ax = plt.subplots() style
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each function
ax.plot(x, f1, linewidth=1.5)
ax.plot(x, f2, linewidth=1.5)
ax.plot(x, f3, linewidth=1.5)
ax.plot(x, f4, linewidth=1.5)
ax.plot(x, f5, 'k-', linewidth=2.5, label=r'$v_s \leq v_\sigma$ for all $s \in \Sigma$')

# Add grid, legend and labels
ax.legend(loc='upper left', fontsize=16)

# Add text annotation explaining the dominant function
mid_index = 500
ax.annotate('greatest element $v_\\sigma$', 
            fontsize=16,
            xy=(x[mid_index], f5[mid_index]),
            xytext=(-1, f5[mid_index] + 5),
            arrowprops=dict(arrowstyle='->'))
plt.savefig('maximal_func.pdf')
plt.show()
