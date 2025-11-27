import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Create the figure and axes with fig, ax = plt.subplots() style
fig, ax = plt.subplots(figsize=(10, 6))

# Define the x range
x = np.linspace(0, 10, 1000)

# Create a function with several local maxima
def multi_peak_function(x):
    return 2*np.sin(x) + np.sin(2*x) + 0.5*np.sin(5*x) + 0.2*np.sin(10*x) + np.exp(-(x-5)**2/8)

# Calculate y values
y = multi_peak_function(x)

# Find peaks (local maxima)
peaks, _ = find_peaks(y, height=0, distance=50)
peak_heights = y[peaks]

# Find the global maximum
global_max_idx = np.argmax(y)
global_max_x = x[global_max_idx]
global_max_y = y[global_max_idx]

# Plot the function
ax.plot(x, y, 'b-', linewidth=2, label='$f$')

# Mark all local maxima
ax.plot(x[peaks], y[peaks], 'ro', markersize=8, label='local maxima')

# Highlight the global maximum with a different marker and color
ax.plot(global_max_x, global_max_y, 'go', markersize=8, label='global maximum')

# Annotate the global maximum
# ax.annotate(f'Global Max: ({global_max_x:.2f}, {global_max_y:.2f})',
#             xy=(global_max_x, global_max_y),
#             xytext=(global_max_x-1.5, global_max_y+0.5),
#             fontsize=10,
#             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

# Set labels and title
ax.set_xlabel('$\\theta$', fontsize=12)
ax.set_ylabel('$f(\\theta)$', fontsize=12)
ax.legend(loc='lower left')

# Add a shaded region around the global maximum for emphasis
ax.axvspan(global_max_x-0.3, global_max_x+0.3, alpha=0.2, color='green')

# Add text explaining the visualization
# ax.text(0.5, -1.1, 
#         'This visualization highlights the global maximum (green star) among several local maxima (red dots).\n'
#         'The dashed line shows the x-coordinate of the global maximum.',
#         transform=ax.transAxes, ha='center', va='center', fontsize=9)
plt.savefig('gmax.pdf')
plt.show()
