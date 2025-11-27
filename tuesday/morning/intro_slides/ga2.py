import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Define the objective function (we want to maximize this)
def objective_function(x, y):
    """A function with multiple peaks - good for demonstrating gradient ascent"""
    return -(x**2 + y**2) + 0.5 * np.sin(4*x) * np.cos(4*y) + 2

# Define the gradient of the objective function
def gradient(x, y):
    """Compute the gradient (partial derivatives)"""
    dx = -2*x + 2*np.cos(4*x)*np.cos(4*y)
    dy = -2*y - 2*np.sin(4*x)*np.sin(4*y)
    return dx, dy

# Gradient ascent algorithm
def gradient_ascent(start_x, start_y, learning_rate=0.01, num_iterations=100):
    """Perform gradient ascent starting from (start_x, start_y)"""
    path_x = [start_x]
    path_y = [start_y]
    path_z = [objective_function(start_x, start_y)]
    
    x, y = start_x, start_y
    
    for i in range(num_iterations):
        # Calculate gradient
        dx, dy = gradient(x, y)
        
        # Update position (moving in direction of gradient for ascent)
        x = x + learning_rate * dx
        y = y + learning_rate * dy
        
        # Store path
        path_x.append(x)
        path_y.append(y)
        path_z.append(objective_function(x, y))
    
    return np.array(path_x), np.array(path_y), np.array(path_z)

# Create the 3D visualization
fig = plt.figure(figsize=(14, 10))

# Create two subplots: 3D surface and contour plot
ax1 = fig.add_subplot(111, projection='3d')

# Create a grid for the surface plot
x_range = np.linspace(-2, 2, 50)
y_range = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x_range, y_range)
Z = objective_function(X, Y)

# Plot the 3D surface
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, edgecolor='none')


# Run gradient ascent from multiple starting points
starting_points = [(-1.5, -1.5), (1.2, -0.8), (-0.5, 1.3), (1.8, 1.2)]
colors = ['black', 'blue', 'green', 'orange']

for i, (start_x, start_y) in enumerate(starting_points):
    # Perform gradient ascent
    path_x, path_y, path_z = gradient_ascent(start_x, start_y, learning_rate=0.05, num_iterations=50)
    
    color = colors[i]
    
    # Plot 3D path
    ax1.plot(path_x, path_y, path_z, color=color, linewidth=3, alpha=0.8, label=f'Path {i+1}')
    ax1.scatter([start_x], [start_y], [objective_function(start_x, start_y)], 
                color=color, s=100, marker='o', label=f'Start {i+1}')
    ax1.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], 
                color=color, s=100, marker='*', label=f'End {i+1}')
    
    

plt.tight_layout()
plt.savefig('ga2.pdf')
plt.show()

