import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function F and its gradient
def F(x, y):
    """Function to maximize: a bivariate Gaussian-like function"""
    return -(x**2 + y**2 - 2*x - 4*y + 3) + 10

def gradient_F(x, y):
    """Gradient of F (partial derivatives)"""
    dF_dx = -2*x + 2
    dF_dy = -2*y + 4
    return np.array([dF_dx, dF_dy])

# Gradient ascent algorithm
def gradient_ascent(start_point, learning_rate=0.1, max_iterations=50, tolerance=1e-6):
    """Perform gradient ascent to find maximum"""
    path = [start_point]
    point = np.array(start_point)
    
    for i in range(max_iterations):
        grad = gradient_F(point[0], point[1])
        new_point = point + learning_rate * grad
        
        # Check for convergence
        if np.linalg.norm(new_point - point) < tolerance:
            break
            
        point = new_point
        path.append(point.copy())
    
    return np.array(path)

# Create meshgrid for contour plot
x = np.linspace(-2, 4, 100)
y = np.linspace(-1, 5, 100)
X, Y = np.meshgrid(x, y)
Z = F(X, Y)

# Create the visualization
fig = plt.figure(figsize=(10, 8))

# 3D surface plot
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

start = (-1.5, -0.5)
path_3d = gradient_ascent(start)
z_path = [F(point[0], point[1]) for point in path_3d]
ax.plot(path_3d[:, 0], path_3d[:, 1], z_path, 'o-', 
        color='black', linewidth=3, markersize=6, label=f'Start: {start}')
# Mark starting point
ax.plot([start[0]], [start[1]], [F(start[0], start[1])], 's', 
        color='black', markersize=8)
# Mark ending point
ax.plot([path_3d[-1, 0]], [path_3d[-1, 1]], [z_path[-1]], '*', 
        color='black', markersize=12)

# Mark the true maximum
true_max = (1, 2)
ax.plot([true_max[0]], [true_max[1]], [F(true_max[0], true_max[1])], 
        'k*', markersize=15, label='True Maximum')

ax.set_xlabel('$\\theta_1$')
ax.set_ylabel('$\\theta_2$')
ax.set_zlabel('$F(\\theta)$')

plt.savefig('grad_ascent.pdf')
plt.show()

