import numpy as np
import matplotlib.pyplot as plt

# Define a 2D function (a simple paraboloid, negated so maximum is at center)
def ell(theta1, theta2):
    return -(theta1**2 + 2*theta2**2)

# Define the gradient of ell
def gradient(theta1, theta2):
    d_ell_dtheta1 = -2*theta1
    d_ell_dtheta2 = -4*theta2
    return d_ell_dtheta1, d_ell_dtheta2

# Create a grid of points
theta1 = np.linspace(-3, 3, 300)
theta2 = np.linspace(-3, 3, 300)
Theta1, Theta2 = np.meshgrid(theta1, theta2)
Z = ell(Theta1, Theta2)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Choose a single point to show the gradient
ptheta1, ptheta2 = 1.5, 1.0
gtheta1, gtheta2 = gradient(ptheta1, ptheta2)

# Calculate the contour value at this point to ensure a contour goes through it
contour_at_point = ell(ptheta1, ptheta2)

# Create levels that include the value at our point
levels = np.linspace(Z.min(), Z.max(), 10)
# Add the exact contour value at our point if it's not already close to existing levels
if not any(abs(levels - contour_at_point) < 0.1):
    levels = np.sort(np.append(levels, contour_at_point))

# Plot contours
contours = ax.contour(Theta1, Theta2, Z, levels=levels, cmap='viridis', alpha=0.7, linewidths=1.5)
ax.clabel(contours, inline=True, fontsize=10)

# Plot filled contours for better visualization
ax.contourf(Theta1, Theta2, Z, levels=levels, cmap='viridis', alpha=0.2)

# Plot the point
ax.plot(ptheta1, ptheta2, 'ro', markersize=10, zorder=5)

# Scale the gradient vector for better visualization
scale = 0.2  # Reduced by 50%
arrow = ax.arrow(ptheta1, ptheta2, gtheta1*scale, gtheta2*scale,
                 head_width=0.2, head_length=0.15,
                 fc='darkgreen', ec='darkgreen', linewidth=3, alpha=0.9, zorder=4)

# Add annotation for the gradient vector (text only, no arrow)
ax.text(ptheta1 + gtheta1*scale + 0.5, ptheta2 + gtheta2*scale + 0.3,
        r'$\nabla \ell$',
        fontsize=18, color='darkgreen', fontweight='bold')

# Add the steepest descent vector (-∇ℓ)
arrow_descent = ax.arrow(ptheta1, ptheta2, -gtheta1*scale, -gtheta2*scale,
                         head_width=0.2, head_length=0.15,
                         fc='saddlebrown', ec='saddlebrown', linewidth=3, alpha=0.9, zorder=4)

# Add annotation for the descent vector
ax.text(ptheta1 - gtheta1*scale - 0.7, ptheta2 - gtheta2*scale - 0.3,
        r'$-\nabla \ell$',
        fontsize=18, color='saddlebrown', fontweight='bold')

# Draw a tangent line to the contour (perpendicular to gradient)
# Tangent direction is perpendicular to gradient: (-gtheta2, gtheta1)
tangent_length = 1.0
ttheta1, ttheta2 = -gtheta2, gtheta1
# Normalize
t_norm = np.sqrt(ttheta1**2 + ttheta2**2)
ttheta1, ttheta2 = ttheta1/t_norm, ttheta2/t_norm

# Draw tangent line segment
ax.plot([ptheta1 - ttheta1*tangent_length, ptheta1 + ttheta1*tangent_length],
        [ptheta2 - ttheta2*tangent_length, ptheta2 + ttheta2*tangent_length],
        'b--', linewidth=2.5, label='Tangent to contour', zorder=3)

# Labels and title
ax.set_xlabel(r'$\theta_1$', fontsize=14)
ax.set_ylabel(r'$\theta_2$', fontsize=14)
ax.set_title(r'Gradient $\nabla \ell$ of $\ell$ points in direction of steepest ascent',
             fontsize=15, fontweight='bold')
ax.legend(loc='upper left', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('gradient_steepest_ascent.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as 'gradient_steepest_ascent.png'")
