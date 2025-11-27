import matplotlib.pyplot as plt
import numpy as np

# Create data
inflation = np.linspace(0, 10, 100)

# Best response function (slightly nonlinear with cosine)
best_response = 1.5 * inflation + 2 + 0.2 * np.cos(inflation * 0.8)

# DL approximation (similar but slightly different)
dl_approximation = 1.5 * inflation + 2.5 + 0.3 * np.sin(inflation * 0.5)

# Create the plot
plt.figure(figsize=(10, 6))
line1, = plt.plot(inflation, best_response, linewidth=2)
line2, = plt.plot(inflation, dl_approximation, linewidth=2)

plt.xlabel('inflation', fontsize=16)
plt.ylabel('interest rate', fontsize=16)

# Remove ticks
plt.xticks([])
plt.yticks([])

# Add annotations with curved arrows
# Best response annotation (arrow from below, shifted down to y=5)
br_x, br_y = 7, 1.5 * 7 + 2 + 0.2 * np.cos(7 * 0.8)
plt.annotate('best response',
             xy=(br_x, br_y),
             xytext=(br_x - 2.5, 5),
             fontsize=16,
             color=line1.get_color(),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=1.5, color=line1.get_color()))

# DL approximation annotation (arrow from above)
dl_x, dl_y = 7, 1.5 * 7 + 2.5 + 0.3 * np.sin(7 * 0.5)
plt.annotate('DL approximation',
             xy=(dl_x, dl_y),
             xytext=(dl_x - 1.5, dl_y + 2),
             fontsize=16,
             color=line2.get_color(),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3', lw=1.5, color=line2.get_color()))

plt.tight_layout()

# Save the figure
plt.savefig('inflation_interest.pdf', bbox_inches='tight')
print("Plot saved as 'inflation_interest.pdf'")

# Display the plot
plt.show()
