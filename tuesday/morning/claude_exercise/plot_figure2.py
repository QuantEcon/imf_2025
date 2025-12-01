import numpy as np
import matplotlib.pyplot as plt

# Read data
data = np.loadtxt('figure2_data.txt')
t = data[:, 0]
x_t = data[:, 1]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(t, x_t, 'o-', linewidth=1, markersize=3, color='#1f77b4')

plt.xlabel('t', fontsize=12)
plt.ylabel(r'$x_t$', fontsize=12)
plt.title(r'Time series from the quadratic map when $x_0 = 0.3$', fontsize=12)
plt.xlim(0, 150)
plt.ylim(0.0, 1.0)
plt.grid(False)

# Style similar to the PDF
plt.gca().set_facecolor('#E8E8F0')
plt.tight_layout()

# Save the figure
plt.savefig('figures/figure2.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure2.pdf', bbox_inches='tight')
print('Figure 2 saved to figures/figure2.png and figures/figure2.pdf')
plt.show()
plt.close()
