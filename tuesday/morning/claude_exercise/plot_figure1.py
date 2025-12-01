import numpy as np
import matplotlib.pyplot as plt

# Read data
data = np.loadtxt('figure1_data.txt')
x = data[:, 0]
g_x = data[:, 1]
diagonal = data[:, 2]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x, g_x, 'b-', linewidth=2, label=r'$g(x) = 4x(1-x)$')
plt.plot(x, diagonal, 'k--', linewidth=1, label='45 degrees')

plt.xlabel('x', fontsize=12)
plt.ylabel(r'$g(x)$', fontsize=12)
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.legend(fontsize=11)
plt.grid(False)

# Style similar to the PDF
plt.gca().set_facecolor('#E8E8F0')
plt.tight_layout()

# Save the figure
plt.savefig('figures/figure1.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure1.pdf', bbox_inches='tight')
print('Figure 1 saved to figures/figure1.png and figures/figure1.pdf')
plt.show()
plt.close()
