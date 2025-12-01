import numpy as np
import matplotlib.pyplot as plt

# Read data
data = np.loadtxt('figure3_data.txt')

# Create the histogram
plt.figure(figsize=(10, 6))
plt.hist(data, bins=100, color='#6baed6', edgecolor='#4292c6', alpha=0.9)

plt.xlabel('state', fontsize=12)
plt.ylabel('frequency', fontsize=12)
plt.title(r'A histogram of $(x_0, \ldots, x_n)$ when $n = 100,000$', fontsize=12)
plt.xlim(0.0, 1.0)
plt.grid(False)

# Style similar to the PDF
plt.gca().set_facecolor('#E8E8F0')

# Add legend
plt.legend(['observations'], fontsize=10, loc='upper center')
plt.tight_layout()

# Save the figure
plt.savefig('figures/figure3.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/figure3.pdf', bbox_inches='tight')
print('Figure 3 saved to figures/figure3.png and figures/figure3.pdf')
plt.show()
plt.close()
