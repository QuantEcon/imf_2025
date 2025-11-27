import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 200)
f_x = x * np.tanh(x/4) / 2

fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(x, f_x, 'b-', linewidth=2, label='consumption policy $\\sigma$')
ax.plot(x, x, 'k--', linewidth=1.5, label='45 degrees')

ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

ax.legend(fontsize=16)

ax.set_xlabel('wealth', fontsize=16)
ax.set_ylabel('consumption', fontsize=16)

# Set axis bounds to show relevant part of the function
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

plt.savefig('conspol.pdf')
plt.show()
