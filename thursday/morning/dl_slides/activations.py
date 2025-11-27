import numpy as np
import matplotlib.pyplot as plt

# Define x range
x = np.linspace(-5, 5, 100)

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def swish(x, beta=1):
    return x * sigmoid(beta * x)

def selu(x, alpha=1.6732, scale=1.0507):
    return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1))


# Create the figure and axis
fig, ax = plt.subplots()

# Plot activation functions
ax.plot(x, sigmoid(x),  label='sigmoid', linewidth=2)
ax.plot(x, tanh(x), label='tanh', linewidth=2)
ax.plot(x, relu(x), label='ReLU', linewidth=2)
ax.plot(x, swish(x), label='swish', linewidth=2)
ax.plot(x, selu(x), label='selu', linewidth=2)

ax.set_xlabel('$x$')
ax.set_ylabel('$\sigma(x)$')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax.set_ylim(-2, 2)
ax.set_xlim(-5, 5)
ax.legend(loc='lower right', frameon=False)

plt.savefig('activations.pdf')
plt.show()
