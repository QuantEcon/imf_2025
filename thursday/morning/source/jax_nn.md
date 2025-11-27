---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Neural Network Regression with JAX and Optax

**Prepared for the IMF Computational Economics Workshop (Dec 2025)**

**Author:** [John Stachurski](https://johnstachurski.net)

+++

# Introduction


In this lecture, we implement nonlinear regression using neural networks.

We'll write our code using JAX.

The objective is to understand the nuts and bolts of ANNs, as
well as to explore more features of JAX.

The lecture proceeds in three stages:

1. We solve the problem using Keras, to give ourselves a benchmark.  
1. We solve the same problem in pure JAX, using pytree operations and gradient descent.  
1. We solve the same problem using a combination of JAX and [Optax](https://optax.readthedocs.io/en/latest/index.html), an optimization library build for JAX.  


We begin with imports and installs.

```{code-cell} ipython3
:hide-output: false

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from time import time
```

```{code-cell} ipython3
:hide-output: false

#!pip install keras  # Uncomment if not installed
```

```{code-cell} ipython3
:hide-output: false

#!pip install optax  # Uncomment if not installed
```

```{code-cell} ipython3
:hide-output: false

os.environ['KERAS_BACKEND'] = 'jax'
```

(Without setting the backend to JAX, the imports below might fail – unless you have PyTorch or Tensorflow set up. If you have problems running the next cell in Jupyter, try quitting, running export KERAS_BACKEND="jax" and then starting Jupyter on the command line from the same terminal.)

```{code-cell} ipython3
:hide-output: false

import keras
from keras import Sequential
from keras.layers import Dense
import optax
```

## Set Up

Let’s set some of the learning-related constants we’ll use across all
implementations.

```{code-cell} ipython3
:hide-output: false

from typing import NamedTuple

class Config(NamedTuple):
    epochs: int = 4000           # Number of passes through the data set
    data_size: int = 400         # Sample size
    num_layers: int = 4          # Depth of the network
    output_dim: int = 10         # Output dimension of input and hidden layers
    learning_rate: float = 0.001   # Learning rate for gradient descent
```

The next piece of code is repeated from our Keras lecture and generates the data.

```{code-cell} ipython3
:hide-output: false

def generate_data(
        key: jax.Array,         # JAX random key
        config: Config,         # contains configuration data
        x_min: float = 0,       # Minimum x value
        x_max: float = 5        # Maximum x value
    ):
    """
    Generate synthetic regression data.
    Pure functional version using JAX random keys.
    """
    x = jnp.linspace(x_min, x_max, num=config.data_size)
    ϵ = 0.2 * jax.random.normal(key, shape=(config.data_size,))
    y = x**0.5 + jnp.sin(x) + ϵ
    # Return observations as column vectors
    x = jnp.reshape(x, (config.data_size, 1))
    y = jnp.reshape(y, (config.data_size, 1))
    return x, y
```

## Training with Keras

We repeat the Keras training exercise from our earlier Keras lecture as a benchmark.

The code is essentially the same, although written slightly more succinctly.

Here is a function to build the model.

```{code-cell} ipython3
:hide-output: false

def build_keras_model(config: Config,              # contains configuration data
                      activation_function: str = 'tanh'):
    model = Sequential()
    # Add layers to the network sequentially, from inputs towards outputs
    for i in range(config.num_layers-1):
        model.add(
           Dense(units=config.output_dim,
                 activation=activation_function)
           )
    # Add a final layer that maps to a scalar value, for regression.
    model.add(Dense(units=1))
    # Embed training configurations
    model.compile(
        optimizer=keras.optimizers.SGD(),
        loss='mean_squared_error'
    )
    return model
```

Here is a function to train the model.

```{code-cell} ipython3
:hide-output: false

def train_keras_model(
        model, x, y, x_validate, y_validate, config: Config):  # contains configuration data
    print(f"Training NN using Keras.")
    start_time = time()
    training_history = model.fit(
        x, y,
        batch_size=max(x.shape),
        verbose=0,
        epochs=config.epochs,
        validation_data=(x_validate, y_validate)
    )
    elapsed = time() - start_time
    mse = model.evaluate(x_validate, y_validate, verbose=2)
    print(f"Trained Keras model in {elapsed:.2f} seconds with final MSE on validation data = {mse}")
    return model, training_history
```

The next function visualizes the prediction.

```{code-cell} ipython3
:hide-output: false

def plot_keras_output(model, x, y, x_validate, y_validate):
    y_predict = model.predict(x_validate, verbose=2)
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, y_predict, label="fitted model", color='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()
```

Let's run the Keras training:

```{code-cell} ipython3
:hide-output: false

config = Config()
model = build_keras_model(config)
key = jax.random.PRNGKey(1234)
key, subkey1, subkey2 = jax.random.split(key, 3)
x, y = generate_data(subkey1, config)
x_validate, y_validate = generate_data(subkey2, config)
model, training_history = train_keras_model(
        model, x, y, x_validate, y_validate, config
)
plot_keras_output(model, x, y, x_validate, y_validate)
```

We’ve seen this figure before and we note the relatively low final MSE.

+++

## Training with JAX

For the JAX implementation, we need to construct the network ourselves, as a map
from inputs to outputs.

We’ll use the same network structure we used for the Keras implementation.

+++

### Background and set up

The neural network as the form

$$
f(\theta, x) 
    = (A_3 \circ \sigma \circ A_2 \circ \sigma \circ A_1 \circ \sigma \circ A_0)(x)
$$

Here

- $ x $ is a scalar input – a point on the horizontal axis in the Keras estimation above,  
- $ \circ $ means composition of maps,  
- $ \sigma $ is the activation function – in our case, $ \tanh $, and  
- $ A_i $ represents the affine map $ A_i x = W_i x + b_i $.  


Each matrix $ W_i $ is called a **weight matrix** and each vector $ b_i $ is called **bias** term.

The symbol $ \theta $ represents the entire collection of parameters:

$$
\theta = (W_0, b_0, W_1, b_1, W_2, b_2, W_3, b_3)
$$

In fact, when we implement the affine map $ A_i x = W_i x + b_i $, we will work
with row vectors rather than column vectors, so that

- $ x $ and $ b_i $ are stored as row vectors, and  
- the mapping is executed by JAX via the expression `x @ W + b`.  


We work with row vectors because Python numerical operations are row-major rather than column-major, so that row-based operations tend to be more efficient.

Here’s a function to initialize parameters.

The parameter “vector” `θ`  will be stored as a list of dicts.

```{code-cell} ipython3
:hide-output: false

def initialize_params(
        key: jax.Array,     # JAX random key
        config: Config      # contains configuration data
    ):
    """
    Generate an initial parameterization for a feed forward neural network.
    Pure functional version using JAX random keys.
    """
    k = config.output_dim
    shapes = (
        (1, k),  # W_0.shape
        (k, k),  # W_1.shape
        (k, k),  # W_2.shape
        (k, 1)   # W_3.shape
    )
    # A function to generate weight matrices using JAX random
    def w_init(key, m, n):
        return jax.random.normal(key, shape=(m, n)) * jnp.sqrt(2 / m)
    # Build list of dicts, each containing a (weight, bias) pair
    θ = []
    for w_shape in shapes:
        m, n = w_shape
        # Split key for each layer to ensure different initialization
        key, subkey = jax.random.split(key)
        θ.append(dict(W=w_init(subkey, m, n), b=jnp.ones((1, n))) )
    return θ
```

Wait, you say!

Shouldn’t we concatenate the elements of $ \theta $ into some kind of big array, so that we can do autodiff with respect to this array?

Actually we don’t need to, as will become clear below.

+++

### Coding the network

Here’s our implementation of $ f $:

```{code-cell} ipython3
:hide-output: false

@jax.jit
def f(
        θ: list,                        # Network parameters (pytree)
        x: jnp.ndarray,                 # Input data (row vector)
        σ: callable = jnp.tanh          # Activation function
    ):
    """
    Perform a forward pass over the network to evaluate f(θ, x).
    """
    *hidden, last = θ
    for layer in hidden:
        W, b = layer['W'], layer['b']
        x = σ(x @ W + b)
    W, b = last['W'], last['b']
    x = x @ W + b
    return x 
```

The function $ f $ is appropriately vectorized, so that we can pass in the entire
set of input observations as `x` and return the predicted vector of outputs `y_hat = f(θ, x)`
corresponding  to each data point.

The loss function is mean squared error, the same as the Keras case.

```{code-cell} ipython3
:hide-output: false

@jax.jit
def loss_fn(
        θ: list,            # Network parameters (pytree)
        x: jnp.ndarray,     # Input data
        y: jnp.ndarray      # Target data
    ):
    "Loss is mean squared error."
    return jnp.mean((f(θ, x) - y)**2)
```

We’ll use its gradient to do stochastic gradient descent.

(Technically, we will be doing gradient descent, rather than stochastic
gradient descent, since will not randomize over sample points when we
evaluate the gradient.)

The gradient below is with respect to the first argument `θ`.

```{code-cell} ipython3
:hide-output: false

loss_gradient = jax.jit(jax.grad(loss_fn))
```

The line above seems kind of magical, since we are differentiating with respect
to a parameter “vector” stored as a list of dictionaries containing arrays.

How can we differentiate with respect to such a complex object?

The answer is that the list of dictionaries is treated internally as a
[pytree](https://docs.jax.dev/en/latest/pytrees.html).

The JAX function `grad` understands how to

1. extract the individual arrays (the ``leaves’’ of the tree),  
1. compute derivatives with respect to each one, and  
1. pack the resulting derivatives into a pytree with the same structure as the parameter vector.

+++

### Gradient descent

Using the above code, we can now write our rule for updating the parameters via gradient descent, which is the
algorithm we covered in our [lecture on autodiff](https://jax.quantecon.org/autodiff.html).

In this case, however, to keep things as simple as possible, we’ll use a fixed learning rate for every iteration.

```{code-cell} ipython3
:hide-output: false

@jax.jit
def update_parameters(
        θ: list,            # Current parameters (pytree)
        x: jnp.ndarray,     # Input data
        y: jnp.ndarray,     # Target data
        config: Config      # contains configuration data
    ):
    """
    Update parameters using gradient descent.
    Pure function - returns new parameters without mutation.
    """
    gradient = loss_gradient(θ, x, y)
    θ_new = jax.tree.map(lambda p, g: p - config.learning_rate * g, θ, gradient)
    return θ_new
```

+++ {"hide-output": false}

We are implementing the gradient descent update

    new_params = current_params - learning_rate * gradient_of_loss_function

+++

This is nontrivial for a complex structure such as a neural network, so how is
it done?

The key line in the function above is 

    Θ = jax.tree.map(lambda p, g: p - λ * g, θ, gradient)

The `jax.tree.map` function understands `θ` and `gradient` as pytrees of the
same structure and executes `p - λ * g` on the corresponding leaves of the pair
of trees.

This means that each weight matrix and bias vector is updated by gradient
descent, exactly as required.

Here’s code that puts this all together.

```{code-cell} ipython3
:hide-output: false

def train_jax_model(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        x_validate: jnp.ndarray,    # Validation input data
        y_validate: jnp.ndarray,    # Validation target data
        config: Config              # contains configuration data
    ):
    """
    Train model using gradient descent via JAX autodiff.
    Pure functional version using jax.lax.scan.
    """
    def train_step(carry, _):
        θ, train_losses, val_losses = carry
        # Record losses
        train_loss = loss_fn(θ, x, y)
        val_loss = loss_fn(θ, x_validate, y_validate)
        # Update parameters
        θ_new = update_parameters(θ, x, y, config)
        return (θ_new, train_losses, val_losses), (train_loss, val_loss)

    # Initialize with empty arrays
    init_carry = (θ, None, None)
    (θ_final, _, _), (training_loss, validation_loss) = jax.lax.scan(
        train_step, init_carry, None, length=config.epochs
    )

    return θ_final, training_loss, validation_loss
```

### Execution

Let’s run our code and see how it goes.

```{code-cell} ipython3
:hide-output: false

config = Config()
key = jax.random.PRNGKey(1234)
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
θ = initialize_params(subkey1, config)
x, y = generate_data(subkey2, config)
x_validate, y_validate = generate_data(subkey3, config)
```

```{code-cell} ipython3
:hide-output: false

start_time = time()

# Create custom config with different epochs and learning_rate
train_config = Config(epochs=8000, learning_rate=0.01)
θ, training_loss, validation_loss = train_jax_model(
    θ, x, y, x_validate, y_validate, train_config
)

elapsed = time() - start_time
print(f"Trained model with JAX in {elapsed:.2f} seconds.")
```

This figure shows MSE across iterations:

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.plot(range(len(validation_loss)), validation_loss, label='validation loss')
ax.legend()
plt.show()
```

Let’s check the final MSE on the validation data, at the estimated parameters.

```{code-cell} ipython3
:hide-output: false

print(f"""
Final MSE on test data set = {loss_fn(θ, x_validate, y_validate)}.
"""
)
```

This MSE is not as low as we got for Keras, but we did quite well given how
simple our implementation is.

Here’s a visualization of the quality of our fit.

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x.flatten(), f(θ, x).flatten(), 
        label="fitted model", color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

## JAX plus Optax

Our hand-coded optimization routine above was quite effective, but in practice
we might wish to use an optimization library written for JAX.

One such library is [Optax](https://optax.readthedocs.io/en/latest/).

+++

### Optax with SGD

Here’s a training routine using Optax’s stochastic gradient descent solver.

```{code-cell} ipython3
:hide-output: false

def train_jax_optax(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        epochs: int = 4000,         # Number of training epochs
        learning_rate: float = 0.001  # Learning rate for optimizer
    ):
    """
    Train model using Optax SGD optimizer.
    Pure functional version using jax.lax.scan.
    """
    solver = optax.sgd(learning_rate)
    opt_state = solver.init(θ)

    def train_step(carry, _):
        θ, opt_state = carry
        grad = loss_gradient(θ, x, y)
        updates, opt_state_new = solver.update(grad, opt_state, θ)
        θ_new = optax.apply_updates(θ, updates)
        return (θ_new, opt_state_new), None

    (θ_final, _), _ = jax.lax.scan(train_step, (θ, opt_state), None, length=epochs)
    return θ_final
```

Let’s try running it.

```{code-cell} ipython3
:hide-output: false

# Reset parameter vector
key, subkey = jax.random.split(key)
θ = initialize_params(subkey, config)
# Train network

start_time = time()

θ = train_jax_optax(θ, x, y)

elapsed = time() - start_time
print(f"Trained model with JAX and Optax in {elapsed:.2f} seconds.")
```

The resulting MSE is the same as our hand-coded routine.

```{code-cell} ipython3
:hide-output: false

print(f"""
Final MSE on test data set = {loss_fn(θ, x_validate, y_validate)}.
"""
)
```

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x.flatten(), f(θ, x).flatten(), 
        label="fitted model", color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

### Optax with ADAM

We can also consider using a slightly more sophisticated gradient-based method,
such as [ADAM](https://arxiv.org/pdf/1412.6980).

You will notice that the syntax for using this alternative optimizer is very
similar.

```{code-cell} ipython3
:hide-output: false

def train_jax_optax_adam(
        θ: list,                    # Initial parameters (pytree)
        x: jnp.ndarray,             # Training input data
        y: jnp.ndarray,             # Training target data
        epochs: int = 4000,         # Number of training epochs
        learning_rate: float = 0.001  # Learning rate for optimizer
    ):
    """
    Train model using Optax ADAM optimizer.
    Pure functional version using jax.lax.scan.
    """
    solver = optax.adam(learning_rate)
    opt_state = solver.init(θ)

    def train_step(carry, _):
        θ, opt_state = carry
        grad = loss_gradient(θ, x, y)
        updates, opt_state_new = solver.update(grad, opt_state, θ)
        θ_new = optax.apply_updates(θ, updates)
        return (θ_new, opt_state_new), None

    (θ_final, _), _ = jax.lax.scan(train_step, (θ, opt_state), None, length=epochs)
    return θ_final
```

```{code-cell} ipython3
:hide-output: false

# Reset parameter vector
key, subkey = jax.random.split(key)
θ = initialize_params(subkey, config)
# Train network
start_time = time()

θ = train_jax_optax_adam(θ, x, y)

elapsed = time() - start_time
print(f"Trained model with Optax and ADAM in {elapsed:.2f} seconds.")
```

Here’s the MSE.

```{code-cell} ipython3
:hide-output: false

print(f"""
Final MSE on test data set = {loss_fn(θ, x_validate, y_validate)}.
"""
)
```

Here’s a visualization of the result.

```{code-cell} ipython3
:hide-output: false

fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x.flatten(), f(θ, x).flatten(), 
        label="fitted model", color='black')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
