---
jupyter:
  jupytext:
    default_lexer: ipython3
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Advanced Nonlinear Regression with JAX

**Prepared for the IMF Computational Economics Workshop (Dec 2025)**

**Author:** [John Stachurski](https://johnstachurski.net)

## Introduction

Our next task is a challenging nonlinear regression with neural networks using JAX and Optax.

In this task we will face a far more complex function, which cannot be fitted without a significant number of parameters.

More parameters means minimization over a higher-dimensional loss surface, which will force us to  work harder with our optimization procedure.

Thus, our JAX and Optax code will be correspondingly more advanced.

At the same time, you will recognize many of the same core ideas.

We begin with the following imports.

```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from typing import List, Tuple, NamedTuple
from functools import partial
from time import time
```

Let's check our environment.

```python
print(f"Using JAX version: {jax.__version__}")
print(f"Device: {jax.devices()[0]}")
```

## Set Up

The default configuration will have around 21,000 parameters

```python
# Configuration
class Config(NamedTuple):
    # Data parameters
    data_size: int = 4_000
    train_ratio: float = 0.8
    noise_scale: float = 0.25
    # Model parameters
    hidden_layers: List[int] = [128, 128, 32]
    activation: str = "selu"  # Options: "relu", "selu", "tanh", "sigmoid"
    # Training parameters
    batch_size: int = 128
    epochs: int = 20_000
    init_lr: float = 0.001
    min_lr: float = 0.0001
    warmup_steps: int = 100
    decay_steps: int = 300
    regularization_term: float = 1e-5
    # Evaluation
    eval_every: int = 100
```

Here is the function we will try to recover from noisy data.

```python
@jax.jit
def f(x):
    """
    Function to be estimated.
    """
    term1 = 2 * jnp.sin(3 * x) * jnp.cos(x/2)
    term2 = 0.5 * x**2 * jnp.cos(5*x) / (1 + 0.1 * x**2)
    term3 = 3 * jnp.exp(-0.2 * (x - 4)**2) * jnp.sin(10*x)
    term4 = 1.5 * jnp.tanh(x/3) * jnp.sin(7*x)
    term5 = 0.8 * jnp.log(jnp.abs(x) + 1) * jnp.cos(x**2 / 8)
    term6 = jnp.where(x > 0, 2 * jnp.sin(3*x), -2 * jnp.sin(3*x))  # Discontinuity
    return term1 + term2 + term3 + term4 + term5 + term6
```

As you can see, this function is quite complex.

```python
x_grid = jnp.linspace(-10.0, 10.0, 200)
fig, ax = plt.subplots()
y_true = f(x_grid)
ax.plot(x_grid, y_true, 
         color='black', 
         linewidth=2, label='true function')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.legend()
plt.show()
```

We will use the following function to produce noisy observations of $f$.

```python
def generate_data(
        key: jax.Array,
        config: Config      # contains configuration data
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic nonlinear regression data.

    """
    x_key, y_key = jax.random.split(key)
    # x is generated uniformly
    x = jax.random.uniform(
            x_key, (config.data_size, 1), minval=-10.0, maxval=10.0
        )
    # y = f(x) + noise
    σ = config.noise_scale
    w = σ * jax.random.normal(y_key, shape=x.shape)
    y = f(x) + w
    return x, y
```

## Constructing a network


Previously we used a dictionary to store the weights and biases associated with a single layer.

Here we will used a `NamedTuple`, which feels slightly more elegant in scientific work.

```python
class LayerParams(NamedTuple):
    """
    Stores parameters for one layer of the neural network.

    """
    W: jnp.ndarray     # weights
    b: jnp.ndarray     # biases
```

The weights and biases in each layer will be initialized randomly.

We use standard initialization procedures according to the specified activation function.

The next function initializes parameters in a single layer.

```python
def init_layer_params(
        key: jax.Array,
        in_dim: int,
        out_dim: int,
        activation_name: str
    ) -> Tuple[LayerParams, jax.Array]:
    """
    Initialize parameters for a single layer using appropriate initialization
    based on the activation function.

    - He initialization for ReLU and its variants
    - LeCun initialization for SELU
    - Glorot/Xavier initialization for tanh and sigmoid

    """
    key, w_key, b_key = jax.random.split(key, 3)

    # Choose initialization strategy based on activation function
    if activation_name == "selu":
        # LeCun initialization
        s = jnp.sqrt(1.0 / in_dim)
        W = jax.random.normal(w_key, (in_dim, out_dim)) * s
        b = jnp.zeros((out_dim,))
    elif activation_name in ["tanh", "sigmoid"]:
        # Glorot/Xavier initialization
        s = jnp.sqrt(6.0 / (in_dim + out_dim))
        W = jax.random.uniform(w_key, (in_dim, out_dim), minval=-s, maxval=s)
        b = jnp.zeros((out_dim,))
    else:
        # He initialization (default for ReLU and variants)
        s = jnp.sqrt(2.0 / in_dim)
        W = jax.random.normal(w_key, (in_dim, out_dim)) * s
        b = jnp.zeros((out_dim,))

    return LayerParams(W=W, b=b), key
```

Here's a function that uses the preceding logic to construct a Pytree of suitably initialized network parameters.

```python
def initialize_network_params(
        key: jax.Array,
        layer_sizes: List[int],
        activation_name: str
    ) -> List[LayerParams]:
    """
    Initialize all parameters for the network and store them as a list of
    instances of LayerParams (a Pytree).

    """
    θ = []
    # For all layers but the last one
    for i in range(len(layer_sizes) - 1):
        # Generate an instance of LayerParams corresponding to layer i
        layer, key = init_layer_params(
            key,
            layer_sizes[i],      # in dimension for layer
            layer_sizes[i + 1],  # out dimension for layer
            activation_name
        )
        # And append it to the list the contains all network parameters.
        θ.append(layer)

    return θ
```

Here's a jitted function that maps inputs to outputs for a given parameterization of the network.

```python
@partial(jax.jit, static_argnames=['activation'])
def forward(
        θ: List[LayerParams],
        x: jnp.ndarray,
        activation: str
    ) -> jnp.ndarray:

    """
    Forward pass through the neural network.

    Args:
        θ: network parameters
        x: input data
        activation: activation function name (static argument)
    """

    # Select the activation function based on name
    if activation == "relu":
        σ = jax.nn.relu
    elif activation == "selu":
        σ = jax.nn.selu
    elif activation == "tanh":
        σ = jnp.tanh
    elif activation == "gelu":
        σ = jax.nn.gelu
    elif activation == "sigmoid":
        σ = jax.nn.sigmoid
    elif activation == "elu":
        σ = jax.nn.elu
    else:
        # Default to selu
        σ = jax.nn.selu

    # Apply all layers except the last, with activation
    for W, b in θ[:-1]:
        x = σ(x @ W + b)
    # Apply last layer without activation (for regression)
    W, b = θ[-1]
    output = x @ W + b

    return output
```

## Loss


The next function calculates loss associated with a given prediction vector in terms of MSE, conditional on the data set.

```python
@partial(jax.jit, static_argnames=['activation'])
def mse_loss(
        params: List[LayerParams], 
        x: jnp.ndarray,
        y: jnp.ndarray,
        activation: str = "relu"
    ) -> jnp.ndarray:

    """
    Mean squared error loss function.

    """
    y_pred = forward(params, x, activation=activation)
    return jnp.mean((y_pred - y) ** 2)
```

When we compute loss, we will use a small amount of regularization to help prevent us from overfitting the existing data set.

```python
@partial(jax.jit, static_argnames=['activation'])
def regularized_loss(
        params: List[LayerParams],
        x: jnp.ndarray,
        y: jnp.ndarray,
        activation: str,
        λ: float
    ) -> jnp.ndarray:
    """
    Loss function with L2 regularization.

    """
    mse = mse_loss(params, x, y, activation=activation)

    # L2 regularization
    l2_penalty = 0.0
    for layer in params:
        l2_penalty += jnp.sum(layer.W ** 2)

    return mse + λ * l2_penalty
```

## Training Components

In this section we implement some training components that execute key steps
associated with updating the weights.

First we write a function factory that performs a single update of the Pytree
containing all parameters.

The update uses Optax.

```python
def training_step_factory(optimizer, activation: str, regularization_term: float):
    """
    Create a JIT-compiled training step function.

    """

    # Create a specialized loss gradient function for this activation
    loss_grad = jax.grad(lambda p, x, y: regularized_loss(p, x, y, activation=activation, λ=regularization_term))

    @jax.jit
    def train_step(θ, opt_state, x_batch, y_batch):
        """Single training step."""
        grads = loss_grad(θ, x_batch, y_batch)
        loss_val = regularized_loss(θ, x_batch, y_batch, activation=activation, λ=regularization_term)

        updates, new_opt_state = optimizer.update(grads, opt_state, θ)
        θ = optax.apply_updates(θ, updates)

        return θ, new_opt_state, loss_val

    return train_step
```

Now we create an Optax learning rate schedule with warmup and decay.

The role of the schedule is to adjust the learning rate as training progresses.

For details we refer to the Optax documentation.

```python
def create_lr_schedule(config: Config):  # contains configuration data
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.init_lr,
        transition_steps=config.warmup_steps
    )

    decay_fn = optax.exponential_decay(
        init_value=config.init_lr,
        transition_steps=config.decay_steps,
        decay_rate=0.5,
        end_value=config.min_lr
    )

    return optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps]
    )
```

We also produce a data batch iterator, which generates a list containing data
batches.  

Each data batch is a subset of the data set containing matched input-output
pairs.

The collection of batches in the list can be understood as a random partition of
the data set, where each element of the partition has the same size.

```python
def create_data_batch_iterator(
        x: jnp.ndarray, 
        y: jnp.ndarray, 
        key: jax.Array,
        batch_size: int,
    ) -> List[Tuple[jax.Array]]:
    """
    Create a list of batched data.  Each element of the list is a tuple
    (x_batch, y_batch), containing a batch of data for training.

    """
    num_samples = x.shape[0]
    
    # Shuffle the data
    indices = jax.random.permutation(key, jnp.arange(num_samples))
    
    # Create batches
    num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
    
    batches = []
    for i in range(num_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        x_batch = x[batch_indices]
        y_batch = y[batch_indices]
        batches.append((x_batch, y_batch))
    
    return batches
```

## Training

We are ready to train the network.

First we set the seed.

```python
SEED = 42 # Set random seed for reproducibility
key = jax.random.PRNGKey(SEED)
```

Now we produce separate keys for training and validation data.

```python
key, train_data_key, val_data_key = jax.random.split(key, 3)
```

Next we generate training and validation data

```python
config = Config()
print("Generating data...")
x_train, y_train = generate_data(train_data_key, config)
val_config = Config(data_size=int(config.data_size * 0.5))
x_val, y_val = generate_data(val_data_key, val_config)
```

We also define model architecture and activation function.

```python
input_dim = 1  # scalar input
output_dim = 1 # scalar output
layer_sizes = [input_dim] + config.hidden_layers + [output_dim]
activation = config.activation
print(f"Using activation function: {activation}")
```

Let's initialize all the parameters in the network

```python
print(f"Initializing model with layer sizes: {layer_sizes}")
key, subkey = jax.random.split(key)
θ = initialize_network_params(subkey, layer_sizes, activation)
```

Now let's train the network.

Note that we are training a relatively large network and hence the training
process takes a nontrivial amount of time.

```python
# Create optimizer with learning rate schedule
lr_schedule = create_lr_schedule(config)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),  # Gradient clipping for stability
    optax.adam(learning_rate=lr_schedule)
)
opt_state = optimizer.init(θ)

# Create training step function
train_step_fn = training_step_factory(optimizer, activation, config.regularization_term)

# Training loop
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_params = θ
patience_counter = 0
patience = 50   # Early stopping patience (in terms of evaluation intervals)

print(f"Starting training for {config.epochs} epochs...")
start = time()

# One epoch is a complete pass through the data set
for epoch in range(config.epochs):

    # Create shuffled batches for this epoch
    key, subkey = jax.random.split(key)
    batches = create_data_batch_iterator(x_train, y_train, subkey, config.batch_size)

    # Process each batch, updating parameters
    epoch_losses = []
    for x_batch, y_batch in batches:
        θ, opt_state, loss = train_step_fn(θ, opt_state, x_batch,
                                                y_batch)
        epoch_losses.append(loss)

    # Calculate average loss for this epoch
    avg_train_loss = jnp.mean(jnp.array(epoch_losses))
    train_losses.append(avg_train_loss)

    # Evaluate on validation set periodically
    if epoch % config.eval_every == 0 or epoch == config.epochs - 1:

        val_loss = float(mse_loss(θ, x_val, y_val, activation))
        val_losses.append(val_loss)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_theta = jax.tree.map(lambda p: p, θ)  # Copy the params
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

elapsed = time() - start

print(f"Training completed in {elapsed:.2f} seconds.")
print(f"Best validation loss: {best_val_loss:.6f}")
```

Here we plot the MSE curves on training and validation data over epochs.

```python
fig, ax = plt.subplots()
ax.plot(train_losses, label='training Loss')
ax.plot(np.arange(0, len(val_losses) * config.eval_every, config.eval_every),
             val_losses, label='validation Loss')
ax.set_xlabel('epoch')
ax.set_ylabel('MSE Loss')
ax.set_title(f'Learning curves with {config.activation.upper()}')
ax.legend()
plt.show()
```

Finally, let's plot the original and fitted functions.

```python
x_grid = jnp.linspace(-10.0, 10.0, 200)
y_pred = forward(θ, x_grid.reshape(-1, 1), activation=activation)

fig, ax = plt.subplots()
# Plot training data
ax.scatter(x_train.flatten(), y_train.flatten(), 
            alpha=0.2, color='blue', label='training data')

# Plot the predicted curve
ax.plot(x_grid, y_pred.flatten(), 
         color='red', 
         linewidth=2, 
         linestyle='--',
         label='model prediction')

# Plot the true function (without noise)
y_true = f(x_grid)
ax.plot(x_grid, y_true, 
         color='black', ls='--',
         linewidth=2, label='true function')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.legend()

plt.show()
```
