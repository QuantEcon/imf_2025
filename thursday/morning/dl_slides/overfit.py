import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data (noisy sine curve)
n_samples = 30
X = np.sort(np.random.uniform(0, 1, n_samples))[:, np.newaxis]
y = np.sin(2 * np.pi * X.ravel()) + np.random.normal(0, 0.1, n_samples)

# Split into training and test sets randomly (mixed along x-axis)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Create dense grid for visualization
X_plot = np.linspace(0, 1, 1000)[:, np.newaxis]

# Create and plot models with different polynomial degrees
plt.figure(figsize=(14, 10))
degrees = [1, 3, 6, 12]  # Different polynomial degrees

for i, degree in enumerate(degrees):
    # Create the model
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_plot = model.predict(X_plot)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    
    # Plot the results
    plt.subplot(2, 2, i+1)
    plt.scatter(X_train, y_train, color='blue', s=30, alpha=0.8, label='training data')
    plt.scatter(X_test, y_test, color='red', s=30, alpha=0.8, label='test data')
    plt.plot(X_plot, y_plot, color='green', lw=2, label=f'polynomial (degree={degree})')
    
    # Plot the true function
    true_y = np.sin(2 * np.pi * X_plot.ravel())
    plt.plot(X_plot, true_y, color='black', lw=1, linestyle='--', label='true function')
    
    plt.title(f'train MSE: {train_error:.4f}, test MSE: {test_error:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-1.5, 1.5)
    plt.legend(loc='best', frameon=False)

plt.tight_layout()
plt.savefig('overfit.pdf', dpi=300, bbox_inches='tight')
plt.show()
