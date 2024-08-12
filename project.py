import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Plot the data
plt.scatter(X, y)
plt.title("Synthetic Data")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Add bias term to X
X_b = np.c_[np.ones((100, 1)), X]

# Calculate the best theta using the Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Optimal parameters (theta):", theta_best)

# Make predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

# Plot the predictions
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.scatter(X, y)
plt.title("Linear Regression Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

# Evaluate the model using MSE
y_pred = X_b.dot(theta_best)
mse = np.mean((y - y_pred) ** 2)
print("Mean Squared Error:", mse)
