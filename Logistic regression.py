# Import libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Sample dataset (two features)
X = np.array([[2, 3],
              [1, 5],
              [2, 8],
              [7, 3],
              [8, 5],
              [9, 6]])   # Independent variables
y = np.array([0, 0, 0, 1, 1, 1])          # Labels (0 or 1)

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Print results
print("Predicted labels:", y_pred)
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Plot decision boundary
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')
x_values = np.linspace(min(X[:,0]), max(X[:,0]), 100)
y_values = -(model.intercept_[0] + model.coef_[0][0]*x_values) / model.coef_[0][1]
plt.plot(x_values, y_values, color='black', label="Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()