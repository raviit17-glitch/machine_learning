# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])   # Independent variable
y = np.array([2, 4, 5, 4, 5])             # Dependent variable

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Print coefficients
print("Slope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)

# Plot results
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, y_pred, color="red", label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()