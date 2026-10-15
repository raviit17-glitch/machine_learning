# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset
# X has two independent variables (e.g., study hours and sleep hours)
X = np.array([[5, 7],
              [8, 8],
              [6, 9],
              [7, 6],
              [9, 7]])   # Independent variables
y = np.array([75, 90, 80, 70, 88])        # Dependent variable (e.g., exam score)

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Print coefficients
print("Coefficients (slopes):", model.coef_)
print("Intercept:", model.intercept_)

# Compare actual vs predicted
for actual, predicted in zip(y, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")