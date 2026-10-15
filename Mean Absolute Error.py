import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])   # Features
y = np.array([2, 4, 6, 8, 10])            # Target values

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Calculate Mean Squared Error
mse = mean_squared_error(y, y_pred)

print("Predictions:", y_pred)
print("Mean Squared Error:", mse)