import numpy as np
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Apply LDA to reduce dimensions to 2
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

print("Original shape:", X.shape)
print("Transformed shape:", X_lda.shape)

# Plot the LDA result
plt.figure(figsize=(8,6))
plt.scatter(X_lda[:,0], X_lda[:,1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA on Iris Dataset")
plt.colorbar(label='Species')
plt.show()