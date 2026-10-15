from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()

print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)
print("First 5 rows of data:\n", iris.data[:5])
print("First 5 target labels:", iris.target[:5])