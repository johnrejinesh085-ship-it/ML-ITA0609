# Program: Implementation of K-Nearest Neighbours (K-NN)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)

# Create K-NN Classifier (K = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model
knn.fit(X_train, y_train)

# Predict the test data
y_pred = knn.predict(X_test)

print("Actual Output:")
print(y_test)

print("\nPredicted Output:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy =", accuracy * 100, "%")
