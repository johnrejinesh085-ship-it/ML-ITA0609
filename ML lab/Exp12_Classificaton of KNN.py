import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Different dataset: Wine Quality
data = {
    'Fixed_Acidity': [7.4, 7.8, 6.8, 7.1, 7.5, 8.2, 6.9, 7.3, 8.0, 6.5,
                      7.2, 7.9, 6.7, 8.1, 7.6],
    'Volatile_Acidity': [0.70, 0.88, 0.65, 0.72, 0.60, 0.55, 0.75, 0.68,
                         0.62, 0.58, 0.70, 0.65, 0.80, 0.52, 0.66],
    'Citric_Acid': [0.00, 0.00, 0.02, 0.01, 0.12, 0.30, 0.03, 0.05,
                    0.10, 0.08, 0.00, 0.04, 0.01, 0.32, 0.15],
    'Residual_Sugar': [1.9, 2.6, 2.1, 2.3, 2.0, 2.5, 2.2, 2.4,
                       2.1, 2.0, 2.3, 2.5, 2.1, 2.8, 2.2],
    'Alcohol': [9.4, 9.8, 9.6, 10.0, 10.5, 11.5, 9.5, 10.2,
                11.0, 10.8, 9.7, 10.6, 9.3, 11.8, 10.4],
    'Quality': ['Bad', 'Bad', 'Bad', 'Bad', 'Good',
                'Good', 'Bad', 'Good', 'Good', 'Good',
                'Bad', 'Good', 'Bad', 'Good', 'Good']
}

df = pd.DataFrame(data)

# Input and output
X = df.drop('Quality', axis=1)
y = df['Quality']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=3)

# Train model
knn.fit(X_train, y_train)

# Prediction
y_pred = knn.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test.values)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
