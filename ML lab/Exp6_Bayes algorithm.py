import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

data = {
    'Age': [25, 30, 35, 40, 45, 50, 55, 60],
    'Income': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
    'Buy': [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Age', 'Income']]
y = df['Buy']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:")
print(accuracy_score(y_test, y_pred))
