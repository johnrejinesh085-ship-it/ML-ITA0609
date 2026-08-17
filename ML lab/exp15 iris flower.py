# IRIS FLOWER CLASSIFICATION USING NAIVE BAYES CLASSIFIER

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ---------------------------------------------------
# STEP 1: CREATE UNIQUE IRIS DATASET
# ---------------------------------------------------

# Features:
# 1. Sepal Length (cm)
# 2. Sepal Width (cm)
# 3. Petal Length (cm)
# 4. Petal Width (cm)
#
# Classes:
# 0 = Setosa
# 1 = Versicolor
# 2 = Virginica

data = np.array([
    # Setosa
    [5.1, 3.5, 1.4, 0.2, 0],
    [4.9, 3.0, 1.4, 0.2, 0],
    [5.0, 3.4, 1.5, 0.2, 0],
    [5.2, 3.6, 1.5, 0.2, 0],
    [4.8, 3.1, 1.6, 0.2, 0],
    [5.4, 3.7, 1.5, 0.2, 0],
    [4.7, 3.2, 1.3, 0.2, 0],
    [5.1, 3.8, 1.6, 0.2, 0],
    [4.6, 3.1, 1.5, 0.2, 0],
    [5.3, 3.7, 1.5, 0.2, 0],

    # Versicolor
    [7.0, 3.2, 4.7, 1.4, 1],
    [6.4, 3.2, 4.5, 1.5, 1],
    [6.9, 3.1, 4.9, 1.5, 1],
    [6.5, 2.8, 4.6, 1.5, 1],
    [6.3, 3.3, 4.7, 1.6, 1],
    [6.6, 2.9, 4.6, 1.3, 1],
    [5.9, 3.0, 4.2, 1.5, 1],
    [6.7, 3.1, 4.4, 1.4, 1],
    [6.2, 2.9, 4.3, 1.3, 1],
    [6.1, 3.0, 4.6, 1.4, 1],

    # Virginica
    [6.3, 3.3, 6.0, 2.5, 2],
    [5.8, 2.7, 5.1, 1.9, 2],
    [7.1, 3.0, 5.9, 2.1, 2],
    [6.3, 2.9, 5.6, 1.8, 2],
    [6.5, 3.0, 5.8, 2.2, 2],
    [7.6, 3.0, 6.6, 2.1, 2],
    [7.3, 2.9, 6.3, 1.8, 2],
    [6.7, 2.5, 5.8, 1.8, 2],
    [7.2, 3.6, 6.1, 2.5, 2],
    [6.5, 3.2, 5.1, 2.0, 2]
])


# ---------------------------------------------------
# STEP 2: SEPARATE INPUT AND OUTPUT
# ---------------------------------------------------

X = data[:, 0:4]      # Flower measurements
y = data[:, 4]        # Flower class


# ---------------------------------------------------
# STEP 3: SPLIT DATASET
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# STEP 4: CREATE NAIVE BAYES CLASSIFIER
# ---------------------------------------------------

model = GaussianNB()


# ---------------------------------------------------
# STEP 5: TRAIN MODEL
# ---------------------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------------------
# STEP 6: PREDICT TEST DATA
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# STEP 7: DISPLAY PREDICTIONS
# ---------------------------------------------------

print("\nIRIS FLOWER CLASSIFICATION")
print("==============================")

print("\nActual Class    Predicted Class")
print("-------------------------------")

class_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

for actual, predicted in zip(y_test, y_pred):
    print(f"{class_names[int(actual)]:12s} {class_names[int(predicted)]}")


# ---------------------------------------------------
# STEP 8: CALCULATE ACCURACY
# ---------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")


# ---------------------------------------------------
# STEP 9: CONFUSION MATRIX
# ---------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print("==============================")

print(cm)


# ---------------------------------------------------
# STEP 10: CLASSIFICATION REPORT
# ---------------------------------------------------

print("\nCLASSIFICATION REPORT")
print("==============================")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Setosa", "Versicolor", "Virginica"]
))


# ---------------------------------------------------
# STEP 11: PREDICT A NEW IRIS FLOWER
# ---------------------------------------------------

# New flower measurements:
# Sepal Length = 6.2 cm
# Sepal Width  = 3.1 cm
# Petal Length = 4.8 cm
# Petal Width  = 1.5 cm

new_flower = np.array([
    [6.2, 3.1, 4.8, 1.5]
])


prediction = model.predict(new_flower)

probability = model.predict_proba(new_flower)


# ---------------------------------------------------
# STEP 12: DISPLAY NEW FLOWER PREDICTION
# ---------------------------------------------------

print("\nNEW FLOWER PREDICTION")
print("==============================")

print("Sepal Length : 6.2 cm")
print("Sepal Width  : 3.1 cm")
print("Petal Length : 4.8 cm")
print("Petal Width  : 1.5 cm")

print("\nPredicted Flower:",
      class_names[int(prediction[0])])


# ---------------------------------------------------
# STEP 13: DISPLAY PROBABILITY
# ---------------------------------------------------

print("\nPREDICTION PROBABILITY")
print("==============================")

print("Setosa     :", round(probability[0][0] * 100, 2), "%")
print("Versicolor :", round(probability[0][1] * 100, 2), "%")
print("Virginica  :", round(probability[0][2] * 100, 2), "%")
