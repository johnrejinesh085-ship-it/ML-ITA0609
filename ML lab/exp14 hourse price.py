# HOUSE PRICE PREDICTION USING MACHINE LEARNING
# Algorithm: Linear Regression

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------
# STEP 1: CREATE DATASET
# ---------------------------------------------------
# Features:
# Area_sqft
# Bedrooms
# Bathrooms
# Age_years
# Parking
#
# Target:
# Price in Lakhs

data = np.array([
    [800,  2, 1, 10, 1, 45],
    [1000, 2, 2, 8,  1, 58],
    [1200, 3, 2, 6,  1, 72],
    [1500, 3, 2, 5,  2, 92],
    [1800, 3, 3, 4,  2, 115],
    [2000, 4, 3, 3,  2, 135],
    [2200, 4, 3, 2,  2, 150],
    [2500, 4, 4, 2,  3, 175],
    [900,  2, 1, 12, 1, 48],
    [1100, 2, 2, 7,  1, 65],
    [1300, 3, 2, 9,  1, 76],
    [1600, 3, 3, 6,  2, 98],
    [1900, 4, 3, 5,  2, 125],
    [2100, 4, 3, 4,  2, 140],
    [2400, 4, 4, 3,  3, 165],
    [2700, 5, 4, 2,  3, 190],
    [850,  2, 1, 15, 1, 42],
    [1250, 3, 2, 10, 1, 70],
    [1750, 3, 3, 7,  2, 108],
    [2300, 4, 3, 6,  2, 145]
])


# ---------------------------------------------------
# STEP 2: SEPARATE INPUT AND OUTPUT
# ---------------------------------------------------

# Input features
X = data[:, 0:5]

# Target price
y = data[:, 5]


# ---------------------------------------------------
# STEP 3: SPLIT DATASET
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------------------------
# STEP 4: CREATE LINEAR REGRESSION MODEL
# ---------------------------------------------------

model = LinearRegression()


# ---------------------------------------------------
# STEP 5: TRAIN MODEL
# ---------------------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------------------
# STEP 6: PREDICT HOUSE PRICES
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# STEP 7: DISPLAY ACTUAL VS PREDICTED
# ---------------------------------------------------

print("\nHOUSE PRICE PREDICTION")
print("==============================")

print("\nActual Price    Predicted Price")
print("------------------------------")

for actual, predicted in zip(y_test, y_pred):
    print(f"{actual:10.2f}       {predicted:10.2f}")


# ---------------------------------------------------
# STEP 8: MODEL EVALUATION
# ---------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\nMODEL PERFORMANCE")
print("==============================")

print("Mean Absolute Error     :", round(mae, 2))

print("Mean Squared Error      :", round(mse, 2))

print("Root Mean Squared Error :", round(rmse, 2))

print("R2 Score                :", round(r2, 2))


# ---------------------------------------------------
# STEP 9: PREDICT PRICE OF A NEW HOUSE
# ---------------------------------------------------

# New house:
# Area = 1500 sq.ft
# Bedrooms = 3
# Bathrooms = 2
# Age = 4 years
# Parking = 2

new_house = np.array([
    [1500, 3, 2, 4, 2]
])

predicted_price = model.predict(new_house)


# ---------------------------------------------------
# STEP 10: DISPLAY NEW HOUSE PREDICTION
# ---------------------------------------------------

print("\nNEW HOUSE DETAILS")
print("==============================")

print("Area       : 1500 sq.ft")
print("Bedrooms   : 3")
print("Bathrooms  : 2")
print("Age        : 4 years")
print("Parking    : 2")

print("\nPredicted House Price:",
      round(predicted_price[0], 2), "Lakhs")


# ---------------------------------------------------
# STEP 11: DISPLAY MODEL COEFFICIENTS
# ---------------------------------------------------

print("\nMODEL COEFFICIENTS")
print("==============================")

features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age",
    "Parking"
]

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature:15s}: {coefficient:.4f}")

print("\nIntercept:", round(model.intercept_, 4))
