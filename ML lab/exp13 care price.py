# CAR PRICE PREDICTION USING LINEAR REGRESSION

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------
# STEP 1: CREATE UNIQUE CAR PRICE DATASET
# ---------------------------------------------------

# Features:
# 1. Engine_CC
# 2. Mileage_km
# 3. Car_Age
# 4. Horsepower
# 5. Owners
# 6. Fuel_Efficiency
#
# Target:
# Price (in Lakhs)

data = np.array([
    [1200, 18000, 2,  85, 1, 20, 5.8],
    [1500, 25000, 3, 105, 1, 18, 7.2],
    [1000, 35000, 4,  70, 2, 22, 4.5],
    [1800, 15000, 1, 140, 1, 16, 10.5],
    [2200, 20000, 2, 170, 1, 14, 13.2],
    [1300, 45000, 5,  90, 2, 19, 5.2],
    [1600, 30000, 3, 120, 1, 17, 8.5],
    [2000, 50000, 6, 150, 3, 13, 7.8],
    [1100, 22000, 2,  75, 1, 21, 5.4],
    [2500, 12000, 1, 190, 1, 12, 15.5],
    [1400, 40000, 4, 100, 2, 18, 6.0],
    [1700, 28000, 3, 130, 1, 16, 9.2],
    [1200, 60000, 7,  80, 3, 20, 3.8],
    [2100, 18000, 2, 160, 1, 14, 12.5],
    [1500, 55000, 6, 110, 3, 17, 5.0],
    [1800, 35000, 4, 145, 2, 15, 9.0],
    [1000, 15000, 1,  72, 1, 23, 5.6],
    [2300, 25000, 2, 180, 1, 13, 13.8],
    [1300, 50000, 5,  95, 2, 19, 5.0],
    [1900, 20000, 2, 155, 1, 15, 11.0]
])


# ---------------------------------------------------
# STEP 2: SEPARATE INPUT AND OUTPUT
# ---------------------------------------------------

# X = Input features
X = data[:, 0:6]

# y = Target price
y = data[:, 6]


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
# STEP 5: TRAIN THE MODEL
# ---------------------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------------------
# STEP 6: PREDICT TEST DATA
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# STEP 7: DISPLAY ACTUAL VS PREDICTED PRICE
# ---------------------------------------------------

print("\nCAR PRICE PREDICTION")
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

print("Mean Absolute Error      :", round(mae, 2))

print("Mean Squared Error       :", round(mse, 2))

print("Root Mean Squared Error  :", round(rmse, 2))

print("R2 Score                 :", round(r2, 2))


# ---------------------------------------------------
# STEP 9: PREDICT PRICE FOR A NEW CAR
# ---------------------------------------------------

# New car details:
# Engine = 1600 CC
# Mileage = 25000 km
# Age = 2 years
# Horsepower = 120 HP
# Owners = 1
# Fuel Efficiency = 18 km/l

new_car = np.array([
    [1600, 25000, 2, 120, 1, 18]
])


predicted_price = model.predict(new_car)


# ---------------------------------------------------
# STEP 10: DISPLAY NEW CAR PREDICTION
# ---------------------------------------------------

print("\nNEW CAR DETAILS")
print("==============================")

print("Engine Capacity    : 1600 CC")
print("Mileage            : 25000 km")
print("Car Age            : 2 years")
print("Horsepower         : 120 HP")
print("Number of Owners   : 1")
print("Fuel Efficiency    : 18 km/l")

print("\nPredicted Car Price:",
      round(predicted_price[0], 2), "Lakhs")


# ---------------------------------------------------
# STEP 11: DISPLAY MODEL COEFFICIENTS
# ---------------------------------------------------

print("\nMODEL COEFFICIENTS")
print("==============================")

features = [
    "Engine CC",
    "Mileage",
    "Car Age",
    "Horsepower",
    "Owners",
    "Fuel Efficiency"
]

for feature, coefficient in zip(features, model.coef_):
    print(f"{feature:20s}: {coefficient:.4f}")

print("\nIntercept:", round(model.intercept_, 4))
