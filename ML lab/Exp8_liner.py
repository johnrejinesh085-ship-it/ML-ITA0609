import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
#House Size (sq.ft)
X = np.array([600, 800, 1000, 1200, 1400, 1600, 1800, 2000]).reshape(-1, 1)

# House Price (in ₹ Lakhs)
y = np.array([18, 24, 30, 36, 42, 48, 54, 60])

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict prices
y_pred = model.predict(X)

# Display model parameters
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict price for a 1500 sq.ft house
new_size = np.array([[1500]])
predicted_price = model.predict(new_size)
print("Predicted Price for 1500 sq.ft:", predicted_price[0], "Lakhs")

#matplotlib
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("House Size (sq.ft)")
plt.ylabel("House Price (Lakhs)")
plt.title("Linear Regression - House Price Prediction")
plt.legend()
plt.show()
