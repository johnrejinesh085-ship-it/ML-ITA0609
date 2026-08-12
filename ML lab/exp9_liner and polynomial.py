# Linear vs Polynomial Regression
# No pip, NumPy, Matplotlib or Scikit-learn required

# Dataset
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [3, 5, 7, 10, 15, 22, 32, 45, 60, 78]

n = len(x)

# -------------------------------
# Linear Regression
# y = a + bx
# -------------------------------

sum_x = sum(x)
sum_y = sum(y)
sum_xy = sum(x[i] * y[i] for i in range(n))
sum_x2 = sum(x[i] ** 2 for i in range(n))

b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
a = (sum_y - b * sum_x) / n

linear_predictions = [a + b * value for value in x]

# -------------------------------
# Polynomial Regression
# y = a + bx + cx²
# -------------------------------

# Calculate required sums
Sx = sum(x)
Sx2 = sum(value ** 2 for value in x)
Sx3 = sum(value ** 3 for value in x)
Sx4 = sum(value ** 4 for value in x)

Sy = sum(y)
Sxy = sum(x[i] * y[i] for i in range(n))
Sx2y = sum((x[i] ** 2) * y[i] for i in range(n))

# Gaussian elimination
A = [
    [n, Sx, Sx2, Sy],
    [Sx, Sx2, Sx3, Sxy],
    [Sx2, Sx3, Sx4, Sx2y]
]

for i in range(3):
    pivot = A[i][i]

    for j in range(i, 4):
        A[i][j] = A[i][j] / pivot

    for k in range(3):
        if k != i:
            factor = A[k][i]

            for j in range(i, 4):
                A[k][j] = A[k][j] - factor * A[i][j]

a2 = A[0][3]
b2 = A[1][3]
c2 = A[2][3]

polynomial_predictions = [
    a2 + b2 * value + c2 * value ** 2
    for value in x
]

# -------------------------------
# R² Calculation
# -------------------------------

mean_y = sum(y) / n

total_sum = sum((value - mean_y) ** 2 for value in y)

linear_error = sum(
    (y[i] - linear_predictions[i]) ** 2
    for i in range(n)
)

polynomial_error = sum(
    (y[i] - polynomial_predictions[i]) ** 2
    for i in range(n)
)

linear_r2 = 1 - (linear_error / total_sum)
polynomial_r2 = 1 - (polynomial_error / total_sum)

# -------------------------------
# Display Results
# -------------------------------

print("===== LINEAR REGRESSION =====")
print("Equation: y =", round(a, 3), "+", round(b, 3), "x")
print("R² Score:", round(linear_r2, 4))

print("\n===== POLYNOMIAL REGRESSION =====")
print("Equation: y =", round(a2, 3),
      "+", round(b2, 3), "x +",
      round(c2, 3), "x²")
print("R² Score:", round(polynomial_r2, 4))

# -------------------------------
# Prediction
# -------------------------------

new_x = 11

linear_result = a + b * new_x
polynomial_result = a2 + b2 * new_x + c2 * new_x ** 2

print("\n===== PREDICTION FOR X = 11 =====")
print("Linear Regression Prediction:",
      round(linear_result, 2))

print("Polynomial Regression Prediction:",
      round(polynomial_result, 2))

# -------------------------------
# Comparison
# -------------------------------

print("\n===== COMPARISON =====")

if polynomial_r2 > linear_r2:
    print("Polynomial Regression performs better.")
else:
    print("Linear Regression performs better.")

