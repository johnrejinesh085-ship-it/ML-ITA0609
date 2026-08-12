import numpy as np

# Credit Score Classification Dataset
# Features: Credit Score, Annual Income (in lakhs)
X = np.array([
    [300, 2],
    [350, 2.5],
    [400, 3],
    [450, 3.5],
    [700, 8],
    [750, 9],
    [800, 10],
    [850, 12]
], dtype=float)

# Number of clusters
k = 2

# Initial means
means = np.array([
    [400, 3],
    [750, 9]
], dtype=float)

# Initial covariance matrices
covariances = np.array([
    [[10000, 0], [0, 1]],
    [[10000, 0], [0, 1]]
], dtype=float)

# Initial weights
weights = np.array([0.5, 0.5])

# Gaussian probability function
def gaussian(x, mean, covariance):
    d = len(x)

    det = np.linalg.det(covariance)
    inv = np.linalg.inv(covariance)

    return (
        1 / np.sqrt((2 * np.pi) ** d * det)
    ) * np.exp(
        -0.5 * np.dot(
            np.dot((x - mean).T, inv),
            (x - mean)
        )
    )

# EM Algorithm
for iteration in range(10):

    # -------- E-STEP --------
    responsibilities = np.zeros((len(X), k))

    for i in range(len(X)):
        for j in range(k):
            responsibilities[i, j] = (
                weights[j] *
                gaussian(X[i], means[j], covariances[j])
            )

        responsibilities[i] /= np.sum(responsibilities[i])

    # -------- M-STEP --------
    for j in range(k):

        N_j = np.sum(responsibilities[:, j])

        # Update mean
        means[j] = np.sum(
            responsibilities[:, j].reshape(-1, 1) * X,
            axis=0
        ) / N_j

        # Update covariance
        diff = X - means[j]

        covariances[j] = (
            np.dot(
                (responsibilities[:, j].reshape(-1, 1) * diff).T,
                diff
            ) / N_j
        )

        # Prevent singular covariance
        covariances[j] += np.eye(2) * 0.0001

        # Update weight
        weights[j] = N_j / len(X)

# Final classification
clusters = np.argmax(responsibilities, axis=1)

print("Final Cluster Means:")
print(means)

print("\nCredit Score Classification:")
for i in range(len(X)):
    if clusters[i] == 0:
        result = "Low Credit Score"
    else:
        result = "High Credit Score"

    print(
        "Credit Score =", X[i][0],
        "Income =", X[i][1],
        "->", result
    )
