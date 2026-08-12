import numpy as np

# Dataset
X = np.array([1, 2, 3, 8, 9, 10], dtype=float)

# Number of clusters
k = 2

# Initial parameters
means = np.array([2.0, 9.0])
variances = np.array([1.0, 1.0])
weights = np.array([0.5, 0.5])

# Gaussian probability density function
def gaussian(x, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * \
           np.exp(-((x - mean) ** 2) / (2 * variance))

# EM iterations
for iteration in range(10):

    # ---------------- E-STEP ----------------
    responsibilities = np.zeros((len(X), k))

    for i in range(len(X)):
        for j in range(k):
            responsibilities[i][j] = weights[j] * gaussian(
                X[i], means[j], variances[j]
            )

        # Normalize responsibilities
        responsibilities[i] /= np.sum(responsibilities[i])

    # ---------------- M-STEP ----------------
    for j in range(k):

        # Total responsibility for cluster j
        N_j = np.sum(responsibilities[:, j])

        # Update mean
        means[j] = np.sum(responsibilities[:, j] * X) / N_j

        # Update variance
        variances[j] = np.sum(
            responsibilities[:, j] * (X - means[j]) ** 2
        ) / N_j

        # Update weight
        weights[j] = N_j / len(X)

    print("Iteration", iteration + 1)
    print("Means:", means)
    print("Variances:", variances)
    print("Weights:", weights)
    print()

# Final cluster assignment
clusters = np.argmax(responsibilities, axis=1)

print("Final Cluster Assignment:")
for i in range(len(X)):
    print("Data:", X[i], "-> Cluster:", clusters[i] + 1)
