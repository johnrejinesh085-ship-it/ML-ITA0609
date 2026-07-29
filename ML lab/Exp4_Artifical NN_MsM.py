# Program 4: Artificial Neural Network using Backpropagation
# XOR Problem

import numpy as np

# Input Dataset (XOR)
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Target Output
y = np.array([[0],
              [1],
              [1],
              [0]])

# Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and bias
np.random.seed(1)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

# Weights
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Bias
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Learning Rate
learning_rate = 0.5

# Training
epochs = 10000

for epoch in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(final_input)

    # Error
    error = y - predicted_output

    # Backpropagation
    d_predicted = error * sigmoid_derivative(predicted_output)

    error_hidden = d_predicted.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update Weights
    weights_hidden_output += hidden_output.T.dot(d_predicted) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

    # Update Bias
    bias_output += np.sum(d_predicted, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Output
print("Predicted Output after Training:")
print(np.round(predicted_output, 3))
