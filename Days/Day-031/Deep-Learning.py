# 🚀 Day 31/100 of #100DaysOfCode
# 🎯 Deep Learning Intro: Perceptron, Backpropagation 

import numpy as np
import matplotlib.pyplot as plt

# Perceptron class implementation
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=20):
        # Random weight initialization
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate
        self.epochs = epochs

    def activation(self, x):
        # Step function as activation
        return np.where(x >= 0, 1, 0)
    
    def forward(self, X):
        # Weighted sum plus bias
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y):
        errors = []
        for epoch in range(self.epochs):
            total_error = 0
            for xi, target in zip(X, y):
                # Forward pass
                linear_output = self.forward(xi)
                y_pred = self.activation(linear_output)

                # Error calculation
                error = target - y_pred

                # Backward pass (weight update)
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

                total_error += abs(error)
            errors.append(total_error)
            print(f"Epoch {epoch+1}/{self.epochs}, Total Error: {total_error}")
        return errors

    def predict(self, X):
        linear_output = self.forward(X)
        return self.activation(linear_output)


# Training data (Logic Gate Example)
# Here we train perceptron to learn an OR gate
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 1])  # OR gate output

# Train Perceptron
model = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
errors = model.train(X, y)

# Plotting error reduction
plt.plot(range(1, len(errors) + 1), errors, marker='o')
plt.title("Training Error per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Total Error")
plt.show()

# Predictions
print("Final Weights:", model.weights)
print("Final Bias:", model.bias)
print("Predictions:")
for sample in X:
    print(sample, "->", model.predict(sample))
