# 🚀 Day 34/100 of #100DaysOfCode
# 🎯 Activation Functions & Optimizers  

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# Load and preprocess MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

def build_model(activation='relu'):
    return Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])

activations = ['relu', 'sigmoid', 'tanh']
optimizer_constructors = {
    'SGD': lambda: SGD(learning_rate=0.01),
    'Adam': lambda: Adam(learning_rate=0.001),
    'RMSprop': lambda: RMSprop(learning_rate=0.001)
}

for act in activations:
    for opt_name, opt_fn in optimizer_constructors.items():
        print(f"\nTraining with Activation={act}, Optimizer={opt_name}")
        model = build_model(activation=act)
        optimizer = opt_fn()  # Create a new optimizer every time
        model.compile(optimizer=optimizer, 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=1)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc:.4f}")

