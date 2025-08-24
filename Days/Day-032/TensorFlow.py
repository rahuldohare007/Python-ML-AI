# 🚀 Day 32/100 of #100DaysOfCode
# 🎯 TensorFlow/Keras Setup + First Neural Net

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

# Generate some dummy data
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = (np.sum(X, axis=1) > 1.5).astype(int)  # simple threshold to create binary labels

# Define a simple Sequential model (modern style)
model = Sequential([
    Input(shape=(3,)),                 # explicitly define input shape
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=20, batch_size=8, verbose=1)

# Evaluate the model
loss, acc = model.evaluate(X, y, verbose=0)
print(f"Training Accuracy: {acc*100:.2f}%")
