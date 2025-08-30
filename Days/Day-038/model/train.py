# 🚀 Day 37/100 of #100DaysOfCode
# 🎯 Image Classifier App using TensorFlow/Keras

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load dataset (example: flowers or cats vs dogs)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = train_datagen.flow_from_directory(
    'dataset_path',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_gen = train_datagen.flow_from_directory(
    'dataset_path',
    target_size=(150,150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Simple CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=5)
model.save('../model/classifier.h5')
