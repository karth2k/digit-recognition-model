import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import keras
from keras import layers

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

x_train = x_train.reshape(-1, 784)  
x_test = x_test.reshape(-1, 784)    

# Uncomment the following lines to train the model
"""
model = keras.Sequential([
    keras.Input(shape=(784, )),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=128, activation='relu'),
    layers.Dense(units=10, activation='softmax')
])

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

model.save('mnist_model.keras')
"""

model = keras.models.load_model('mnist_model.keras')

loss, accuracy = model.evaluate(x_test, y_test)

print("Model accuracy: ", accuracy)
print("Model loss: ",loss)