"""
This project aims to identify handwritten digits using a neural network coded on the TensorFlow/Keras python library.
See the readme.md file for instructions on how to use this.
"""

#These are the import statements
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers


mnist = tf.keras.datasets.mnist                            #This is the dataset which contains many images to train and test the model I created
(x_train, y_train), (x_test, y_test) = mnist.load_data()   #Here I create two tuples which are the train and test and I load the dataset into them to be trained and tested on

x_train = keras.utils.normalize(x_train, axis=1)           #Normalizes the x_train to have a Greyscale range from 0 to 1 rather than 0 to 255 (black is 0 and white is 255)
x_test = keras.utils.normalize(x_test, axis=1)             #Normalizes the x_test to have a Greyscale range from 0 to 1 rather than 0 to 255 (black is 0 and white is 255)

x_train = x_train.reshape(-1, 784)                         #Takes the image which is of the size 28 by 28 and flattens it to be a column vector of dimension n=784 to be inputted in the model to be trained
x_test = x_test.reshape(-1, 784)                           #Has the same function of the above comment 



#The below lines of code will train the neural network two layers with the activation function relu where if the signal x is positive then output is x and if x is negative then output is 0
#The final layer of the neural network is the softmax layer which performs the multi-classification 
#The model.compile is essentially the cost function of the neural network as it uses a loss function and uses the adam optimzer to train the model
#The model.fit is the function of the neural network that essentially trains the model by running gradient descent, the epochs is how many times the model is trained
#model.save and model.load saves and load the model


model = keras.Sequential([                                 #Sequential means the layers are initialized in a sequential order
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

model.fit(x_train, y_train, epochs=10)

model.save('digit_recognition_model.keras')



model = keras.models.load_model('digit_recognition_model.keras')

loss, accuracy = model.evaluate(x_test, y_test)

print("Model accuracy:", accuracy)
print("Model loss:",loss)

