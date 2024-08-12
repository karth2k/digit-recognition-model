from training_recognition import model
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import keras


"Run this file after training and saving the model and having images ready to go in the numbers file."

model = keras.models.load_model('digit_recognition_model.keras')

#The code below goes through the numbers folder for images pertaining to the task of identifying the digit listed from 0-9.

image_folder = "numbers"
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder,f))]

for image_file in image_files:
    try:
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"{image_file} is not a valid image.")
            continue
        img = np.invert(np.array([img]))
        img = img.reshape(1, 784)
        pred = model.predict(img)
        print(f"This is the number {np.argmax(pred)}.")
        plt.imshow(img.reshape(28,28), cmap=plt.cm.binary)
        plt.show()
    except:
        print(f"Error processing {image_file}.")
        
