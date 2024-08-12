By Karthik Muthukumar

In this project I wanted to create my first neural network in which it identifies hand written digits. I will be trying to code through using the TensorFlow library as well as doing the project all over again in PyTorch in order to get proficient with it. For this you will need to install the opencv-python, matplotlib, numpy, and tensorflow packages.

STEPS:
1. Install the required packages into your IDE using the following command in your terminal without the quotations "pip install tensorflow opencv-python numpy matplotlib"
2. Download the project into a common folder and open it in your IDE
3. Draw some single digit numbers ranging from 0 to 9 whether it be through MS Paint or by hand but make sure the size of the image is a 28 by 28 pixel.
4. Put the images in the numbers folder
5. Run the digit_recognition.py file and see if the models guess on your drawing is correct, this will inturn train the model. In the training_recognition.py file you can mess with the number of epochs. After the model is trained and saved, you can comment out the code in training_recognition.py and proceed to add more images if needed.