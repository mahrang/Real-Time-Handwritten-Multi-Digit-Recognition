Real-Time Handwritten Multi-Digit Recognition

Code uses OpenCV v. 4.1.0

Built a feedforward neural network (FNN) to classify MNIST dataset using Keras.  
Can classify digits even when rotated up to 45º.

First build the FNN in "CMPE 297 train summary save.ipynb".  
98.13% accuracy was achieved on classifying images in test dataset.
In "CMPE 297 train summary save.ipynb", you will save the FNN model you have built as "mnist.h5" or "model.h5".  
Both of these files are provided.
You can then load either "mnist.h5" or "model.h5" in "CMPE 297 Project 1 Code Submitted.py" or "video_test.py" to classify digits in images, videos, or in real-time.
Photos of rotated hand-written digits detected are provided in the "Digits Detected" folder.
