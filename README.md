# Real-Time Handwritten Multi-Digit Recognition

Code uses OpenCV v. 4.1.0

Built a feedforward neural network (FNN) to classify MNIST dataset using Keras. 

98.13% accuracy was achieved on classifying images in test dataset.

FNN has 1 hidden layer.
It can classify digits even when rotated up to 45º, although accuracy decreases as rotation increases.

First build the FNN in "CMPE 297 train summary save.ipynb".  

In "CMPE 297 train summary save.ipynb", you will save the FNN model you have built as "mnist.h5" or "model.h5".  
Both of these files are provided.

You can then load either "mnist.h5" or "model.h5" in "CMPE 297 Project 1 Code Submitted.py" or "video_test.py" to classify digits in images, videos, or in real-time.

Sample of digits detected shown below.  Accuracy decreases when digits are rotated.

![WX20190502-172401](https://github.com/user-attachments/assets/7b6fd529-dcd7-49a6-a3b9-5e34aa184ee5)
![WX20190502-172332](https://github.com/user-attachments/assets/d1bf09bf-f1ba-49dc-ab9d-3906e59b9089)
