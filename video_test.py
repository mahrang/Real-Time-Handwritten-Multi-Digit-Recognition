from keras.models import load_model
import cv2
import numpy as np
from PIL import Image

def predict(roi):       
        img = np.resize(roi, (28,28,1))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,28*28)
        im2arr = im2arr.astype('float32')/255
        y_pred = model.predict_classes(im2arr)
        return y_pred

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

model = load_model('mnist.h5')
        
model.compile(optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

# Load an color image in grayscale

ans = input("Cam or Video: ")
if(ans == "Cam"):
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture('video3.mp4')

while(1):

    _, frame = cap.read()

    gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #res = cv2.resize(gry,None,fx=1/5,fy=1/5,interpolation = cv2.INTER_CUBIC)
    #shape = res.shape

    # Blurring using Gaussian filtering
    blur = cv2.GaussianBlur(gry, (5, 5), 0)

    # Binarization --- this line is from Tanny's program
    #edge = cv2.Canny(blur,70,70)
    
    ret, thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, cnt in enumerate (contours):
        area = cv2.contourArea(cnt)
        if(area > 250):
            x1 = True
            x2 = True

            x,y,w,h = cv2.boundingRect(cnt)

            # Stage II:
            # Aspect ratio
            aspect_ratio = float(w)/h
            if(aspect_ratio > 1.5):
                x1 = False

            # Orientation
            if(cnt.shape[0] > 4):
                (xo,yo),(MA,ma),angle = cv2.fitEllipse(cnt)
                if(angle > 15 and angle < 160):
                    x2 = False      

            if(x1 or x2  == True):
                #Stage II: Bouding rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,100),2)
                roi = gry[y-5:y+h+5, x-5:x+w+5]
                
                roi = cv2.GaussianBlur(roi, (5, 5), 0)
                
                roi = cv2.bitwise_not(roi)

                _, roi = cv2.threshold(roi,127,255,cv2.THRESH_BINARY)
                #roiRes = cv2.resize(roi,(28,28),interpolation = cv2.INTER_AREA)
                #cv2.imwrite('ROI' + str(i) + '.png',roi)
                digit = predict(roi)
                digit = digit[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,str(digit),(x - 10,y - 10), font, 3,(255,255,255),2,cv2.LINE_AA)
                
    cv2.imshow('CAM',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:         # wait for ESC key to exit
        break

cv2.destroyAllWindows()
