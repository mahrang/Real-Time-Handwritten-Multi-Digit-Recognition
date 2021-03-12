import numpy as np
import cv2 as cv #code uses OpenCV 4.1.0
import keras

way = input("loading method: ")
print('the video stream is from: ', way)

if way == 'CAM':
    name = 0
elif way == 'FILE':
    name = input('file name: ')
else:
    name = 0
cap = cv.VideoCapture(name)
model = keras.models.load_model('model.h5')

while(True):
    ret, img = cap.read()
    if ret == True:
    ##    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    ##    lower_white = np.array([0,0,221])
    ##    upper_white = np.array([180,30,255])
    ##    lower_black = np.array([0,0,0])
    ##    upper_black = np.array([180,255,46])
    ##    mask = cv.inRange(hsv,lower_white, upper_white)
    ##    mask1 = cv.inRange(hsv,lower_black, upper_black)
    ##    mask = mask
    ##    canvas = np.ones((img.shape[0],img.shape[1],3),np.uint8)
    ##    res = cv.bitwise_and(img,img, mask = mask)
    ##    cv.imshow('res',res)
    ##    gray = cv.cvtColor(res,cv.COLOR_HSV2BGR)
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(gray,(3,3),0)
        ret,thresh = cv.threshold(blur,150,255,cv.THRESH_BINARY_INV) #thresh is binarized image
        #edge = cv.Canny(thresh,70,70)  this line was left uncommented but edge is only used in next 2 commented lines
        ##cnt1,hierarchy = cv.findContours(edge,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        ##IAP = cv.drawContours(edge,cnt1,-1,(255,255,255),3)
        kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
        dilated = cv.dilate(thresh,kernel,iterations = 1)
        #cv.imshow('d',dilated)
        cnt2,hierarchy1 = cv.findContours(dilated,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) #findContours only works on binary 
                                                                                           #images
        
        cnts3 = []
        aspect_ration1 = 0
        for i in range(len(cnt2)):
            x1,x2,x3 = False,False,False
            area = cv.contourArea(cnt2[i])
            if area > 100:
                x3 = True
            if(len(cnt2[i]) >= 5):
                x,y,w,h = cv.boundingRect(cnt2[i])
                aspect_ration1 = float(w)/h
                _,_,angle1 = cv.fitEllipse(cnt2[i])

                #if (2.25 > aspect_ration1 > 1) and not (100 < angle1 < 80):
                #    x1 = True
                    #print(aspect_ration1, angle1)
                #if (0.15 < aspect_ration1 < 0.95) and (80 <= angle1 <= 100):
                #    x2 = True
                #print(aspect_ration1, angle1)
                if 3 > aspect_ration1:
                    x1 = True
                if angle1 > 150 or angle1 < 20:
                    x2 = True
                if x1 and x2 and x3:
                    cnts3.append(cnt2[i])
        count = 0            
        for i in range(len(cnts3)):
            x,y,w,h = cv.boundingRect(cnts3[i])
            ROI = thresh[y:y+h+1,x:x+w+1]
            size = ROI.shape
            if size[0]*size[1] > 0:
                ROI = cv.resize(ROI,(28,28))
                ROI = cv.copyMakeBorder(ROI,10,10,10,10,cv.BORDER_CONSTANT,value=[0,0,0])
                ROI = cv.resize(ROI,(28,28))
##                ROI = cv.resize(ROI,(20,20),interpolation = cv.INTER_CUBIC)
##                #ret, mask = cv.threshold(ROI,150,255,cv.THRESH_BINARY)
##                mask_inv = cv.bitwise_not(ROI)
##                canvas = np.zeros((28,28), np.uint8)
##                roi = canvas[4:24,4:24]
##                img1 = cv.bitwise_and(roi,roi,mask = ROI)
##                img2 = cv.bitwise_and(ROI,ROI, mask = mask_inv)
##                dst = cv.add(img1,img2)
##                canvas[4:24,4:24] = dst
##                canvas = cv.dilate(canvas,(3,3),iterations = 1)
##                canvas = np.resize(canvas,(28,28,1))
##                cv.imshow('canvas',canvas)
                canvas = np.array(ROI)
                canvas = canvas.reshape(1,28*28)
                canvas = canvas.astype('float32') / 255
                result = model.predict_classes(canvas)
                proba = model.predict_proba(canvas)
                if max(proba[0]) >= 2e-01:
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(img,str(result[0]),(x,y-10),font,1,(255,0,0),2)
            #title = 'ROI' + str(i+1) + '.jpg'
            #cv.imwrite(title,roi)
                    IAP1 = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            if(IAP1.any()):
                count = 1

        if(count == 0):
            cv.imshow('IAP1',img)
            #cv.waitKey(30)
        else:
            cv.imshow('IAP1',IAP1)
            #cv.waitKey(30)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cv.destroyAllWindows()
cap.release()
