'''
Enter the id of the user, and then the camera will take n photos, stored in the ./faces/new
'''

import cv2
import os
import numpy as np  # Add this import for rotation

img_src='faces'
os.system('mkdir -p '+img_src+'/new/')



cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cv2.namedWindow('frame',cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

font = cv2.FONT_HERSHEY_SIMPLEX
detector=cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
sampleNum = 0
ID = input('enter your id: ')


while True:
    ret, img = cap.read()
    if ret:
        # Rotate the image 90 degrees clockwise
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        h, w, c = img.shape
        w1 = h*240//320
        x1 = (w-w1)//2
        img = img[:, x1:x1+w1]
        img = cv2.resize(img, (240, 320))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces: 
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
            sampleNum = sampleNum + 1 
            if sampleNum <= 100: 
                cv2.putText(img, 'shooting', (10, 50), font, 0.6, (0, 255, 0), 2) 
                
                cv2.imwrite(img_src +'/new/'+ str(sampleNum) + '.' + str(ID) + ".jpg",gray[y:y + h, x:x + w])
            else:
                cv2.putText(img, 'Done,Please quit', (10, 50), font, 0.6, (0, 255, 0), 2)
        cv2.imshow('frame', img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('b'): #quit the program with b
            break


cap.release()
cv2.destroyAllWindows()