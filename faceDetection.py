import numpy as np
import cv2
faceCascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 400) # set Width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 300) # set Height
while True:
    ret, img = cam.read()
    #img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (20, 20),
        )
    '''
    scaleFactor is the parameter specifying how much the image size
    is reduced at each image scale. It is used to create the scale pyramid.

    minNeighbors is a parameter specifying how many
    neighbors each candidate rectangle should have, to retain it.
    A higher number gives lower false positives.

    minSize is the minimum rectangle size to be considered a face.
    '''

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    cv2.imshow('video', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cam.release()
cv2.destroyAllWindows()
