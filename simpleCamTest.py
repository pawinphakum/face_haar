import numpy as np
import cv2
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 400) # set Width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 300) # set Height
while(True):
    ret, frame = cam.read()
    #frame = cv2.flip(frame, -1) # Flip camera vertically
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)
    cv2.imshow('gray', gray)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break
cam.release()
cv2.destroyAllWindows()
