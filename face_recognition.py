''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition
Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18
'''

import cv2
import numpy as np
import os
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = 'cascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
#id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Pop', 'Ple', 'Tunyong', 'Giwha',]

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 400) # set Width
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 300) # set Height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

startTime = datetime.now()
print(f'\nstart time = {startTime}\n')

while True:

    ret, img = cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        id, notmatch = recognizer.predict(gray[y:y+h, x:x+w])
        confidence = 100 - notmatch
        name = 'unknown'
        match_percent = f' {round(confidence)}%'

        if (confidence >= 55):
            name = names[id]
            #matchTimeStamp = datetime.now()
            #roundUsedTime =  matchTimeStamp - startTime
            #startTime = matchTimeStamp
            #print(f'{name} : {match_percent}  :  {round(roundUsedTime.microseconds/100000.0, 2)} s.')
            print(f'{name} : {match_percent}')

        cv2.putText(img, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(match_percent), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print('\n [INFO] Exiting Program and cleanup stuff')
cam.release()
cv2.destroyAllWindows()
