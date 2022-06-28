#https://www.youtube.com/watch?v=MrRGVOhARYY
#https://docs.opencv.org/4.x/d2/d42/tutorial_face_landmark_detection_in_an_image.html
import cv2
import dlib
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import model_from_json

import numpy as np
import os
import time 
import tensorflow as tf
from tensorflow import keras
root_dir = os.getcwd()
json_file = open('vgg16.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights
model.load_weights('vgg16.h5')

cap = cv2.VideoCapture(0)
#time.sleep(2.0)"/dev/video2"
face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("data")
currentframe=0
while True:

    suceess, frame = cap.read()
    currentframe+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    # detect the face
    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right() ,face.bottom()
        face = frame[x1:x2,y1:y2]
        face = cv2.resize(face,(224,224))
        face = img_to_array(face)
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)
        preds = model.predict(face)[0]
        j = np.argmax(preds)
        print(preds,"   =  " ,j , " c=  " ,currentframe)
        if j==0 :
                label = 'spoof'
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (x1,y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.rectangle(frame,(x1+5,y1+2),(x2+5,y2+2),(0,255,0),2)
                
        else:
                label = 'real'
                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (x1,y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.rectangle(frame,(x1+5,y1+2),(x2+5,y2+2),(0,255,0),2)
        
   #frame = cv2.resize(frame,(1000,800))
    cv2.imshow('img',frame) #show frame in screen 
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break 
print("done saving")
cap.release()

cv2.destroyAllWindows()
