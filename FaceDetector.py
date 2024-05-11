import cv2
import numpy as np 
import os
from PIL import Image

class Face_Recog:
    def __init__(self):
        pass
       
    def face_recog(self,img):        
        labels = ['Natalie',"cs"] 

        # Load the Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Create an LBPH face recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # Load the trained LBPH face recognizer model
        recognizer.read("face-trainner.yml")

        name="unknown"
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert Video frame to Greyscale
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) #Recog. faces
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w] #Convert Face to greyscale 

            id_, conf = recognizer.predict(roi_gray) #recognize the Face
                
            if conf>=120:
                font = cv2.FONT_HERSHEY_SIMPLEX #Font style for the name 
                name = labels[id_] #Get the name from the List using ID number 
                cv2.putText(img, name, (x,y), font, 1, (0,0,255), 2)
                
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.imshow('Preview',img) #Display the Video
        print('------',name)
        



