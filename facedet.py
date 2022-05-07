import cv2
import numpy as np 
from tkinter import Tk
from tkinter.filedialog import askopenfilename
#Tk().withdraw() 
#media_photo = askopenfilename()   
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Getting starrted with cv2
#########################################
image = cv2.imread("image1.jpg" ,-1)


#resizing image 
#img = cv2.resize(image,(300,300))
#cv2.imshow('image2',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#command below used to display the 3D array of our image 
#print(img)
#print(img.shape)
#########################################




#Getting started with Videocapture function  
########################################
#cap = cv2.VideoCapture(0)

#while True:
    #ret, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Frame', frame)
    #if(cv2.waitKey(1) == ord('e')):
     #   break
#cap.release()
#cap.destroyAllWindows()
#######################################################


####Rectangle drawing ######################
#img = cv2.rectangle((300, 300), (200, 200), (134, 56, 132), 6)
while True:
   
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, 1.3,5)

   for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame',image)
   if(cv2.waitKey(1)==ord('e')):
        break

cv2.destroyAllWindows()
########################################################\

