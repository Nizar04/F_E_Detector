import tkinter
import customtkinter
import cv2
import numpy as np 
from tkinter import Tk
from tkinter.filedialog import askopenfilename

customtkinter.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

root_tk = customtkinter.CTk()  # create CTk window like you do with the Tk window (you can also use normal tkinter.Tk window)
root_tk.geometry("1024x576")
root_tk.title("CustomTkinter Test")


#Giving a Function To The Buttons
def button_1():
  cap=cv2.VideoCapture(0)

  face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

  while True:
      ret,frame = cap.read()

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
      faces = face_cascade.detectMultiScale(gray, 1.3,5)

      for (x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)

          roi_gray = gray[y:y+w,x:x+w]
          roi_color = frame[y:y+h,x:x+w]

          eyes= eye_cascade.detectMultiScale(roi_gray,1.3,5)

          for (ex,ey,ew,eh) in eyes:
              cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),5)

      cv2.imshow('frame',frame)

      if(cv2.waitKey(1)==ord('q')):
        break

  cap.release()
  cv2.destroyAllWindows()
def button_2():
   Tk().withdraw() 
   media = askopenfilename()
   cap=cv2.VideoCapture(media)

   face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
   eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

   while True:
      ret,frame = cap.read()

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
      faces = face_cascade.detectMultiScale(gray, 1.3,5)

      for (x,y,w,h) in faces:
          cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),5)

          roi_gray = gray[y:y+w,x:x+w]
          roi_color = frame[y:y+h,x:x+w]

          eyes= eye_cascade.detectMultiScale(roi_gray,1.3,5)

          for (ex,ey,ew,eh) in eyes:
              cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),5)

      cv2.imshow('frame',frame)

      if(cv2.waitKey(1)==ord('q')):
        break

      cap.release()
      cv2.destroyAllWindows()
def button_3():
    Tk().withdraw() 
    media_photo = askopenfilename()   
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread("media_photo" ,-1)
    while True:
   
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
     faces = face_cascade.detectMultiScale(gray, 1.3,5)

     for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('frame',image)
     if(cv2.waitKey(1)==ord('e')):
        break

     cv2.destroyAllWindows()

y_padding = 13
frame_1 = customtkinter.CTkFrame(master=root_tk, corner_radius=15)
frame_1.pack(pady=20, padx=60, fill="both", expand=True)

label_1 = customtkinter.CTkLabel(master=frame_1, justify=tkinter.LEFT)
label_1.pack(pady=y_padding, padx=10)

#Creating The Button
button_1 = customtkinter.CTkButton(master=frame_1, text ="Picture",corner_radius=8, command=button_1)
button_1.pack(pady=y_padding, padx=10)

button_2 = customtkinter.CTkButton(master=frame_1, text ="Live on WebCamera " ,corner_radius=8, command=button_2)
button_2.pack(pady=y_padding, padx=10)

button_3 = customtkinter.CTkButton(master=frame_1,text ="From File", corner_radius=8, command=button_3)
button_3.pack(pady=y_padding, padx=10)
#put on screen

root_tk.mainloop()

#print(filename)