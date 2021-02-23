# -*- coding: utf-8 -*-
"""
@author: BeyzaTosun
"""
import cv2
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img , img_to_array
import numpy as np

model =load_model('models/facemask.h5')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('examples/02.mp4')
img_count_full = 0
org = (1,1)
class_label = ''
fontScale = 1
color = (255,0,0)
thickness = 2
while True:
	img_count_full += 1
	response , color_img = cap.read()

	if response == False:
		break
	scale = 50
	width = int(color_img.shape[1]*scale /100)
	height = int(color_img.shape[0]*scale/100)
	dim = (width,height)
    #görüntüyü yeniden boyutlandırır INTER_AREA= piksel alanı ilişkisini kullanarak yeniden boyutlandırır
	color_img = cv2.resize(color_img, dim ,interpolation= cv2.INTER_AREA)
    #cvtColor()=bir görüntüyü bir renk uzayından diğerine dönüştürmek için kullanılır
	#gray_img = cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)
    #img=Nesnelerin algılandığı bir görüntüyü içeren CV_8U tipi matris
    #scaleFactor=Her bir görüntü ölçeğinde görüntü boyutunun ne kadar küçüleceğini belirten parametre
    #minNeighbors=Her bir aday dikdörtgenin kaç tane komşusu tutması gerektiğini belirten parametre
	faces = face_cascade.detectMultiScale(color_img, 1.1, 6)

	img_count = 0
	for (x,y,w,h) in faces:
		org = (x-10,y-10)
		img_count += 1
		color_face = color_img[y:y+h,x:x+w]
        #görüntüyü kaydetmek için kullanılır
		cv2.imwrite('video_input/%d%dface.jpg'%(img_count_full,img_count),color_face)
        #load_img ()=görüntüyü dosyadan bir PIL görüntü nesnesi olarak yükler,boyutlarıda belirlenir
		img = load_img('video_input/%d%dface.jpg'%(img_count_full,img_count),target_size=(150,150))
        #görüntüyü numpy dizisine dönüştürür
		img = img_to_array(img)
        #dizinin şeklini değiştirir test_image (3,)-->(1,3) e değiştirilir
		img = np.expand_dims(img,axis=0)
        #sınıf değerlerini tahmin etmek için kullanılır
		prediction = model.predict(img)
		if prediction==0:
			class_label = "MASKE VAR"
			color = (0,255,0) 
            
		else:
			class_label = "MASKE TAK!"
			color = (0,0,255)
		cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,255),3)
		cv2.putText(color_img, class_label, org,cv2.FONT_HERSHEY_SIMPLEX ,fontScale, color, thickness,cv2.LINE_AA)
    #görüntüleri göstermek için kullanılır '' arasında yazılan pencere adıdır
	cv2.imshow('Maske Tespiti', color_img)
    #kullanıcı q tuşa basınca döngü sonlanır
	if cv2.waitKey(1) == ord('e'):
		break
#döngü bittikten sonra cap nesnesini serbest bırakır
cap.release()
#Başlatılan tüm pencereleri kapatır
cv2.destroyAllWindows()