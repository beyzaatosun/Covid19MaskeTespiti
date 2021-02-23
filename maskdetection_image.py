# -*- coding: utf-8 -*-
"""
@author: BeyzaTosun
"""
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array

img='examples/2.jpg'
model = load_model('models/facemask.h5')
#görüntü okunur
img = plt.imread(img)
#cvtColor()=bir görüntüyü bir renk uzayından diğerine dönüştürmek için kullanılır
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#img=Nesnelerin algılandığı bir görüntüyü içeren CV_8U tipi matris
#scaleFactor=Her bir görüntü ölçeğinde görüntü boyutunun ne kadar küçüleceğini belirten parametre
#minNeighbors=Her bir aday dikdörtgenin kaç tane komşusu tutması gerektiğini belirten parametre
faces = face_cascade.detectMultiScale(gray, 1.1, 10)

#Her yüzün etrafına dikdörtgenler çizmek için
for (x, y, w, h) in faces:
    #dikdörtgen yüzleri kaydetmek için
    #x ve y değerlerin dikdörtgenin sol üst koordinatlarını temsil eder
    #w ve h değerleri sırasıyla dikdörtgenin genişliğini ve yüksekliğini temsil eder
    face = img[y:y+h, x:x+w]
    #görüntüyü 150,150 olarak yeniden boyutlandırır
    face = cv2.resize(face, (150, 150))
    #görüntüyü numpy dizisine dönüştürür
    face = img_to_array(face)
    #face = preprocess_input(face)
    #dizinin şeklini değiştirir test_image (3,)-->(1,3) e değiştirilir
    face = np.expand_dims(face, axis=0)
    #sınıf değerlerini tahmin etmek için kullanılır
    pred=model.predict_classes(face)[0][0]
    if pred==1:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
        cv2.putText(img,'MASKE TAK!',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
    else:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.putText(img,'MASKE VAR',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

#görüntüleri göstermek için
plt.imshow(img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imwrite("examples/detected.jpg",img)
	
#Başlatılan tüm pencereleri kapatır
cv2.destroyAllWindows()   




