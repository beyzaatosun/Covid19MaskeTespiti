# -*- coding: utf-8 -*-
"""
@author: BeyzaTosun
"""
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import cv2
import numpy as np
import datetime

model=load_model('models/facemask.h5')
#yüz tespiti için kullanılacak xml dosya yolunu verir
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#kameradan canlı akışı yakalamak için kullanılır
cap=cv2.VideoCapture(0)
#cap.isOpened() yöntemiyle kamera başlatılıp başlatılmadıgı kontrol edilir
while cap.isOpened():
   #cap.read bool türündedir.çerçeve dogru okunnursa true döndürür
    _,img=cap.read()
    #img=Nesnelerin algılandığı bir görüntüyü içeren CV_8U tipi matris
    #scaleFactor=Her bir görüntü ölçeğinde görüntü boyutunun ne kadar küçüleceğini belirten parametre
    #minNeighbors=Her bir aday dikdörtgenin kaç tane komşusu tutması gerektiğini belirten parametre
    face=face_cascade.detectMultiScale(img,scaleFactor=3,minNeighbors=4)
    #Her yüzün etrafına dikdörtgenler çizmek için
    for(x,y,w,h) in face:
        #dikdörtgen yüzleri kaydetmek için
        #x ve y değerlerin dikdörtgenin sol üst koordinatlarını temsil eder
        #w ve h değerleri sırasıyla dikdörtgenin genişliğini ve yüksekliğini temsil eder
        face_img = img[y:y+h, x:x+w]
        #görüntüyü kaydetmek için kullanılır
        cv2.imwrite('examples/temp.jpg',face_img)
        #load_img ()=görüntüyü dosyadan bir PIL görüntü nesnesi olarak yükler,boyutlarıda belirlenir
        test_image=image.load_img('examples/temp.jpg',target_size=(150,150,3))
        #görüntüyü numpy dizisine dönüştürür
        test_image=image.img_to_array(test_image)
        #dizinin şeklini değiştirir test_image (3,)-->(1,3) e değiştirilir
        test_image=np.expand_dims(test_image,axis=0)
        #sınıf değerlerini tahmin etmek için kullanılır
        pred=model.predict_classes(test_image)[0][0]
        if pred==1:
            #cv2.rectangle=bir görüntü üzerine bir dikdörtgen çizmek için kullanılır
            #start_point=dikdörtgenin başlangıç koordinatları
            #end_point=dikdörtgenin bitiş koordinatları
            #color=dikdörtgen rengidir BGR sırasında
            #thickness=dikdörtgen kenar çizgisi piksel kalınlığı parametresi 
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            
            cv2.putText(img,'MASKE TAK!',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'MASKE VAR',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        datet=str(datetime.datetime.now())
        cv2.putText(img,datet,(400,450),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    #görüntüleri göstermek için kullanılır '' arasında yazılan pencere adıdır    
    cv2.imshow('LIVE',img)
    #kullanıcı e tuşa basınca döngü sonlanır
    if cv2.waitKey(1)==ord('e'):
        break
#döngü bittikten sonra cap nesnesini serbest bırakır     
cap.release()
#tüm pencereleri kapatır
cv2.destroyAllWindows()
    