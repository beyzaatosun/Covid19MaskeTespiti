# -*- coding: utf-8 -*-
"""
@author: BeyzaTosun
"""
import os 
import matplotlib.pyplot as plt

#veri setinin yolunu belirtiyoruz.
base_dir= r'C:\Users\User\Documents\YazılımStajı\New Masks Dataset'
#train, validation ve test için alt dizinleri belirtiyoruz.
train_dir = os.path.join(base_dir,'Train')
test_dir = os.path.join(base_dir,'Test')
valid_dir = os.path.join(base_dir,'Validation')
#train sınıfına ait alt dizinleri belirtiyoruz.
train_mask_dir = os.path.join(train_dir,'Mask')
train_nomask_dir = os.path.join(train_dir,'Non Mask')
#train sınıfına ait maskeli ve maskesiz sınıflarda toplam kaç tane görüntü oldugunu ekrana yazdırıyoruz.
print('total mask train images:',len(os.listdir(train_mask_dir)))
print('total no mask train images:',len(os.listdir(train_nomask_dir)))
#train sınıfına ait maskeli ve maskesiz sınıfların ilk 10 tanesini değiskene atıyoruz ve adlarını ekrana yazdırıyoruz.
train_mask_names = os.listdir(train_mask_dir)
print('train mask names:',train_mask_names[:10])
train_nomask_names = os.listdir(train_nomask_dir)
print('train no mask names:',train_nomask_names[:10])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,#piksel degerlerini 0-1 arasında olacak sekilde ayarlar
                                   zoom_range = 0.2,#resimlerin içini rastgele yakınlaştırmak icindir
                                   rotation_range = 40,#(0-180)derece cinsinden degerdir, resimleri rastgele döndürür
                                   width_shift_range=3/32,#resimleri yatay olarak çevirir
                                   height_shift_range=3/32,#resimleri dikey olarak çevirir
                                   horizontal_flip =True#resimleri rastgele olarak yarısını çevirir                                   
                                   )

test_datagen = ImageDataGenerator(rescale=1./255)#veriler 0-1 arasında ölçeklenir

validation_datagen = ImageDataGenerator(rescale=1./255)#veriler 0-1 arasında ölçeklenir

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150,150),#görüntüleri 150x150 boyutlu olarak ayarlar
                                                    batch_size = 20,#her bir eğitim verisi grubuna dahil edilecek görüntü sayisi
                                                    class_mode ='binary' #ikili sınıflandırma
                                                    )

test_generator = test_datagen.flow_from_directory(test_dir,
                                                    target_size=(150,150),#görüntüleri 150x150 boyutlu olarak ayarlar
                                                    batch_size = 20,#her bir eğitim verisi grubuna dahil edilecek görüntü sayisi
                                                    class_mode ='binary' #ikili sınıflandırma
                                                    )

valid_generator = validation_datagen.flow_from_directory(valid_dir,
                                                    target_size=(150,150),#görüntüleri 150x150 boyutlu olarak ayarlar
                                                    batch_size = 18,#her bir eğitim verisi grubuna dahil edilecek görüntü sayisi
                                                    class_mode ='binary' #ikili sınıflandırma
                                                    )

#ağ yapısı için gerekli olan kütüphaneler import edilir
from keras import layers
from keras import models
from tensorflow import keras
model=models.Sequential()
#ağ yapısı tanımlanır
#2 adet konvolüsyon katmanı ve 2 adet Max pooling uygulanmıştır
model.add(layers.Conv2D(32,
                        (3,3),
                        activation='relu',
                        padding='same',#çıkış görüntüsünün boyutunu sabit tutmak için dolgulama yapılabilir
                        input_shape=(150,150,3)))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))#aşırı uydurmayı(overfitting) düşürmek için dropout eklenir

model.add(layers.Conv2D(64,
                        (3,3),
                        padding='same',
                        activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Dropout(0.5))#aşırı uydurmayı(overfitting) düşürmek için dropout eklenir

model.add(layers.Flatten())#düzleştirme işlemi yapılır
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))#aşırı uydurmayı(overfitting) düşürmek için dropout eklenir
model.add(layers.Dense(1,activation='sigmoid'))#ikili sınıflandırma oldugu için activation olarak sigmoid kullanılmıstır
#model ağ katmanları hakkında bilgileri yazdırır
model.summary()

#Optimizasyon parametreleri
opt = keras.optimizers.Adam(learning_rate=0.001)#optimizasyon algroitması olarak Adam seçildi
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              #ağın cıkısı iki sınıfa ait bir olasılık belirtecegi icin binary_crossentropy kullanılır
              metrics=['acc'])#eğitim sırasında dogruluk gözlenir

#ImageDataGenerator kullandığımız için fit_genetor ile modelimizi eğitiyoruz
history = model.fit_generator(train_generator,
                            steps_per_epoch=30,
                            #agırlıklar bir epoch içinde kaç kez güncelleneceğini belirtir
                            #step_per_epoch=total training samples/training batch size ile hesaplanır
                            epochs = 30,
                            validation_steps=17,
                            #validation_steps=total validation samples/ validation batch size ile hesaplanır
                            validation_data = valid_generator)

#Eğitilmiş ağı kaydetmek için kullanılır
model.save('facemask.h5')

#eğitim sonunda fit() ile döndürülen history nesnesinden optimizasyon parametrelerinin epocha baglı değisimi incelenebilir
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'bo', label='Training acc')#1.grafik
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')#2.grafik
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
