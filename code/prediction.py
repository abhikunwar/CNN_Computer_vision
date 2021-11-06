import tensorflow as tf
import keras
from keras.preprocessing.image import load_img,img_to_array
import os
model = keras.models.load_model('../models/first_ann_2021_11_05_14_07_06.h5')

image = r'E:\ineuron_learning\tfod1_web_app\dog_cat\data\validation\cats\cat.2002 - Copy.jpg'

img = load_img(image, target_size= (64,64))
img_arr = img_to_array(img)
img_arr = img_arr.reshape(1,64,64,3)
print(img_arr.shape)

print(model.predict_classes(img_arr))