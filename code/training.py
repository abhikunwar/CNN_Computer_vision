#import pandas as pd
import numpy as np
import tensorflow as tf
import keras
import os
from keras.preprocessing.image import load_img,img_to_array
import cv2
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,MaxPooling2D
from keras.models import Sequential
import time


train_cat_folder = '../data/train/cats'
train_dog_folder = '../data/train/dogs'
test_cat_folder = '../data/validation/cats'
test_dog_folder = '../data/validation/dogs'
training_image_array = []
testing_image_array = []
training_label = []
testing_label = []


class DataProcessing:

    def __init__(self):
        #self.training_data = training_data
        pass

    def load_convert_to_array(self, train_cat_folder, train_dog_folder, test_cat_folder, test_dog_folder):
        for image in os.listdir(train_cat_folder):
            img = load_img(os.path.join(train_cat_folder, image), target_size= (64,64))
            img_arr = img_to_array(img)
            # img_arr = img_arr.flatten()
            # print(img_arr.shape)
            # img_arr = img_arr.reshape(img_arr ,(64, 64,3))
            # print(img_arr.shape)
            training_label.append(0)
            training_image_array.append(img_arr)

        for image in os.listdir(train_dog_folder):
            img = load_img(os.path.join(train_dog_folder, image), target_size= (64,64))
            img_arr = img_to_array(img)
            training_label.append(1)
            training_image_array.append(img_arr)

        for image in os.listdir(test_cat_folder):
            img = load_img(os.path.join(test_cat_folder, image),target_size= (64,64))
            img_arr = img_to_array(img)
            testing_label.append(0)
            testing_image_array.append(img_arr)

        for image in os.listdir(test_dog_folder):
            img = load_img(os.path.join(test_dog_folder, image),target_size= (64,64))
            img_arr = img_to_array(img)
            testing_label.append(1)
            testing_image_array.append(img_arr)



        return training_image_array, testing_image_array,training_label,testing_label

    def model_bulding(self):

        layers = [Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'),#62
                  MaxPooling2D(pool_size=(2, 2)),#31
                  Flatten(),
                  Dense(units=128, activation='relu'),
                  Dense(units=1, activation='sigmoid')]

        model = Sequential(layers)
        print(model.summary)
        return model

    def model_compile_training(self,model,training_image_array,training_label,testing_image_array,testing_label):

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        validation_set = (testing_image_array,testing_label)

        history =model.fit(training_image_array,training_label, epochs=20, validation_data=validation_set)
        return model,history

    def save_model(self,model):

        filename = time.strftime('first_ann_%Y_%m_%d_%H_%M_%S.h5')

        #tf.keras.models.save_model(model, '../data/models' + '/' + filename)
        model.save('../models' + '/' + filename)












# if __name__=='__main__':
#     t = DataProcessing()
#     training_image,testing_image, training_label, testing_label = t.load_convert_to_array(train_cat_folder, train_dog_folder, test_cat_folder, test_dog_folder)
#     training_image = np.array(training_image)
#     testing_image = np.array(testing_image)
#     training_label = np.array(training_label)
#     testing_label = np.array(testing_label)
#     #print(training_image)
#     print(training_image.shape)
#     print(testing_image.shape)
#     print(training_label.shape)
#     print(testing_label.shape)






