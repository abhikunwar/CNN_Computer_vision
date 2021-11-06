from training import DataProcessing
import keras
import tensorflow as tf
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense
from keras.models import Sequential
import numpy as np
train_cat_folder = '../data/train/cats'
train_dog_folder = '../data/train/dogs'
test_cat_folder = '../data/validation/cats'
test_dog_folder = '../data/validation/dogs'
training_image_array = []
testing_image_array = []
training_label = []
testing_label = []



if __name__=='__main__':
    tr = DataProcessing()
    training_image, testing_image, training_label, testing_label = tr.load_convert_to_array(train_cat_folder,
                                                                                           train_dog_folder,
                                                                                           test_cat_folder, test_dog_folder)
    training_image = np.array(training_image)
    testing_image = np.array(testing_image)
    training_label = np.array(training_label)
    testing_label = np.array(testing_label)
    model =  tr.model_bulding()
    print(training_image.shape)
    print(testing_image.shape)
    print(training_label.shape)
    print(testing_label.shape)
    print(model.summary())
    model,history = tr.model_compile_training(model,training_image,training_label,testing_image,testing_label)
    tr.save_model(model)
