import cv2
import os 
import tensorflow as tf
from tensorflow import keras

from PIL import Image
import numpy as np

from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense
from sklearn.model_selection import train_test_split


images_directory='Training_Dataset/'
#List the content of the Dataset
no_tumor_images = os.listdir (images_directory + 'no/')
yes_tumor_images = os.listdir (images_directory + 'yes/')
#print(no_tumor_images)

dataset=[]
label=[]

#Preprocessing of the Normal images
for i, image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(images_directory + 'no/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64, 64))
        #Rendre les images matrix
        dataset.append(np.array(image))
        #LAbel = 0 : no Brain Tumor
        label.append(0)


#Preprocessing of the Tumor images
for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(images_directory + 'yes/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((64, 64))
        #Rendre les images matrix
        dataset.append(np.array(image))
        #LAbel = 0 : no Brain Tumor
        label.append(1)

#Print the matrix of each dataset's image
dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

#Normalize Data
x_train=normalize(x_train, axis=1)
x_test=normalize(x_test, axis=1)

#Build Model
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(64,64,3)))
model.add(Activation ('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3), kernel_initializer='he_uniform'))
model.add(Activation ('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), input_shape=(64,64,3)))
model.add(Activation ('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


#Flatten
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#Dense=1 because it's a Classification Problem (Tumor/Non Tumor)
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_cossentropy', optimizer='adam', metrics= ['accuracy'])
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=False)

model.save('BrainTumorClassification10Epochs.h4')