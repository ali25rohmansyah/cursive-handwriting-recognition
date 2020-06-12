# -*- coding: utf-8 -*-
"""Detection

"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model

import os
import numpy
import tensorflow as tf 

K.common.set_image_dim_ordering('th')

image_folder = './result/image/'

# In[baselineK_model]

#Defining the model
def baselineK_model():
    # create model
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(1,28,28), activation='relu'))
    model.add(Conv2D(32,(3,3),strides=(1,1),padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32,(3,3),strides=(1,1),padding="same"))
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3),strides=(1,1),padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# In[Main]

if not (os.path.exists('./gzip/Mnist1L_5Conv.h5')):
    mnist = tf.keras.datasets.mnist

    (train_images, y_train), (test_images, y_test) = mnist.load_data()
    
    test_images= test_images.astype('float32')
    train_images = train_images.astype('float32')
    
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28).astype('float32') 
    
    seed = 7
    numpy.random.seed(seed) 
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

#if not (os.path.exists('./gzip/Mnist1L_5Conv.h5')):
    print("No model found, creating new")
    model = baselineK_model()
    print(model.summary())
    # Fit the model
    model.fit(train_images, y_train, validation_data=(test_images, y_test), epochs=8, batch_size=30, verbose=2)
    model.save('./gzip/Mnist1L_5Conv.h5')
else:
    print('Model found')
    model = load_model('./gzip/Mnist1L_5Conv.h5')
    print(model.summary())
    
# Final evaluation of the model
scores = model.evaluate(test_images, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))



















