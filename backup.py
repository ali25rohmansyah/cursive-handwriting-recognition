# import package
import numpy as np
import mnist # get data from
import matplotlib.pyplot as plt # grph
from keras.models import Sequential # ann architecture
from keras.layers import Dense # Layer in ann
from keras.utils import to_categorical

# load dataset
train_images = mnist.train_images() # train data image
train_labels = mnist.train_labels() # train  data labels
test_images = mnist.test_images() # train  data images
test_labels = mnist.test_labels() # train  data labels

# normize the images. normalize the pixel values from [0, 255]tp
# [-0.5, 0,5] to make our network easier to train
train_images = (train_images/255) - 0.5
test_images = (test_images/255) - 0.5

# flatten the images. flatten each 28x28 image into  a 28^2 = 784 dimensional vector
# to pass into the neural network
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# print the shape
print(train_images.shape)
print(test_images.shape)

# build the model
# 3 layers, 2 layers with 64 neurons and the relu function
# 1 layers with  10 neurons and softmax function
model = Sequential()
model.add( Dense(64, activation='relu', input_dim=784))
model.add( Dense(64, activation='relu'))
model.add( Dense(10, activation='softmax'))

# compile the model
# the loss function measures how well the model did on training, and then tries
# to improve on it using the optimizer
model.compile(
    optimizer='adam',
    loss= 'categorial_crossentropy', 
    metrics=['accuracy']
)

# train model
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32
)

# evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)



