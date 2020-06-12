from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical

import matplotlib.pyplot as plt # grph
import os
import numpy as np
import tensorflow as tf 

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = (train_images / 255) - 0.5, (test_images / 255) - 0.5

train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# print the shape
print(train_images.shape)
print(test_images.shape)

model = Sequential()
model.add( Dense(64, activation='relu', input_dim=784))
model.add( Dense(64, activation='relu'))
model.add( Dense(10, activation='softmax'))

# Compile model
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
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

# model.save_weight('model.h5')

# predict
predictions = model.predict(test_images[:5])
# print(predictions)

# print model prediction
print(np.argmax(predictions, axis =1))
print(test_labels[:5])

for i in range(0,5):
    first_image = test_images[i]
    first_image = np.array(first_image, dtype='float')
    pixels = first_image.reshape((28,28))
    # plt.imshow(pixels)
    # plt.show()

# x=[]
# res=[]
# fname=[]
# folder='./result/resized_images/'
# dirFiles=os.listdir(folder)
# dirFiles = sorted(dirFiles,key=lambda x: int(os.path.splitext(x)[0]))
# for filename in dirFiles:
#     imt = cv2.imread(os.path.join(folder,filename))
#     imt = cv2.blur(imt,(6,6))
#     gray = cv2.cvtColor(imt,cv2.COLOR_BGR2GRAY)
#     ret, imt = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
#     if imt is not None:
#         imt = imt.reshape((-1, 28, 28))
# #        plt.imshow(imt)
# #        plt.show()
#         imt=imt/255
#         x.append(imt)
#         fname.append(filename)

# x=np.array(x);    
# predictions = model.predict(x)
# classes = np.argmax(predictions, axis=1)    

# for i in range(len(classes)):
#     imt = cv2.imread(os.path.join(folder,dirFiles[i]))
#     plt.imshow(imt)
#     plt.show()
#     print([k for k,v in letter_count.items() if v == classes[i]])