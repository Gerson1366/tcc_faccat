# This code is based on
# https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
# https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py

import numpy as np
import scipy.misc
from PIL import Image
from keras import backend as K
from keras import initializers
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


nb_classes = 3036
# input image dimensions
img_rows, img_cols = 32, 32
#img_rows, img_cols = 127, 128

#ary = np.load("kanji_01.npz")['arr_0'].reshape([-1, 127, 128]).astype(np.float32) / 15
ary = np.load("/content/drive/My Drive/Colab Notebooks/npy/ary01.npy", mmap_mode="r+")
ary2 = np.load("/content/drive/My Drive/Colab Notebooks/npy/ary02.npy",mmap_mode="r+")
ary3 = np.load("/content/drive/My Drive/Colab Notebooks/npy/ary03.npy", mmap_mode="r+")
ary4 = np.load("/content/drive/My Drive/Colab Notebooks/npy/ary04.npy",mmap_mode="r+")
X_train = np.zeros([nb_classes * 200, img_rows, img_cols], dtype=np.float32)
for i in range(1000 * 200):
      X_train[i] = Image.fromarray(ary[i]).resize((img_rows, img_cols))
     #X_train[i] = ary[i]
j=0
for i in range(200000, 400000):
    X_train[i] = Image.fromarray(ary2[j]).resize((img_rows, img_cols))
    j+=1
j=0
for i in range(400000, 600000):
    X_train[i] = Image.fromarray(ary3[j]).resize((img_rows, img_cols))
    j+=1
j=0
for i in range(600000, 607200):
    X_train[i] = Image.fromarray(ary4[j]).resize((img_rows, img_cols))
    j+=1
Y_train = np.repeat(np.arange(nb_classes), 200)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

if K.image_data_format() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.20)
datagen.fit(X_train)

model = Sequential()


def my_init(shape, dtype=None):
    initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    return initializer(shape=shape)

# Best val_loss: 0.0205 - val_acc: 0.9978 (just tried only once)
# 30 minutes on Amazon EC2 g2.2xlarge (NVIDIA GRID K520)
def m6_1():
    model.add(Convolution2D(32, 3, 3, init=my_init, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, init=my_init))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, init=my_init))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


def classic_neural():
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))


m6_1()
# classic_neural()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=800), samples_per_epoch=X_train.shape[0],
                    nb_epoch=400, validation_data=(X_test, Y_test))

model.save('my_model')
print("Saved model to disk")