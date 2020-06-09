import tensorflow as tf
import cv2
from keras import initializers
import numpy as np
from PIL import Image
from skimage import transform

def my_init(shape, dtype=None):
    initializer = initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    return initializer(shape=shape)

new_model = tf.keras.models.load_model('kanji_model',custom_objects={'my_init':my_init})

filename = 'images/suki2.PNG'

#img = cv2.imread('images/hibi.png')
#img = cv2.resize(img,(32,32))
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img = np.reshape(img,[1,32,32,1])
#img = tf.cast(img, tf.float32)

np_image = Image.open(filename)
np_image = np.array(np_image).astype('float32')/255
np_image = transform.resize(np_image, (32, 32, 1))
np_image = np.expand_dims(np_image, axis=0)

print(new_model.predict_classes(np_image))

