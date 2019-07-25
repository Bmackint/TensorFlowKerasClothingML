import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape
test_images.shape

# ##visualizing
# len(train_labels)
# print(len(train_images))
# print(train_images[0])
# print(len(train_labels))
# print(train_labels[0])
# # print(len(test_images))
# plt.imshow(train_images[0])
# plt.show()

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


##process data

#scale data (/by max value in tensor: 255)
#option 1
train_images = train_images / 255.0
test_images = test_images / 255.0
#or
#option 2 (test this later, different values..)
# train_images = tf.keras.utils.normalize(train_images, axis=1)
# test_images = keras.utils.normalize(test_images, axis=1)

# print(train_images[0])

# ##verify data is correct, display first 25 with labels
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
plt.show()

##build model & add layers
model = keras.Sequential([ 
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu), ##note: read up on and tutorial of Rectified linear activation function
    keras.layers.Dense(10, activation=tf.nn.softmax) ##note: read up on and tutorial of softmax activation function
])
