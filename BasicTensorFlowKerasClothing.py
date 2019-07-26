##https://www.tensorflow.org/tutorials/keras/basic_classification
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

##compile model
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'
])

##train model
bTrainSuccess = False
numberOfTrainFalures = 0
while(not bTrainSuccess):
    try:
        model.fit(train_images, train_labels, epochs=5)
        bTrainSuccess = True
    except:
        numberOfTrainFalures = numberOfTrainFalures + 1

## evaluate accuracy
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('Test accuracy : ' + str(test_accuracy))
print("Test loss : {0}".format(test_loss))


## make predictions
predictions = model.predict(test_images)

print("prediction at [0] : " + str([predictions[0]])) #prints confidence that image corresponds to each article of clothing

np.argmax(predictions[0]) # highest confidence value
print("actual label value = " + str(test_labels[0]))



# ##visualize first 25 predictions

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[np.argmax(predictions[i])]) #shows number... rather have name, below not working...
#     #plt.xlabel(np.argmax(class_names[predictions[i]]))
# plt.show()



def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = (np.argmax(predictions_array))
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], ##note: looke up how the {:2.0f} works...(or test values to see how it equals 80...)
                                  100*np.max(predictions_array),
                                  class_names[true_label]),
                                  color = color)
    # ## almost equivalent... work out {:2.0f} ... (note: .format is better though)
    # plt.xlabel((str(class_names[predicted_label]) + 
    #             str(100*np.max(predictions_array)) + "% " + 
    #             str(class_names[true_label])), color = color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
def plot_image_graph_prediction(i):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1) #rows, columns, index???
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions, test_labels)
    plt.show()

# #vidualize individual prediction, graph, [confidence]*
# plot_image_graph_prediction(0)
# plot_image_graph_prediction(12)

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt. subplot(num_rows, 2*num_cols, 2*i+2) # ...
    plot_value_array(i, predictions, test_labels)
plt.show()

##note: look at adding more detail to graphs (specifically label rows and/or add a legend for colors)

