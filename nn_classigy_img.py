# A neural network that can classify immages of clothing like shirts or sneakers

import tensorflow as tf

# Helper libs
import numpy as np
import matplotlib.pyplot as plt # lib that does visual representation of data, (uses graphs to visulize math essentailly)

# Loads the fashion mnist dataset which is a dataset containing images of clothing like shirts, jackets, shoes, purses, etc...
fashion_mnist = tf.keras.datasets.fashion_mnist

# Comapres the images that the model outputs with the correct images to see if it is correct in its evaluation
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Class names that correspond to a label, used for plotting each image in a visulization graph
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define the format of the images which in this case is 28x28 and 60,000 labels
train_images.shape
len(train_labels)

# Each label is an int ranging from 0-9 corresponding to a classname
train_labels

# Same thing as above but with 10,000 images (these are not labels but the actual images that the model will be processing)
test_images.shape
len(test_labels)

# Show the values for one image in the dataset
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale each rgb value of each image from 0-255 to 0-1, (You do so by dividing by 255)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display first 25 images with its corresponding int/classname to ensure correct format is being used
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# Using tf sequential model, do a process to each layer
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Scale the model processing to 28 x 28 1d array so 28*28 ( important step as it formats the other layers so that it processes correctly)
    tf.keras.layers.Dense(128, activation='relu'), # Main process used by the model called the dense layer using relu activation gets inputs from first layer processes it, and outputs to the next layer ( using 128 nodes in this neural netwoerk )
    tf.keras.layers.Dense(10) # Another dense layer to further improve the model, but without an activation model
])

# Compliling the model using an optimizer called "adam" a default tf loss function, and scales the criteria of the model on a metric called "accuracy"
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Fit all of the preprocessing done previously, in the model, going over the database 10 times (epochs)
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model with a verbose of 2 to print two lines
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) # model.evaluate returns two arrays

print('\nTest accuracy:', test_acc)

# Have the model make predications by adding a softmax layer
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

# See which class name/int the model is most confident in
np.argmax(predictions[0])

# FUNCTIONS THAT GRAPH THE DATA VISUALLY
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

