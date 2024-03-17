# Basic tensorflow sequential model that can classify handwritten numbers going from 0-9

import tensorflow as tf
import numpy

# Loads the mnist dataset
mnist = tf.keras.datasets.mnist

# Converts the pixel rgb value from 0-255 to 0-1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Trains a base model based on the tf sequential models that just stacks layers on top of one another
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # Flatten layer to convert 2D array of pixels into a 1D array
  tf.keras.layers.Dense(128, activation='relu'), # Fully connected (dense) layer with ReLU activation function
  tf.keras.layers.Dropout(0.2), # Dropout layer to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training
  tf.keras.layers.Dense(10) # Fully connected (dense) output layer with 10 units (for 10 classes) but without an activation function, which means it will output logits
])

# Outputs logits or log-odds for each class (outputs negative numbers but the line below fixes that)
predictions = model(x_train[:1]).numpy()

# Converts to actual predictions made by the model 
tf.nn.softmax(predictions).numpy()

# Loss function that determines the loss that a model makes
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Determines the loss made by this specific model
loss_fn(y_train[:1], predictions).numpy()

# Compiles the model and uses an optimizer called "adam", the loss is set to our loss function, and the evaluation criterion is set to "accuracy"
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Applies the model to the training data and labels for a certain number of epochs (iterations over the entire dataset)
model.fit(x_train, y_train, epochs=5)

# Evaluates the model by comparing dataset "x_test" with dataset "y_test" where "y_test" is the correct dataset
model.evaluate(x_test,  y_test, verbose=2) # Verbose controls the amount of output the model will print out

# Outputs the probability that the model based its output on
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])





