
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist

(training_imgs, training_labels), (testing_imgs, testing_labels) = mnist.load_data()

# Assert shapes of data for sanity check
assert training_imgs.shape == (60000, 28, 28)  # 60,000 images, 28x28 pixels each
assert training_labels.shape == (60000,)       # 60,000 labels
assert testing_imgs.shape == (10000, 28, 28)    # 10,000 testing images
assert testing_labels.shape == (10000,)        # 10,000 testing labels

# Preprocess the data by normalizing pixel values to [0, 1] range
training_imgs = training_imgs / 255.0
testing_imgs = testing_imgs / 255.0

# Define the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten the 2D image to 1D vector
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Output layer with 10 neurons (one for each class) and Softmax activation
])

# Compile the model for training
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data for 5 epochs
model.fit(training_imgs, training_labels, epochs=5)

