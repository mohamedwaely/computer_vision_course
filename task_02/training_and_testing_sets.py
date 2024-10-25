
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

# Print and visualize a sample training image and its label
plt.imshow(training_imgs[10])
print("Training label:", training_labels[10])  # Print the corresponding label
print("Training image:", training_imgs[10])    # Print the raw pixel values
plt.show()

