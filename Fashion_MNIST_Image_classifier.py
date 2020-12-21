import tensorflow as tf
from tensorflow import keras

print("TensorFlow Version: ", tf.__version__)
print("Keras Version: ", keras.__version__)

# Building an Image Classifier Using Sequential API

# Using Keras to Load the Dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("X_train_full Shape: ", X_train_full.shape)
print("X_train_full Datatype: ", X_train_full.dtype)

# Creating a Validation Set

# By Division by 255.0 we are scaling the Input Features, Here for simplicity we scaled pixel intensities down to 0-1,
# and converted them to Float.
X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print(class_names[y_train[0]])
print("_____________________________________________")
# Creating a Model Using Sequential API
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

print(model.layers)
print(model.summary())

# Compiling the Model
model.compile(loss="sparse_categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])



