import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

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
print("___________________________________________________________________________________________________")
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

# Training the Model
history = model.fit(X_train, y_train, epochs=150, validation_data=(X_valid, y_valid))

print("____________________________________________________________________________________________________")
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

model.evaluate(X_test, y_test)

print("____________________________________________________________________________________________________")
# Making Predictions
# Probability per class
X_new = X_test[:3]
y_proba = model.predict(X_new)
print("Prediction Probabilities: ", y_proba.round(2))

print("____________________________________________________________________________________________________")

# Predicting Classes
y_pred = model.predict_classes(X_new)
print("Prediction Classes: ", y_pred)

print("Prediction Class Name", np.array(class_names)[y_pred])

y_new = y_test[:3]
print("Actual Class", y_new)


