import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd


print("TensorFlow Version: ", tf.__version__)
print("Keras Version: ", keras.__version__)

# Loading The Dataset
mnist = keras.datasets.mnist

(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

print("Shape of the Full Training set", X_train_full.shape)
print("Shape of the Test set", X_test.shape)

# Converting the Pixel Intensities of the Training and the Test Sets
X_train_full, X_test = X_train_full/255.0, X_test/255.0

# Creating the validation Set
X_val, X_train = X_train_full[:10000], X_train_full[10000:]
y_val, y_train = y_train_full[:10000], y_train_full[10000:]

print("Scaled Training Set: ", X_train.shape)
print("Scaled Val_set: ", X_val.shape)
print("Scaled Test Set: ", X_test.shape)

# Model Creation

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(units=130, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint("mnist_best_model.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), callbacks=[checkpoint_cb,
                                                                                            early_stopping_cb])

print("____________________________________________________________________________________________________")
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
print("____________________________________________________________________________________________________")

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
y_new = y_test[:3]
print("Actual Class", y_new)
