from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt

housing = fetch_california_housing()

# Splitting the Data into Train and Test Set.
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target)

# Splitting the Training data into Train and Validation Sets
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)

# Scaling the Dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Creating a Sequential Model
model = tf.keras.models.Sequential()
model.add(Dense(units=30, activation=tf.keras.activations.relu, input_shape=X_train[1:]))
model.add(Dense(units=1))

model.compile(loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.SGD)

# Model Fitted For Training.
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid))

print("______________________________________________________________________________________________")
# Plotting the MODEL HISTORY
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

print("_____________________________________________________________________________________________")

# Evaluating the Model on the Test Set
mse_test = model.evaluate(X_test, y_test)
print(mse_test)
print("_____________________________________________________________________________________________")

# Predicting on New Values
X_new = X_test[:3]
y_pred = model.predict(X_new)
print(y_pred)
print("______________________________________________________________________________________________")
if __name__ == "__main__":
    print(housing.keys())
