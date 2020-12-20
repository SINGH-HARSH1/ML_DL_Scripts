import tensorflow as tf

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def data_loader():
    """This function Loads the data and prints Useful functions"""
    dataset = load_breast_cancer()
    # print(dataset.keys())
    # print(type(dataset))
    # print(dataset.data.shape)
    # print(dataset.feature_names)
    return dataset


X = data_loader().data
y = data_loader().target

dataset_features = data_loader().feature_names
dataset_descriptions = data_loader().DESCR
dataset_target_names = data_loader().target_names

# print(X)
# print(y)

# print(dataset_features)
# print(dataset_descriptions)
# print(dataset_target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Building the model
model1 = tf.keras.models.Sequential()
model1.add(Dense(units=30, activation='relu')),
model1.add(Dense(units=15, activation='relu')),
model1.add(Dense(units=1, activation='sigmoid'))


model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
r = model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# Evaluating the model
print("Train Score: ", model1.evaluate(X_train, y_train))
print("Test Score: ", model1.evaluate(X_test, y_test))


# Plotting the Model.fit

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='Val_acc')
plt.legend()
plt.show()
