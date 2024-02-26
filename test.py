import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as platform



tf.__version__
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
X_train_full.shape
X_train_full.dtype
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
               "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_names[y_train[0]]


# model=keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.summary()
model.layers
model.layers[0].name
model.get_layer('dense_9').name

weights, biases = model.get_layer('dense_9').get_weights()
weights
biases.shape


model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd", metrics=["accuracy"])


history = model.fit(X_train, y_train, epochs=50,
                    validation_data=(X_valid, y_valid))


pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()

model.evaluate(X_test, y_test)


X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)
