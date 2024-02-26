
# '''
#     import os
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

# print(f'Using {device} device')


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits


# model = NeuralNetwork().to(device)
# print(model)

# X = torch.rand(1, 28, 28, device=device)
# logits = model(X)
# pred_prob = nn.Softmax(dim=1)(logits)
# y_pred = pred_prob.argmax(1)
# print(f"Predicted Class : {y_pred}")

# '''

# '''
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import random

# SEED = 1234

# np.random.seed(SEED)

# random.seed(SEED)


# url = "https://github.com/reisanar/datasets/blob/master/spiral.csv"

# df = pd.read_csv(url, header=0)  # load
# df = df.sample(frac=1).reset_index(drop=True)  # shuffle
# df.head()
# '''

# import tensorflow as tf
# from tensorflow import keras
# from sklearn.preprocessing import StandardScaler
# print(tf.__version__)

# fashion_mnist = keras.datasets.fashion_mnist
# (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
# print(X_train_full.size)
# print(X_train_full.dtype)
# print(y_train_full.size)


# X_valid, X_train = X_train_full[:5000]/255.0, X_train_full[5000:]/255.0

# y_valid, y_train = y_train_full[:5000]/255.0, y_train_full[5000:]/255.0


# class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress",
#                "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankel Boot"]

# model = keras.models.Sequential()
# model.add(keras.layers.Flatten(input_shape=[28, 28]))
# model.add(keras.layers.Dense(300, activation="relu"))
# model.add(keras.layers.Dense(100, activation="relu"))
# model.add(keras.layers.Dense(10, activation="softmax"))

# print(model.summary())

# model.compile(loss="spare_categorical_crossentropy",
#               optimizer="sgd",
#               metrics=["accuracy"]
#               )

# # history=


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Softmax
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

print(f"{train_images}\t\t\t\t\t {train_labels}")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

len(train_labels)
train_labels
test_images.shape
len(test_labels)


#  Preprocess data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


train_images = train_images/255.0
test_images = test_images/255.0

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()


# Set up the model

model = Sequential(
    [
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10)
    ]
)


# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model

model.fit(train_images, train_labels, epochs=10)


#  Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest Accuracy", test_acc)


probability_model = Sequential(
    [
        model, Softmax()
    ]
)

predictions=probability_model.predict(test_images)

predictions[0] # A prediction is an array of 10 numbers. They represent the model's "confidence" that the image corresponds to each of the 10 different articles of clothing. You can see which label has the highest confidence value:

np.argmax(predictions[0])

test_labels[0]
