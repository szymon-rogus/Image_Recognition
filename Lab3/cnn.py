import numpy as np
import tensorflow.keras.datasets.cifar10 as cf10
from tensorflow.keras import datasets, layers, models, losses, optimizers
import matplotlib.pyplot as plt
from numpy import save
import tensorflow as tf
from cv2.cv2 import *

np.set_printoptions(linewidth=200)
(x_train, y_train), (x_test, y_test) = cf10.load_data()

assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

y_train_vector = np.zeros(shape=(50000, 10))
for i in range(len(y_train_vector)):
    value = y_train[i]
    y_train_vector[i][value] = 1.0

y_test_vector = np.zeros(shape=(10000, 10))
for i in range(len(y_test_vector)):
    value = y_test[i]
    y_test_vector[i][value] = 1.0

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.experimental.preprocessing.Rescaling(scale=1./255, offset=0.0))
# model.add(layers.Conv2D(5, (3, 3), activation='sigmoid'))
# model.add(layers.Conv2D(5, (3, 3), activation='sigmoid', input_shape=(28, 28, 5)))
# model.add(layers.MaxPooling2D((8, 8)))
# model.add(layers.Flatten())
# model.add(layers.Dense(10, activation='softmax'))
#
# model.summary()
#
# opt = optimizers.SGD(learning_rate=0.001, momentum=0.9)
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
#
# history = model.fit(x_train, y_train_vector, epochs=150, batch_size=64, validation_data=(x_test, y_test_vector))
# model.save('my_model.h5')
# save('history', history.history)

model = models.load_model('my_model.h5')

predictions = model.predict(x_test)
counter = 0
for i in range(len(predictions)):
    print('Prediction: {}, Actual: {}'.format(predictions[i], y_test_vector[i]))
    if predictions[i][y_test[i]] > 0.5:
        counter += 1

print('Images that were predicted correctly: {}/{}'.format(counter, len(y_test)))
