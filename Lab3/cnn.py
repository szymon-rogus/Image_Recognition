import numpy as np
import tensorflow.keras.datasets.cifar10 as cf10
from tensorflow.keras import models
from model import CfModel
from numpy import save
import tensorflow as tf
from cv2.cv2 import *


def assert_dataset(train, test, train_label, test_label):
    assert train.shape == (50000, 32, 32, 3)
    assert test.shape == (10000, 32, 32, 3)
    assert train_label.shape == (50000, 1)
    assert test_label.shape == (10000, 1)


def test_cnn(cnn, dataset, labels, labels_vector):
    predictions = cnn.predict(dataset)
    counter = 0
    for i in range(len(predictions)):
        print('Prediction: {}, Actual: {}'.format(predictions[i], labels_vector[i]))
        if predictions[i][labels[i]] > 0.5:
            counter += 1

    print('Images that were predicted correctly: {}/{}'.format(counter, len(labels)))


np.set_printoptions(linewidth=200)
(x_train, y_train), (x_test, y_test) = cf10.load_data()
assert_dataset(x_train, x_test, y_train, y_test)

y_train_vector = np.zeros(shape=(50000, 10))
for i in range(len(y_train_vector)):
    value = y_train[i]
    y_train_vector[i][value] = 1.0

y_test_vector = np.zeros(shape=(10000, 10))
for i in range(len(y_test_vector)):
    value = y_test[i]
    y_test_vector[i][value] = 1.0

model = CfModel()
model.create_architecture()
model.compile()

model.get_summary()
history = model.fit(x_train, y_train_vector, x_test, y_test_vector)
model.save_model('my_model4.h5')
save('history8', history.history)
# model = models.load_model('my_model3.h5')

test_cnn(model, x_test, y_test, y_test_vector)
