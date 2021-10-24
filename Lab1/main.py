import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

np.set_printoptions(linewidth=200)

train_mask = np.array(list(map(lambda x: True if x == 3 or x == 5 else False, y_train)))
test_mask = np.array(list(map(lambda x: True if x == 3 or x == 5 else False, y_test)))

x_train = x_train[train_mask]
y_train = y_train[train_mask]
x_test = x_test[test_mask]
y_test = y_test[test_mask]

y_train = np.array(list(map(lambda x: 0 if x == 3 else 1, y_train)))
y_test = np.array(list(map(lambda x: 0 if x == 3 else 1, y_test)))

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

sgd = tf.keras.optimizers.SGD(learning_rate=0.1)
binary_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(loss=binary_loss, optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

loss, acc = model.evaluate(x_test, y_test)

predictions = model.predict(x_test)

epsilon = 0.01
counter = 0
for prediction, label in zip(predictions, y_test):
    print("Etykieta: {}, Prawdopodobieństwo: {}".format(label, prediction[0]))
    if abs(label - prediction[0]) < epsilon:
        counter = counter + 1

print("Poprawnie przewidziane wartości testowe: {}/{}".format(counter, len(y_test)))



# matrix_weights = tf.reshape(model.layers[2].weights[0], [28, 28])
# print(matrix_weights)
#
# plt.figure()
# plt.imshow(matrix_weights, cmap='gray')
# plt.show()