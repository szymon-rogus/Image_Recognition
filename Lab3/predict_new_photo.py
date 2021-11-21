import numpy as np
from keras import models
import cv2.cv2 as cv

np.set_printoptions(linewidth=200)

model = models.load_model('my_model4.h5')

image = cv.imread('plane.jpg')
image = cv.resize(image, (32, 18))

dataset = np.array([image])
predictions = model.predict(dataset)
print(predictions)
