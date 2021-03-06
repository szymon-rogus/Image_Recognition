import numpy as np
import tensorflow as tf
from cv2.cv2 import *
import math

WIDTH = 976
HEIGHT = 600

SOBEL_X = np.array((1, 0, -1, 2, 0, -2, 1, 0, -1))
SOBEL_Y = np.array((1, 2, 1, 0, 0, 0, -1, -2, -1))


def preprocess_image(source):
    image = imread(source, IMREAD_COLOR)
    image = normalize(image, image, 0, 1, NORM_MINMAX, CV_32F)
    gray_image = np.zeros((HEIGHT, WIDTH))
    for x in range(len(image)):
        for y in range(len(image[0])):
            pixel = (image[x][y][0] + image[x][y][1] + image[x][y][2])/3
            gray_image[x][y] = pixel
    return gray_image


def get_gaussian_kernel(size):
    assert size % 2 == 1
    kernel = np.zeros((size, size))
    center = int((size - 1) / 2)
    kernel[center, center] = 1.0
    return GaussianBlur(kernel, (size, size), 1.0, borderType=BORDER_ISOLATED)


def gaussian_blur(image, kernel):
    image = image[tf.newaxis, :, :, tf.newaxis]
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    tensor = tf.nn.conv2d(image, kernel, strides=1, padding='SAME')
    return np.reshape(tensor, (HEIGHT, WIDTH))


def sobel_filter(image, sobel_vector):
    image = image[tf.newaxis, :, :, tf.newaxis]
    sobel = sobel_vector.reshape((3, 3))
    sobel = sobel[:, :, tf.newaxis, tf.newaxis]
    tensor = tf.nn.conv2d(image, sobel, strides=1, padding='SAME')
    return np.reshape(tensor, (HEIGHT, WIDTH))


def combine_sobel_x_y(image_x, image_y):
    edges = np.zeros((HEIGHT, WIDTH))
    for x in range(len(image_x)):
        for y in range(len(image_x[0])):
            edges[x][y] = math.sqrt(image_x[x][y]**2 + image_y[x][y]**2)
    return edges


def threshold_image(image, threshold_value):
    for x in range(len(image)):
        for y in range(len(image[0])):
            if image[x][y] < threshold_value:
                image[x][y] = 0.0
    return normalize(image, image, 0, 1, NORM_MINMAX, CV_32F)


def lower_resolution(image, size):
    image = image[:, :, tf.newaxis, tf.newaxis]
    image = tf.nn.max_pool(image, size, strides=1, padding='SAME')
    return np.reshape(image, (HEIGHT, WIDTH))


def find_centres(image, size):
    image = image[tf.newaxis, :, :, tf.newaxis]
    filter_matrix = np.zeros((size, size))
    for x in range(len(filter_matrix)):
        for y in range(len(filter_matrix[0])):
            if x == 0 or y == 0 or x == size - 1 or y == size - 1:
                filter_matrix[x][y] = 1
    filter_matrix = filter_matrix[:, :, tf.newaxis, tf.newaxis]
    image = tf.nn.conv2d_transpose(image, filter_matrix, strides=1, padding='SAME', output_shape=(HEIGHT, WIDTH))
    image = np.reshape(image, (HEIGHT, WIDTH))
    return normalize(image, image, 0, 1, NORM_MINMAX, CV_32F)


gray_image = preprocess_image('pipes.jpg')
blurred_image = gaussian_blur(gray_image, get_gaussian_kernel(5))
sobel_image = combine_sobel_x_y(sobel_filter(blurred_image, SOBEL_X), sobel_filter(blurred_image, SOBEL_Y))
sobel_image = threshold_image(sobel_image, 0.8)
sobel_image = lower_resolution(sobel_image, 16)

centers = []
for i in range(7, 17, 2):
    centers.append(find_centres(sobel_image, i))

for i in range(len(centers)):
    centers[i] = threshold_image(centers[i], 0.5)

counter = 7
for i in range(len(centers)):
    centers[i] = find_centres(centers[i], counter)
    counter = counter + 2

for i in centers:
    imshow('pipes', i)
    waitKey(0)

destroyAllWindows()
