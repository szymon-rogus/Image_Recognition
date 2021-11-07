from cv2.cv2 import *

image = imread('pipes.jpg', IMREAD_GRAYSCALE)
blurred = GaussianBlur(image, (9, 9), 0)
canny = Canny(blurred, 50, 255)

imshow('canny', canny)
waitKey(0)
destroyAllWindows()
