import cv2
import numpy as np
HEIGHT_IMAGE = 118
WIDTH_IMAGE = 2167


def process_image(pil_img):
    img_file = np.array(pil_img)
    img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    img = cv2.resize(img, (int(HEIGHT_IMAGE / height * width), HEIGHT_IMAGE))

    height, width = img.shape

    img = np.pad(img, ((0, 0), (0, WIDTH_IMAGE - width)), 'median')

    img = cv2.GaussianBlur(img, (5, 5), 0)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    img = np.expand_dims(img, axis=2)

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    return img
