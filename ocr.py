#!/usr/bin/python3
"""
TODO

https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
"""
import cv2
from src.digital_detector import DigitalDetector


# image_path = "data/image_12.jpg"
# img = cv2.imread(image_path)

# detector = DigitalDetector()
# temperature = detector.detect_digits(img)

# print("Temperature: ", temperature)


import os

folder_path = "data/OCR/2021_02_19-09_39_11 - Sample 7"
# folder_path = "data/OCR/2021_02_19-14_08_26 - Sample D"
# folder_path = "data/OCR/2021_02_19-11_52_24 - Sample 3"
# folder_path = "data/"
_, _, filenames = next(os.walk(folder_path))

detector = DigitalDetector()

k = 0
l = 0

for filename in filenames:
    if filename[-3:] != "jpg":
        continue

    image_path = os.path.join(folder_path, filename)
    img = cv2.imread(image_path)

    temperature = detector.detect_digits(img)

    print("Temperature: ", temperature)

    from math import isnan
    if isnan(temperature):
        print(filename)
        k += 1
    else:
        l += 1

print(k, l)
