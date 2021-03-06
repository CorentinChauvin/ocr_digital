#!/usr/bin/python3
"""
TODO

https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
"""
import cv2
from src.digital_detector import DigitalDetector

# import sys
# np.set_printoptions(threshold=sys.maxsize, linewidth=1000)


image_path = "data/image_1.jpg"
img = cv2.imread(image_path)

detector = DigitalDetector()
detector.detect(img)

# TODO: fix screen segmentation
# TODO: put the text on the original image
# TODO: handle videos and live streams
