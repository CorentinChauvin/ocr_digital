#!/usr/bin/python3
"""
    Example script to show usage of the detector

    Author:  Corentin Chauvin-Hameau
    Date:    2021
    License: Apache-2.0 License
"""

import os
import cv2
from src.digital_detector import DigitalDetector


folder_path = "data/"
_, _, filenames = next(os.walk(folder_path))

detector = DigitalDetector()

for filename in filenames:
    if filename[-3:] != "jpg":
        continue

    image_path = os.path.join(folder_path, filename)
    img = cv2.imread(image_path)

    temperature = detector.detect_digits(img, display_debug=True)

    print("Temperature: ", temperature)
