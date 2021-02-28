#!/usr/bin/python3
"""
TODO

https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
"""
from math import atan2, degrees, pi
import numpy as np
import cv2
import imutils

# import sys
# np.set_printoptions(threshold=sys.maxsize, linewidth=1000)


image_path = "data/image_3.jpg"
img = cv2.imread(image_path)

# Post process the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img, (15, 15), 1)
edges = cv2.Canny(blurred, 40, 60)

# Find the screen of the thermometer
screen_rectangle = None
contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)

    if len(approx) != 4:
        continue

    p0 = approx[0]
    p1 = approx[1]
    p2 = approx[2]
    l1 = np.linalg.norm(p0 - p1)
    l2 = np.linalg.norm(p1 - p2)

    if l1 < l2:
        l1, l2 = l2, l1

    if l2 < 10 or l1 / l2 > 1.4:
        continue

    cv2.drawContours(img, [approx], -1, (0, 255, 0), 1)
    screen_rectangle = approx[:, 0, :]

    break  # don't need to get dupplicates

# Compensate for potential thermometer orientation
e1 = screen_rectangle[0] - screen_rectangle[1]
e2 = screen_rectangle[1] - screen_rectangle[2]

if np.linalg.norm(e1) > np.linalg.norm(e2):
    h_edge = e1
else:
    h_edge = e2

orientation = atan2(h_edge[1], h_edge[0])

while orientation < -pi/2 or orientation > pi/2:
    if orientation < -pi/2:
        orientation += pi
    else:
        orientation -= pi

image_center = tuple(np.array(img.shape[1::-1]) / 2)
rot_mat = cv2.getRotationMatrix2D(image_center, degrees(orientation), 1.0)
rotated_img = cv2.warpAffine(gray, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

homogeneous_rectangle = np.ones((3, 4))
homogeneous_rectangle[0:2, :] = screen_rectangle.transpose()
rotated_screen_rectangle = np.matmul(rot_mat, homogeneous_rectangle).transpose()

# Crop the screen
x0 = int(np.min(rotated_screen_rectangle[:, 0]))
x1 = int(np.max(rotated_screen_rectangle[:, 0]))
y0 = int(np.min(rotated_screen_rectangle[:, 1]))
y1 = int(np.max(rotated_screen_rectangle[:, 1]))

screen_img = rotated_img[y0:y1, x0:x1]

# Threshold the image and remove the edges
thresh = cv2.adaptiveThreshold(
    screen_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    19, 4
)

H, W = np.shape(thresh)
labels_nbr, labels = cv2.connectedComponents(255-thresh, connectivity=8,)

def clean_connected(i, j, thresh):
    if thresh[i, j] == 0:
        label = labels[i, j]
        thresh[labels == label] = 255

for i in range(H):
    clean_connected(i, 0, thresh)
    clean_connected(i, W-1, thresh)

for j in range(W):
    clean_connected(0, j, thresh)
    clean_connected(H-1, j, thresh)

# Locate the characters
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
thresh = cv2.morphologyEx(thresh.copy(), cv2.MORPH_OPEN, kernel)

debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
digit_contours = []

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)

    if 0.15 <= float(w) / W < 0.25:
        new_contour = True

        for k in range(len(digit_contours)):
            [x_other, y_other, w_other, h_other] = digit_contours[k]

            if x_other <= x + w/2.0 <= x_other + w_other:
                new_x1 = min(x, x_other)
                new_y1 = min(y, y_other)
                new_x2 = max(x + w, x_other + w_other)
                new_y2 = max(y + h, y_other + h_other)
                digit_contours[k] = (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)
                new_contour = False

        if new_contour:
            digit_contours.append((x, y, w, h))

for (x, y, w, h) in digit_contours:
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

# Detect the segments
def check_segment(x, y, w, h):
    black_pixels_nbr = np.count_nonzero(thresh[y:y+h, x:x+w] == 0)
    ratio = float(black_pixels_nbr) / (h * w)

    return ratio > 0.3

def convert_to_int(binary_string):
    integers = [
        "01111101",
        "01100000",
        "00110111",
        "01100111",
        "01101010",
        "01001010",
        "01011111",
        "01100001",
        "01111111",
        "01101111"
    ]
    integers = [int(x, 2) for x in integers[:]]
    value = int(binary_string, 2)
    return integers.index(value)

for (x, y, w, h) in digit_contours:
    d = int(0.3 * w)
    binary_string = ""

    # Horizontal segments
    for k in range(3):
        x_segment = x + d
        y_segment = y + int(k/2.0 * (h - d))
        if check_segment(x_segment, y_segment, w - 2*d, d):
            cv2.rectangle(debug_img, (x_segment, y_segment), (x_segment + w - 2*d, y_segment + d), (255, 0, 0), 1)
            binary_string =  "1" + binary_string
        else:
            binary_string =  "0" + binary_string

    # Vertical segments
    for k in range(2):
        x_segment = x + int(k * (w - d))

        for l in range(2):
            y_segment = y + int(l * h / 2)
            if check_segment(x_segment, y_segment, d, h//2):
                cv2.rectangle(debug_img, (x_segment, y_segment), (x_segment + d, y_segment + h//2), (255, 0, 0), 1)
                binary_string =  "1" + binary_string
            else:
                binary_string =  "0" + binary_string

    # Display the digit
    digit = convert_to_int(binary_string)

    font = cv2.FONT_HERSHEY_SIMPLEX
    debug_img = cv2.putText(debug_img, str(digit), (x + w//2, int(0.4*H)), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

# TODO: handle ones
# TODO: put the text on the original image
# TODO: refactor as class
# TODO: handle videos and live streams


# Display the result
cv2.imshow('image', debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
