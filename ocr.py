#!/usr/bin/python3
"""
TODO

https://www.pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/
"""
from math import atan2, degrees, pi
import numpy as np
import cv2
import imutils


image_path = "data/image_1.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Post process the image
img = imutils.resize(img, width=1000)
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

    # cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
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
img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

homogeneous_rectangle = np.ones((3, 4))
homogeneous_rectangle[0:2, :] = screen_rectangle.transpose()
rotated_screen_rectangle = np.matmul(rot_mat, homogeneous_rectangle).transpose()

# Crop the screen
x0 = int(np.min(rotated_screen_rectangle[:, 0]))
x1 = int(np.max(rotated_screen_rectangle[:, 0]))
y0 = int(np.min(rotated_screen_rectangle[:, 1]))
y1 = int(np.max(rotated_screen_rectangle[:, 1]))

screen_img = img[y0:y1, x0:x1]

# Detect the characters
thresh = cv2.adaptiveThreshold(screen_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY, 19, 4)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

debug_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
H, W = np.shape(screen_img)

# (x0, y0) = (int(0.80*W), int(0.49*H))
# (x1, y1) = (int(0.98*W), int(0.95*H))
# cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 1)
# (x0, y0) = (int(0.55*W), int(0.49*H))
# (x1, y1) = (int(0.73*W), int(0.95*H))
# cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 1)
# (x0, y0) = (int(0.30*W), int(0.49*H))
# (x1, y1) = (int(0.49*W), int(0.95*H))
# cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 1)


cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
digitCnts = []
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    print(x, y, w, h)

    # if the contour is sufficiently large, it must be a digit
    # if w >= 15 and (h >= 30 and h <= 40):
    # 	digitCnts.append(c)


cv2.imshow('image', debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
