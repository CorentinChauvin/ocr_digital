"""
    Implementation of a simple digital digit detector for thermometers

    Author: Corentin Chauvin-Hameau
    Date: 2021
    License: TODO
"""

from math import atan2, degrees, pi
import numpy as np
import cv2
import imutils


class DigitalDetector:
    """
    """
    def __init__(self):
        pass

    #
    # Public member functions
    #
    def detect(self, img):
        """
        """
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        screen_img = self._crop_screen(gray_img)
        screen_img = cv2.resize(screen_img, (96, 120))
        thresh_img = self._threshold_screen(screen_img)

        digit_rectangles = self._detect_digits(thresh_img)
        digits, digits_position, segments_rectangles = \
            self._detect_segments(thresh_img, digit_rectangles)

        temperature = self._get_temperature(digits, digits_position)
        print(temperature)

        # Debug display
        self._debug_img = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2RGB)

        for (x, y, w, h) in digit_rectangles:
            cv2.rectangle(self._debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for k in range(len(digits)):
            digit = digits[k]
            position = digits_position[k]
            self._debug_img = cv2.putText(self._debug_img, str(digit), position, font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        for segment in segments_rectangles:
            cv2.rectangle(self._debug_img, segment[0], segment[1], (255, 0, 0), 1)

        self._debug_img = cv2.resize(self._debug_img, (360, 288))
        cv2.imshow('image', self._debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #
    # Private member functions
    #
    def _crop_screen(self, img):
        """
        Returns an image of the screen only
        """
        img = img.copy()
        screen_rectangle = self._find_screen(img)
        screen_rectangle, img = self._rotate_image(screen_rectangle, img)

        x0 = int(np.min(screen_rectangle[:, 0]))
        x1 = int(np.max(screen_rectangle[:, 0]))
        y0 = int(np.min(screen_rectangle[:, 1]))
        y1 = int(np.max(screen_rectangle[:, 1]))
        screen_img = img[y0:y1, x0:x1]

        return screen_img

    def _find_screen(self, img):
        """
        Returns the coordinates of the screen on the given image
        """
        blurred = cv2.GaussianBlur(img, (15, 15), 1)
        edges = cv2.Canny(blurred, 40, 60)

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
            e1 = (p0 - p1)[0]
            e2 = (p2 - p1)[0]
            l1 = np.linalg.norm(e1)
            l2 = np.linalg.norm(e2)

            angle = np.arccos(np.dot(e1, e2) / (l1 * l2))

            if l1 < l2:
                l1, l2 = l2, l1

            if l2 < 10 or l1 / l2 > 1.4:
                continue

            if not (pi/2 - 0.1 <= angle <= pi/2 + 0.1):
                continue

            cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
            screen_rectangle = approx[:, 0, :]

            return screen_rectangle  # don't need to get dupplicates

    def _rotate_image(self, screen_rectangle, img):
        """
        Rotates the image too compensate for any angle of the screen
        """
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
        rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

        homogeneous_rectangle = np.ones((3, 4))
        homogeneous_rectangle[0:2, :] = screen_rectangle.transpose()
        rotated_screen_rectangle = np.matmul(rot_mat, homogeneous_rectangle).transpose()

        return rotated_screen_rectangle, rotated_img

    def _threshold_screen(self, screen_img):
        """
        Thresholds the screen image, and remove the edges
        """
        screen_img = screen_img.copy()
        blurred = cv2.GaussianBlur(screen_img, (15, 15), 1)
        alpha = 1.5
        sharpened = cv2.addWeighted(screen_img, 1 + alpha, blurred, -alpha, 0.0)

        thresh = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            19, 4
        )

        H, W = np.shape(thresh)
        labels_nbr, labels = cv2.connectedComponents(255-thresh, connectivity=8)

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

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return thresh

    def _detect_digits(self, img):
        """
        Returns the rectangular bounding box of each digit on the given image
        """
        H, W = np.shape(img)
        labels_nbr, labels, stats, centroids = cv2.connectedComponentsWithStats(255-img, connectivity=8)

        digit_rectangles = []

        for label in range(1, labels_nbr):
            component = img[labels == label]

            x = stats[label][cv2.CC_STAT_LEFT]
            y = stats[label][cv2.CC_STAT_TOP]
            w = stats[label][cv2.CC_STAT_WIDTH]
            h = stats[label][cv2.CC_STAT_HEIGHT]

            if y / H > 0.3 and h / H >= 0.05:
                if h / H <= 0.1 and w/W <= 0.1:  # dot
                    continue

                new_rectangle = True

                for k in range(len(digit_rectangles)):
                    [x_other, y_other, w_other, h_other] = digit_rectangles[k]

                    if x_other <= x + w/2.0 <= x_other + w_other:
                        new_x1 = min(x, x_other)
                        new_y1 = min(y, y_other)
                        new_x2 = max(x + w, x_other + w_other)
                        new_y2 = max(y + h, y_other + h_other)
                        digit_rectangles[k] = (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)
                        new_rectangle = False

                if new_rectangle:
                    digit_rectangles.append((x, y, w, h))

        return digit_rectangles

    def _detect_segments(self, img, digit_rectangles):
        """
        Detects digit segments and returns the corresponding digit

        Args:
            - img:              Thresholded image of the screen
            - digit_rectangles: Bounding boxes of each digit
        Returns:
            - digits:              List of corresponding digits
            - digits_position:     List of position where to display the digit (for debug)
            - segments_rectangles: List of bounding boxes of each detected segment
        """
        H, W = np.shape(img)
        digits = []
        digits_position = []
        segments_rectangles = []

        for (x, y, w, h) in digit_rectangles:
            # Handle the case of ones
            if w / W <= 0.1 and h / H > 0.3:
                digits.append(1)
                digits_position.append((x, int(0.4*H)))

                continue

            # Horizontal segments
            binary_string = ""
            d = int(0.3 * w)

            for k in range(3):
                x_segment = x + d
                y_segment = y + int(k/2.0 * (h - d))
                if self._check_segment_state(img, x_segment, y_segment, w - 2*d, d):
                    segments_rectangles.append([
                        (x_segment, y_segment),
                        (x_segment + w - 2*d, y_segment + d)
                    ])
                    binary_string =  "1" + binary_string
                else:
                    binary_string =  "0" + binary_string

            # Vertical segments
            d = int(0.4 * w)
            for k in range(2):
                x_segment = x + int(k * (w - d))

                for l in range(2):
                    y_segment = y + int(l * h / 2)
                    if self._check_segment_state(img, x_segment, y_segment, d, h//2):
                        segments_rectangles.append([
                            (x_segment, y_segment),
                            (x_segment + d, y_segment + h//2)
                        ])
                        binary_string =  "1" + binary_string
                    else:
                        binary_string =  "0" + binary_string

            # Decode the digit
            digit = self._decode_segments(binary_string)
            digits.append(digit)
            digits_position.append((x, int(0.4*H)))

        return digits, digits_position, segments_rectangles

    def _check_segment_state(self, img, x, y, w, h):
        """
        Returns the state of the segment

        Args:
            - x, y: Coordinates of the top-left corner of the bounding box of the segment
            - w, h: Sizes of the bounding box of the segment
        Returns:
            - Whether the segment is switched on
        """
        black_pixels_nbr = np.count_nonzero(img[y:y+h, x:x+w] == 0)
        ratio = float(black_pixels_nbr) / (h * w)

        return ratio > 0.3

    def _decode_segments(self, binary_string):
        """
        Returns the digit corresponding to a string representing the state of each segment

        Returns numpy.NaN if the digit couldn't be decoded
        """
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

        if value in integers:
            return integers.index(value)
        else:
            return np.NaN

    def _get_temperature(self, digits, digits_position):
        """
        Returns the temperature indicated by the thermometer
        """
        digits = np.array(digits)
        x = np.array(digits_position)[:, 0]
        digits = digits[np.argsort(x)]

        string = ""
        for digit in digits:
            string += str(digit)

        return int(string) / 10
