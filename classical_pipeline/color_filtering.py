import cv2
import numpy as np
from functools import reduce

def filter_red(image_bgr):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    return cv2.bitwise_or(
        cv2.inRange(image_hsv, lower_red1, upper_red1),
        cv2.inRange(image_hsv, lower_red2, upper_red2)
    )

def filter_blue(image_bgr):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])
    return cv2.inRange(image_hsv, lower_blue, upper_blue)

def filter_white(image_bgr):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    return cv2.inRange(image_hsv, lower_white, upper_white)

def combine_color_masks(image_bgr, use_red=True, use_blue=True, use_white=True):
    masks = []
    if use_red:
        masks.append(filter_red(image_bgr))
    if use_blue:
        masks.append(filter_blue(image_bgr))
    if use_white:
        masks.append(filter_white(image_bgr))
    if masks:
        return reduce(cv2.bitwise_or, masks)
    return np.zeros(image_bgr.shape[:2], dtype=np.uint8)
