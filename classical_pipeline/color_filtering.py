import cv2
import numpy as np
from functools import reduce
from typing import Optional


def filter_red(image_bgr: np.ndarray) -> np.ndarray:
    """
    Creates a binary mask that isolates red regions in the input BGR image.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Binary mask where red pixels are set to 255 and others to 0.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    return cv2.bitwise_or(
        cv2.inRange(image_hsv, lower_red1, upper_red1),
        cv2.inRange(image_hsv, lower_red2, upper_red2)
    )


def filter_blue(image_bgr: np.ndarray) -> np.ndarray:
    """
    Creates a binary mask that isolates blue regions in the input BGR image.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Binary mask where blue pixels are set to 255 and others to 0.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 100, 50])
    upper_blue = np.array([130, 255, 255])
    return cv2.inRange(image_hsv, lower_blue, upper_blue)


def filter_white(image_bgr: np.ndarray) -> np.ndarray:
    """
    Creates a binary mask that isolates white regions in the input BGR image.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Binary mask where white pixels are set to 255 and others to 0.
    """
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    return cv2.inRange(image_hsv, lower_white, upper_white)


def combine_color_masks(
    image_bgr: np.ndarray,
    use_red: bool = True,
    use_blue: bool = True,
    use_white: bool = True
) -> np.ndarray:
    """
    Combines binary masks for red, blue, and white regions in an image based on specified flags.

    Args:
        image_bgr (np.ndarray): Input image in BGR format.
        use_red (bool): Whether to include red mask. Default is True.
        use_blue (bool): Whether to include blue mask. Default is True.
        use_white (bool): Whether to include white mask. Default is True.

    Returns:
        np.ndarray: Combined binary mask. Pixels that match any selected color are set to 255.
    """
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
