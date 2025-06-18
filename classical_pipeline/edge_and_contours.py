import cv2
import numpy as np
from typing import List

def get_edges(image_gray: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Applies Canny edge detection to a grayscale image.

    Args:
        image_gray (np.ndarray): Grayscale input image.
        low_threshold (int): Lower threshold for Canny hysteresis. Default is 50.
        high_threshold (int): Upper threshold for Canny hysteresis. Default is 150.

    Returns:
        np.ndarray: Binary image with edges detected.
    """
    return cv2.Canny(image_gray, low_threshold, high_threshold)


def find_contours(edge_img: np.ndarray) -> List[np.ndarray]:
    """
    Finds contours from a binary edge image.

    Args:
        edge_img (np.ndarray): Binary edge image (e.g., output of Canny).

    Returns:
        List[np.ndarray]: List of contours, each represented by a numpy array of points.
    """
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
