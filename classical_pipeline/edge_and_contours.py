import cv2

def get_edges(image_gray, low_threshold=50, high_threshold=150):
    return cv2.Canny(image_gray, low_threshold, high_threshold)

def find_contours(edge_img):
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
