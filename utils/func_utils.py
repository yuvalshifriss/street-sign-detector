import cv2

def show(image):
    cv2.imshow("Window Name", image)
    cv2.waitKey(0)  # Waits for a key press to close the window
    cv2.destroyAllWindows()