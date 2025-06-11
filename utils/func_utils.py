import cv2
import os
import uuid

def show(image, title="Image", wait_time=0, save_if_headless=False, save_dir="/tmp"):
    """
    Display an image in a window or save to disk if running headless.

    Args:
        image: The image to display (numpy array).
        title: Title of the window (or filename prefix if saving).
        wait_time: Milliseconds to wait for key press (0 = wait forever).
        save_if_headless: If True and no display available, image will be saved to disk instead.
        save_dir: Directory to save image if headless.
    """
    try:
        cv2.imshow(title, image)
        cv2.waitKey(wait_time)
        cv2.destroyAllWindows()
    except cv2.error:
        if save_if_headless:
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{title.replace(' ', '_')}_{uuid.uuid4().hex[:6]}.png"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, image)
            print(f"[Headless Mode] Saved image to: {filepath}")
        else:
            print("[Error] Cannot display image. Possibly running in headless mode. Use save_if_headless=True.")
