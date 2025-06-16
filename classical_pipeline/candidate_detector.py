import cv2
import os
import logging

from .color_filtering import combine_color_masks
from .edge_and_contours import get_edges, find_contours

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_candidate_regions(image_bgr, min_area=300, save_path=None):
    """
    Detect potential traffic sign regions based on color and shape.
    Returns a list of bounding boxes: (x, y, w, h)

    If save_path is provided, saves the annotated image with candidate boxes.
    """

    # Step 1: Combine red, blue, and white masks
    color_mask = combine_color_masks(image_bgr, use_red=True, use_blue=True, use_white=True)
    if save_path:
        cur_save_path = save_path.replace('.png', '_color_mask.png')
        cv2.imwrite(cur_save_path, color_mask)
        logging.info(f"Saved annotated image to: {cur_save_path}")

    # Step 2: Apply color mask
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=color_mask)
    if save_path:
        cur_save_path = save_path.replace('.png', '_masked_image.png')
        cv2.imwrite(cur_save_path, color_mask)
        logging.info(f"Saved annotated image to: {cur_save_path}")

    # Step 3: Edge detection
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    edges = get_edges(gray)
    if save_path:
        cur_save_path = save_path.replace('.png', '_edges.png')
        cv2.imwrite(cur_save_path, color_mask)
        logging.info(f"Saved annotated image to: {cur_save_path}")

    # Step 4: Contour detection
    contours = find_contours(edges)

    # Step 5: Extract bounding boxes for large enough contours
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            candidates.append((x, y, w, h))

    # Optional: Save the annotated image
    if save_path:
        annotated = image_bgr.copy()
        for (x, y, w, h) in candidates:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, annotated)

    return candidates
