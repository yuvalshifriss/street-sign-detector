import cv2
import os
import logging
from typing import List, Tuple, Optional

from .color_filtering import combine_color_masks
from .edge_and_contours import get_edges, find_contours

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_candidate_regions(
        image_bgr: 'cv2.typing.MatLike',
        min_area: int = 25,
        save_path: Optional[str] = None
) -> List[Tuple[int, int, int, int]]:
    """
    Detect potential traffic sign regions in an image using color masking, edge detection,
    and contour extraction.

    Args:
        image_bgr (cv2.typing.MatLike): The input image in BGR format.
        min_area (int, optional): Minimum contour area to be considered a valid region. Default is 25.
        save_path (str, optional): If provided, intermediate and final annotated images will be saved
                                   to this path (e.g., 'output/image.png') with suffixes for each step.

    Returns:
        List[Tuple[int, int, int, int]]: A list of bounding boxes, each represented as
                                         (x, y, width, height), corresponding to candidate regions.
    """

    # Step 1: Create a combined mask from red, blue, and white filters
    color_mask = combine_color_masks(image_bgr, use_red=True, use_blue=True, use_white=True)
    if save_path:
        cur_save_path = save_path.replace('.png', '_color_mask.png')
        cv2.imwrite(cur_save_path, color_mask)
        logging.info(f"Saved color mask to: {cur_save_path}")

    # Step 2: Apply the color mask to the image
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=color_mask)
    if save_path:
        cur_save_path = save_path.replace('.png', '_masked_image.png')
        cv2.imwrite(cur_save_path, masked)
        logging.info(f"Saved masked image to: {cur_save_path}")

    # Step 3: Convert to grayscale and detect edges
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    edges = get_edges(gray)
    if save_path:
        cur_save_path = save_path.replace('.png', '_edges.png')
        cv2.imwrite(cur_save_path, edges)
        logging.info(f"Saved edges image to: {cur_save_path}")

    # Step 4: Find contours from the edge-detected image
    contours = find_contours(edges)

    # Step 5: Filter contours based on area and extract bounding boxes
    candidates: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            candidates.append((x, y, w, h))

    # Step 6: Save final annotated image with candidate bounding boxes
    if save_path:
        annotated = image_bgr.copy()
        for (x, y, w, h) in candidates:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, annotated)
        logging.info(f"Saved annotated image to: {save_path}")

    return candidates
