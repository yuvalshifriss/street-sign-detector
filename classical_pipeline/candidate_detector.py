import cv2

from .color_filtering import combine_color_masks
from .edge_and_contours import get_edges, find_contours
from utils.func_utils import show

def get_candidate_regions(image_bgr, output_path, min_area=300):
    """
    Detect potential traffic sign regions based on color + shape.

    Returns a list of bounding boxes: (x, y, w, h)
    """
    # Step 1: Combine red, blue, and white masks
    color_mask = combine_color_masks(image_bgr, use_red=True, use_blue=True, use_white=True)
    if output_path:
        show(color_mask)
    # Step 2: Apply color mask
    masked = cv2.bitwise_and(image_bgr, image_bgr, mask=color_mask)
    if output_path:
        show(masked)
    # Step 3: Edge detection
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    edges = get_edges(gray)
    if output_path:
        show(edges)
    # Step 4: Contour detection
    contours = find_contours(edges)
    # Step 5: Extract bounding boxes for large enough contours
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            candidates.append((x, y, w, h))

    return candidates
