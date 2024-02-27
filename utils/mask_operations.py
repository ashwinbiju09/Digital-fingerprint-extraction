import cv2
import numpy as np
from .shape_detection import detect_basic_shape


def mask_roi_into_shape(image, shape_mask):
    shape_of_image2 = detect_basic_shape(shape_mask)

    # Create a binary mask based on the shape of "shape_mask"
    mask = np.zeros_like(image, dtype=np.uint8)
    if shape_of_image2 == 'Ellipse':
        left_spacing = 30  # Adjust this value as needed
        center_x = shape_mask.shape[1] // 2 + left_spacing
        center_y = shape_mask.shape[0] // 2
        cv2.ellipse(mask, (center_x, center_y),
                    (shape_mask.shape[1] // 2, shape_mask.shape[0] // 2),
                    0, 0, 360, (255, 255, 255), -1)
    else:
        pass

    # Apply the mask for cropping
    return cv2.bitwise_and(image, mask)
