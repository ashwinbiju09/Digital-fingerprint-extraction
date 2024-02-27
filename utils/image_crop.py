import cv2
import numpy as np


def crop_to_focus_roi(image, crop_width=250, crop_height=250):
    height, width = image.shape[:2]

    # Calculate the center coordinates
    center_x = int(width / 2)
    center_y = int(height / 2)

    # Calculate the starting coordinates for the crop
    start_x = max(0, center_x - int(crop_width / 2))
    start_y = max(0, center_y - int(crop_height / 2))
    end_x = min(start_x + crop_width, width)
    end_y = min(start_y + crop_height, height)

    # Crop the image
    return image[start_y:end_y, start_x:end_x]
