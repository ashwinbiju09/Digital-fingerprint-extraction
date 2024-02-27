import cv2
import numpy as np


def detect_basic_shape(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours to identify basic shapes
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check for basic shapes (circle, ellipse, rectangle)
        if len(approx) >= 5:
            return 'Ellipse'  # Ellipse or circle
        elif len(approx) == 4:
            return 'Rectangle'  # Rectangle

    return 'Complex'  # If no basic shape is detected, consider it complex
