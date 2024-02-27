import cv2
import numpy as np


def extract_initial_minutiae_patterns(file_path):
    img = cv2.imread(file_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    details = cv2.Canny(blurred, 10, 15)
    edged = cv2.Canny(blurred, 10, 250)

    # Extract finger boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edged, kernel)
    cnts, _ = cv2.findContours(
        dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)

    # Mask layer to crop outside
    black = np.zeros(edged.shape, dtype=np.uint8)
    cv2.drawContours(black, [contour], -1, (255, 255, 255), -1)
    cv2.floodFill(black, None, (0, 0), (255, 255, 255))

    result = cv2.bitwise_not(cv2.bitwise_or(details, black))
    ROI = result[y:y+h, x:x+w]
    cv2.floodFill(ROI, None, (0, 0), (255, 255, 255))

    return ROI
