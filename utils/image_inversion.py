import cv2
import numpy as np
import fingerprint_enhancer


def invert_and_final_enhancement(image):
    temp = fingerprint_enhancer.enhance_Fingerprint(image)
    return cv2.bitwise_not(temp)
