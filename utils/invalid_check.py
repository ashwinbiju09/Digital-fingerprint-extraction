import cv2
import numpy as np
import os

def dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def image_similarity(image_path1, image_path2, cutoff=5):
    try:
        # Read images using OpenCV
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        # Compute the dHash for both images
        hash1 = dhash(img1)
        hash2 = dhash(img2)

        # Compare dHashes
        return bin(hash1 ^ hash2).count('1') < cutoff
    except (cv2.error, Exception):
        return False  # Unable to open the file or process the image

def folder_check(image_path, folder_path, cutoff=5):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png')):
            full_path = os.path.join(folder_path, filename)
            if image_similarity(image_path, full_path, cutoff):
                return True
    return False
