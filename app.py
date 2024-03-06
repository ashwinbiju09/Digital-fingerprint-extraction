from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from datetime import datetime, timedelta
import hashlib
import cv2
import time
import numpy as np
import os
from utils import (
    detect_basic_shape,
    extract_initial_minutiae_patterns,
    enhance_initial_minutiae_patterns,
    crop_to_focus_roi,
    mask_roi_into_shape,
    invert_and_final_enhancement,
    get_metadata,
    time_check,
)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "static/images/uploads/"
app.config['RESULT_FOLDER'] = "static/images/results/"
app.config['SEARCH_FOLDER'] = "static/images/search/"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=['POST', 'GET'])
def process():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        date_taken = get_metadata(file_path)
        if time_check(date_taken):

            roi_image = extract_initial_minutiae_patterns(file_path)
            enhanced_image = enhance_initial_minutiae_patterns(roi_image)
            cropped_image = crop_to_focus_roi(enhanced_image)
            shape_mask = cv2.imread(
                    'static/images/model/comp.png', cv2.IMREAD_UNCHANGED)
            masked_image = mask_roi_into_shape(cropped_image, shape_mask)
            result_image = invert_and_final_enhancement(masked_image)

            path = os.path.join(app.config['RESULT_FOLDER'], filename)
            cv2.imwrite(path, result_image)

        else:

            error_msg = "No date taken information found or the image was not taken within the last 12 hours."
            return render_template("result.html", error=error_msg)

    return redirect(url_for("result", filename=filename))


@app.route("/result/<filename>")
def result(filename):
    return render_template("result.html", filename=filename)

@app.route("/search", methods=['POST', 'GET'])
def search():
    if 'file' not in request.files:
        return render_template('search.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('search.html', error='No selected file')

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['SEARCH_FOLDER'], filename)
        file.save(file_path)
        fingerprint_test = cv2.imread(file_path)

    for file in [file for file in os.listdir("./static/images/data")]:
        fingerprint_database_image = cv2.imread("./static/images/data/"+file)
        sift = cv2.SIFT_create()
        keypoints_1, descriptors_1 = sift.detectAndCompute(
            fingerprint_test, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(
            fingerprint_database_image, None)
        matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(
            descriptors_1, descriptors_2, k=2)
        match_points = []

        for p, q in matches:
            if p.distance < 0.1*q.distance:
                match_points.append(p)
            keypoints = 0
        if len(keypoints_1) <= len(keypoints_2):
            keypoints = len(keypoints_1)
        else:
            keypoints = len(keypoints_2)

        if (len(match_points) / keypoints) > 0.80:
            print("% match: ", len(match_points) / keypoints * 100)
            value = len(match_points) / keypoints * 100
            print("Fingerprint ID: " + str(file))
            file = str(file)

            return redirect(url_for("output", value=value, file=file))

@app.route("/output/<value>/<file>")
def output(value, file):
    return render_template("output.html", value=value, file=file)

if __name__ == "__main__":
    app.run(port=8008, debug=True)
