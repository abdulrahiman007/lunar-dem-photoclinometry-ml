'''import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import cv2
import numpy as np
import joblib

from ml.feature_extractor import extract_features
from logic.gradient_map import compute_surface_gradients

def refine_dem(image_path, dem_path, out_path):
    img = cv2.imread(image_path, 0) / 255.0
    dem = cv2.imread(dem_path, 0) / 255.0

    gx, gy = compute_surface_gradients(img)
    X = extract_features(img, dem, gx, gy)

    rf = joblib.load("ml/rf_model.pkl")
    residual = rf.predict(X).reshape(dem.shape)

    refined = dem + residual
    refined = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(out_path, refined.astype("uint8"))
'''
'''import cv2
import numpy as np
import joblib
from logic.gradient_map import compute_surface_gradients
from ml.feature_extractor import extract_features


def refine_dem(input_image_path, gray_dem_path, output_path):
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    dem = cv2.imread(gray_dem_path, cv2.IMREAD_GRAYSCALE)

    if img is None or dem is None:
        raise RuntimeError("Image or DEM not found for RF refinement.")

    img = img.astype("float32") / 255.0
    dem = dem.astype("float32") / 255.0

    p, q = compute_surface_gradients(img)
    X = extract_features(img, dem, p, q)

    rf = joblib.load("ml/rf_model.pkl")
    residual = rf.predict(X).reshape(dem.shape)

    refined = dem + residual
    refined = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite(output_path, refined.astype("uint8"))
'''
import cv2
import numpy as np
import joblib
from logic.gradient_map import compute_surface_gradients
from ml.feature_extractor import extract_features


def refine_dem(input_image_path, gray_dem_path, output_dir):
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    dem = cv2.imread(gray_dem_path, cv2.IMREAD_GRAYSCALE)

    if img is None or dem is None:
        raise RuntimeError("Image or DEM not found for RF refinement.")

    img = img.astype(np.float32) / 255.0
    dem = dem.astype(np.float32) / 255.0

    p, q = compute_surface_gradients(img)
    X = extract_features(img, dem, p, q)

    rf = joblib.load("ml/rf_model.pkl")
    residual = rf.predict(X).reshape(dem.shape)

    refined = dem + residual
    refined = cv2.normalize(refined, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # ---- SAVE GRAYSCALE ML DEM ----
    gray_out = f"{output_dir}/ml_gray_dem.png"
    cv2.imwrite(gray_out, refined)

    # ---- SAVE COLOR ML DEM ----
    color_out = f"{output_dir}/ml_color_dem.png"
    color_map = cv2.applyColorMap(refined, cv2.COLORMAP_JET)
    cv2.imwrite(color_out, color_map)

    return gray_out, color_out
