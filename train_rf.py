'''import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

from ml.feature_extractor import extract_features
from logic.gradient_map import compute_surface_gradients

IMG = "static/input/input.png"
DEM = "static/output/color_dem.png"

img = cv2.imread(IMG, 0) / 255.0
dem = cv2.imread(DEM, 0) / 255.0

gx, gy = compute_surface_gradients(img)

# pseudo ground truth
smooth = cv2.GaussianBlur(dem, (9, 9), 0)
residual = smooth - dem

X = extract_features(img, dem, gx, gy)
y = residual.flatten()

idx = np.random.choice(len(y), min(40000, len(y)), replace=False)

rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=22,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

rf.fit(X[idx], y[idx])
joblib.dump(rf, "ml/rf_model.pkl")

print("RF trained safely on residuals.")
'''

'''import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

from logic.gradient_map import compute_surface_gradients
from ml.feature_extractor import extract_features

INPUT_DIR = "static/input"
GRAY_DEM_PATH = "static/output/gray_dem.png"
MODEL_PATH = "ml/rf_model.pkl"


# ---- find latest uploaded image ----
files = os.listdir(INPUT_DIR)
if not files:
    raise RuntimeError("No input image found. Upload image first.")

img_path = os.path.join(INPUT_DIR, files[-1])

if not os.path.exists(GRAY_DEM_PATH):
    raise RuntimeError("gray_dem.png not found. Run SFS first.")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
dem = cv2.imread(GRAY_DEM_PATH, cv2.IMREAD_GRAYSCALE)

if img is None or dem is None:
    raise RuntimeError("Could not load image or DEM.")

img = img.astype("float32") / 255.0
dem = dem.astype("float32") / 255.0

p, q = compute_surface_gradients(img)

# ---- residual learning target ----
smooth = cv2.GaussianBlur(dem, (9, 9), 0)
residual = smooth - dem

X = extract_features(img, dem, p, q)
y = residual.flatten()

idx = np.random.choice(len(y), min(40000, len(y)), replace=False)

rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=22,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

rf.fit(X[idx], y[idx])
joblib.dump(rf, MODEL_PATH)

print("Random Forest trained successfully.")
'''
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

from logic.gradient_map import compute_surface_gradients
from ml.feature_extractor import extract_features


# ---------------- PATHS ----------------
INPUT_DIR = "static/input"
GRAY_DEM_PATH = "static/output/gray_dem.png"
MODEL_PATH = "ml/rf_model.pkl"


# ---------------- LOAD INPUT IMAGE ----------------
files = os.listdir(INPUT_DIR)
if not files:
    raise RuntimeError("No input image found. Upload an image first.")

# take latest uploaded image
img_path = os.path.join(INPUT_DIR, sorted(files)[-1])

if not os.path.exists(GRAY_DEM_PATH):
    raise RuntimeError("gray_dem.png not found. Run SFS first.")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
dem = cv2.imread(GRAY_DEM_PATH, cv2.IMREAD_GRAYSCALE)

if img is None or dem is None:
    raise RuntimeError("Could not load input image or DEM.")

# normalize
img = img.astype(np.float32) / 255.0
dem = dem.astype(np.float32) / 255.0


# ---------------- ALIGN SHAPES (CRITICAL FIX) ----------------
h, w = dem.shape
img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

# recompute gradients AFTER resize
p, q = compute_surface_gradients(img)


# ---------------- RESIDUAL LEARNING TARGET ----------------
smooth_dem = cv2.GaussianBlur(dem, (9, 9), 0)
residual = smooth_dem - dem


# ---------------- FEATURE EXTRACTION ----------------
X = extract_features(img, dem, p, q)
y = residual.reshape(-1)

# sample pixels to reduce training time
num_samples = min(40000, len(y))
idx = np.random.choice(len(y), num_samples, replace=False)


# ---------------- TRAIN RANDOM FOREST ----------------
rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=22,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42
)

rf.fit(X[idx], y[idx])

joblib.dump(rf, MODEL_PATH)
print("✅ Random Forest trained successfully.")
