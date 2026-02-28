'''import numpy as np
import cv2

def simulate_sunlight_effect(image, sun_azimuth_deg=45, sun_elevation_deg=45):
    # Convert image to float32
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    image /= 255.0

    # Normalize image contrast
    image = cv2.equalizeHist((image * 255).astype(np.uint8)).astype(np.float32) / 255.0

    # Compute light direction vector (sun)
    azimuth_rad = np.deg2rad(sun_azimuth_deg)
    elevation_rad = np.deg2rad(sun_elevation_deg)

    lx = np.cos(elevation_rad) * np.cos(azimuth_rad)
    ly = np.cos(elevation_rad) * np.sin(azimuth_rad)
    lz = np.sin(elevation_rad)
    light_vector = np.array([lx, ly, lz])

    # Compute surface gradients using Sobel
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)

    # Create surface normals
    nz = np.ones_like(image)
    normal_vectors = np.stack((-gx, -gy, nz), axis=-1)
    norm = np.linalg.norm(normal_vectors, axis=-1, keepdims=True)
    unit_normals = normal_vectors / (norm + 1e-6)

    # Calculate reflectance using Lambertian model
    dot_product = np.dot(unit_normals.reshape(-1, 3), light_vector)
    reflectance = dot_product.reshape(image.shape)
    reflectance = np.clip(reflectance, 0, 1)

    # Scale back to 0–255 image
    shaded_image = (reflectance * 255).astype(np.uint8)
    return shaded_image'''

import numpy as np
import cv2

def simulate_sunlight_effect(image, sun_azimuth_deg=45, sun_elevation_deg=45):
    # ---- Ensure grayscale safely ----
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32) / 255.0

    # ---- Normalize contrast ----
    gray_eq = cv2.equalizeHist((gray * 255).astype(np.uint8))
    gray_eq = gray_eq.astype(np.float32) / 255.0

    # ---- Sun direction ----
    azimuth_rad = np.deg2rad(sun_azimuth_deg)
    elevation_rad = np.deg2rad(sun_elevation_deg)

    lx = np.cos(elevation_rad) * np.cos(azimuth_rad)
    ly = np.cos(elevation_rad) * np.sin(azimuth_rad)
    lz = np.sin(elevation_rad)
    light_vector = np.array([lx, ly, lz])

    # ---- Surface gradients ----
    gx = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1, ksize=5)

    # ---- Surface normals ----
    nz = np.ones_like(gray_eq)
    normal_vectors = np.stack((-gx, -gy, nz), axis=-1)
    norm = np.linalg.norm(normal_vectors, axis=-1, keepdims=True)
    unit_normals = normal_vectors / (norm + 1e-6)

    # ---- Lambertian reflectance ----
    dot_product = np.dot(unit_normals.reshape(-1, 3), light_vector)
    reflectance = dot_product.reshape(gray_eq.shape)
    reflectance = np.clip(reflectance, 0, 1)

    shaded_image = (reflectance * 255).astype(np.uint8)
    return shaded_image
