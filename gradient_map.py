import cv2
import numpy as np

def compute_surface_gradients(shaded_image):
    # Ensure image is grayscale float32
    if len(shaded_image.shape) == 3:
        shaded_image = cv2.cvtColor(shaded_image, cv2.COLOR_BGR2GRAY)
    shaded_image = shaded_image.astype(np.float32) / 255.0

    # Compute horizontal (p = dz/dx) and vertical (q = dz/dy) gradients
    p = cv2.Sobel(shaded_image, cv2.CV_32F, 1, 0, ksize=5)  # ∂z/∂x
    q = cv2.Sobel(shaded_image, cv2.CV_32F, 0, 1, ksize=5)  # ∂z/∂y

    return p, q