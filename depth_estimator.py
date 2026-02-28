import numpy as np
import cv2

def integrate_gradients(p, q):
    h, w = p.shape
    Z = np.zeros((h, w), dtype=np.float32)

    # Integrate row-wise along x-direction (p)
    for y in range(1, h):
        Z[y, 0] = Z[y-1, 0] + q[y-1, 0]

    for y in range(h):
        for x in range(1, w):
            Z[y, x] = Z[y, x-1] + p[y, x-1]

    return Z

def normalize_dem(Z):
    Z_norm = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX)
    return Z_norm.astype(np.uint8)