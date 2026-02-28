'''import cv2
import numpy as np
from logic.sun_model import simulate_sunlight_effect
from logic.gradient_map import compute_surface_gradients
from logic.depth_estimator import integrate_gradients, normalize_dem

class DEMProcessor:
    def run(self, input_path: str, output_dir: str) -> None:
        image = cv2.imread(input_path)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        # Step 1: Simulate sun effect
        shaded_image = simulate_sunlight_effect(image)

        # Step 2: Compute surface gradients
        p, q = compute_surface_gradients(shaded_image)

        # Step 3: Integrate to get elevation map
        Z = integrate_gradients(p, q)

        # Step 4: Apply bilateral filter to reduce noise and preserve edges
        Z_filtered = cv2.bilateralFilter(Z.astype(np.float32), d=9, sigmaColor=75, sigmaSpace=75)

        # Step 5: Normalize and save grayscale DEM
        Z_gray = normalize_dem(Z_filtered)
        cv2.imwrite(f"{output_dir}/gray_dem.png", Z_gray)

        # Step 6: Apply colormap and save colored DEM
        Z_color = cv2.applyColorMap(Z_gray, cv2.COLORMAP_JET)
        cv2.imwrite(f"{output_dir}/color_dem.png", Z_color)


'''
'''import cv2
import numpy as np
from logic.sun_model import simulate_sunlight_effect
from logic.gradient_map import compute_surface_gradients
from logic.depth_estimator import integrate_gradients, normalize_dem

class DEMProcessor:
    def run(self, input_path: str, output_dir: str):
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        shaded_image = simulate_sunlight_effect(image)
        p, q = compute_surface_gradients(shaded_image)
        Z = integrate_gradients(p, q)

        Z_filtered = cv2.bilateralFilter(
            Z.astype(np.float32),
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )

        Z_gray = normalize_dem(Z_filtered)

        gray_path = f"{output_dir}/gray_dem.png"
        color_path = f"{output_dir}/color_dem.png"

        cv2.imwrite(gray_path, Z_gray)
        cv2.imwrite(color_path, cv2.applyColorMap(Z_gray, cv2.COLORMAP_JET))

        return gray_path, color_path
'''
import cv2
import numpy as np
from logic.sun_model import simulate_sunlight_effect
from logic.gradient_map import compute_surface_gradients
from logic.depth_estimator import integrate_gradients, normalize_dem


class DEMProcessor:
    def run(self, input_path: str, output_dir: str):
        # ---- Read image safely ----
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {input_path}")

        # ---- Step 1: Simulate sun illumination ----
        shaded_image = simulate_sunlight_effect(image)

        # ---- Step 2: Compute surface gradients ----
        p, q = compute_surface_gradients(shaded_image)

        # ---- Step 3: Integrate gradients to get elevation ----
        Z = integrate_gradients(p, q)

        # ---- Step 4: Edge-preserving smoothing ----
        Z_filtered = cv2.bilateralFilter(
            Z.astype(np.float32),
            d=9,
            sigmaColor=75,
            sigmaSpace=75
        )

        # ---- Step 5: Normalize DEM ----
        Z_gray = normalize_dem(Z_filtered)

        gray_path = f"{output_dir}/gray_dem.png"
        color_path = f"{output_dir}/color_dem.png"

        cv2.imwrite(gray_path, Z_gray)

        # ---- Step 6: Color visualization ----
        Z_color = cv2.applyColorMap(Z_gray, cv2.COLORMAP_JET)
        cv2.imwrite(color_path, Z_color)

        # ✅ RETURN paths for Flask + RF
        return gray_path, color_path
