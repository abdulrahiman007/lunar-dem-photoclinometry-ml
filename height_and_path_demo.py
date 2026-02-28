import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

# Path to your DEM output image
dem_path = os.path.join('static', 'output', 'result.png')

# Load DEM image
img = mpimg.imread(dem_path)

# Convert to grayscale if necessary
if img.ndim == 3:
    img_gray = np.mean(img, axis=2)
else:
    img_gray = img

# Normalize pixel values (0–1 range)
img_gray = img_gray / np.max(img_gray)

# Assume a max elevation (scaling factor)
max_elevation = 1000  # meters
elevation_map = img_gray * max_elevation

# Example points to measure height difference
point_A = (50, 100)
point_B = (150, 200)
height_A = elevation_map[point_A]
height_B = elevation_map[point_B]
height_diff = abs(height_A - height_B)

print(f"Height at point A {point_A}: {height_A:.2f} m")
print(f"Height at point B {point_B}: {height_B:.2f} m")
print(f"Relative height difference: {height_diff:.2f} m")

# Example rover path coordinates (row, column)
path_points = [
    (80, 60),
    (100, 100),
    (130, 150),
    (160, 200),
    (190, 250)
]

# Split into x and y for plotting
y_points = [p[0] for p in path_points]
x_points = [p[1] for p in path_points]

# Plot DEM with terrain colormap
plt.figure(figsize=(8, 6))
plt.imshow(elevation_map, cmap='terrain')
plt.plot(x_points, y_points, color='red', linewidth=2, label='Rover Path')
plt.scatter(x_points, y_points, color='blue', s=30)
plt.text(x_points[0], y_points[0], 'Start', color='white', fontsize=9, ha='right')
plt.text(x_points[-1], y_points[-1], 'End', color='white', fontsize=9, ha='left')

plt.title('Lunar DEM with Rover Path Overlay')
plt.colorbar(label='Elevation (meters)')
plt.legend()

# Save the visualization
output_path = os.path.join('static', 'output', 'path_visualization.png')
plt.savefig(output_path)

plt.show()
print(f"Visualization saved to {output_path}")
