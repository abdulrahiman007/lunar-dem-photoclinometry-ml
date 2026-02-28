import numpy as np
import matplotlib.pyplot as plt

def analyze_path(dem_image, path, max_elevation_meters=8000):
    path_stats = {
        'total_distance_pixels': 0,
        'elevation_gain_m': 0,
        'elevation_loss_m': 0,
        'path_segments': [] # To store slope info for color-coding
    }
    
    elevations_m = []
    distances_pixels = [0.0]

    # Convert pixel values to meters for the entire path
    for x, y in path:
        pixel_value = dem_image[y, x]
        elevations_m.append((pixel_value / 255.0) * max_elevation_meters)

    # Analyze each segment of the path
    for i in range(len(path) - 1):
        p1 = path[i]
        p2 = path[i+1]
        
        # Calculate distance for the segment
        dist = np.linalg.norm(np.array(p1) - np.array(p2))
        path_stats['total_distance_pixels'] += dist
        distances_pixels.append(path_stats['total_distance_pixels'])
        
        # Calculate elevation change for the segment
        elev_change = elevations_m[i+1] - elevations_m[i]
        if elev_change > 0:
            path_stats['elevation_gain_m'] += elev_change
        else:
            path_stats['elevation_loss_m'] += abs(elev_change)
            
        # Calculate slope (rise/run). Avoid division by zero.
        slope = abs(elev_change / dist) if dist > 0 else 0
        path_stats['path_segments'].append({'p1': p1, 'p2': p2, 'slope': slope})
        
    return path_stats, distances_pixels, elevations_m

def plot_elevation_profile(distances, elevations, output_path):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(distances, elevations, color='#f26b38', linewidth=2)
    ax.fill_between(distances, elevations, color='#f26b38', alpha=0.2)
    
    ax.set_title('Rover Path Elevation Profile', fontsize=16, color='white')
    ax.set_xlabel('Distance Along Path (in pixels)', fontsize=12, color='white')
    ax.set_ylabel('Elevation (meters)', fontsize=12, color='white')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Customize ticks and spines for better appearance
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
        
    plt.tight_layout()
    plt.savefig(output_path, transparent=True)
    plt.close()