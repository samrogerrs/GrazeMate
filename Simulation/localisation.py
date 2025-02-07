import numpy as np
import matplotlib.pyplot as plt
import noise

# =========================
# TERRAIN GENERATION
# =========================
def generate_topography(grid_size, scale, seed):
    """
    Generate a simple terrain using Perlin noise.
    """
    np.random.seed(seed)
    topo = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            topo[i, j] = noise.pnoise2(
                i / scale, j / scale,
                octaves=6, persistence=0.5, lacunarity=2.0,
                repeatx=grid_size, repeaty=grid_size, base=seed
            ) * 400
    return topo

# =========================
# ROTATION MATRICES
# =========================
def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])

def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

# =========================
# TERRAIN INTERPOLATION
# =========================
def get_terrain_height(x, y, topo, x_vals, y_vals):
    """
    Interpolate terrain height at any (x,y) point using bilinear interpolation.
    """
    # Find the grid cell containing (x,y)
    i = np.searchsorted(x_vals, x) - 1
    j = np.searchsorted(y_vals, y) - 1
    
    # Clamp indices to valid range
    i = max(0, min(i, len(x_vals)-2))
    j = max(0, min(j, len(y_vals)-2))
    
    # Get the four corner points
    x0, x1 = x_vals[i], x_vals[i+1]
    y0, y1 = y_vals[j], y_vals[j+1]
    
    # Get heights at corners
    h00 = topo[j, i]
    h10 = topo[j, i+1]
    h01 = topo[j+1, i]
    h11 = topo[j+1, i+1]
    
    # Compute interpolation weights
    wx = (x - x0) / (x1 - x0)
    wy = (y - y0) / (y1 - y0)
    
    # Bilinear interpolation
    return (1-wx)*(1-wy)*h00 + wx*(1-wy)*h10 + (1-wx)*wy*h01 + wx*wy*h11

# =========================
# PIXEL-TO-TERRAIN PROJECTION USING RAY INTERSECTION
# =========================
def pixel_to_terrain(u, v, image_width, image_height, fov, altitude, pitch, roll, yaw,
                    topo, x_vals, y_vals, max_iterations=100, tolerance=0.1):
    """
    Project a pixel onto the terrain surface using iterative ray-terrain intersection.
    """
    # Compute the focal length in pixels (assuming a square sensor)
    f_rad = np.radians(fov)
    f_px = image_width / (2 * np.tan(f_rad / 2))
    cx = image_width / 2
    cy = image_height / 2

    # Normalize pixel coordinates
    x = (u - cx) / f_px
    y = (v - cy) / f_px
    d_cam = np.array([x, y, 1.0])
    
    # Build the rotation from camera to world
    R0 = np.diag([1, -1, -1])
    R_roll = rotation_matrix_z(np.radians(roll))
    R_pitch = rotation_matrix_x(np.radians(pitch))
    R_yaw = rotation_matrix_z(np.radians(yaw))
    R = R_yaw @ R_pitch @ R_roll @ R0

    # Rotate the ray into world coordinates
    d_world = R @ d_cam
    d_world = d_world / np.linalg.norm(d_world)  # Normalize direction
    
    # Drone position
    drone_pos = np.array([400, 400, altitude])
    
    # Ray-terrain intersection using iterative search
    t = altitude  # Initial guess based on average terrain height
    for _ in range(max_iterations):
        # Current point along ray
        point = drone_pos + t * d_world
        
        # Get terrain height at this (x,y)
        terrain_z = get_terrain_height(point[0], point[1], topo, x_vals, y_vals)
        
        # Check if we're close enough
        if abs(point[2] - terrain_z) < tolerance:
            return point[0], point[1], terrain_z
        
        # Update t based on height difference
        t = t - (point[2] - terrain_z) / d_world[2]
    
    return None

# =========================
# PLOTTING FUNCTION
# =========================
def plot_projection(image_width, image_height, fov, altitude, pitch, roll, yaw,
                   topo, grid_size, scale, detected_pixels=None):
    """
    Plot the drone, terrain, and projected points.
    
    Parameters:
        detected_pixels: List of tuples [(u1, v1), (u2, v2), ...] containing pixel coordinates
                       or None if no pixels to detect
    """
    drone_pos = np.array([400, 400, altitude])
    
    x_vals = np.linspace(-scale * grid_size / 2, scale * grid_size / 2, grid_size)
    y_vals = np.linspace(-scale * grid_size / 2, scale * grid_size / 2, grid_size)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Sample points along image perimeter
    num_samples = 50
    perimeter = []
    
    for edge in [(np.linspace(0, image_width, num_samples), np.zeros(num_samples)),
                 (np.full(num_samples, image_width), np.linspace(0, image_height, num_samples)),
                 (np.linspace(image_width, 0, num_samples), np.full(num_samples, image_height)),
                 (np.zeros(num_samples), np.linspace(image_height, 0, num_samples))]:
        for u, v in zip(edge[0], edge[1]):
            pt = pixel_to_terrain(u, v, image_width, image_height, fov, altitude,
                                pitch, roll, yaw, topo, x_vals, y_vals)
            if pt is not None:
                perimeter.append(pt)
    
    # Project image center
    center = pixel_to_terrain(image_width/2, image_height/2, image_width, image_height,
                            fov, altitude, pitch, roll, yaw, topo, x_vals, y_vals)
    
    # Project all detected pixels
    detected_points = []
    if detected_pixels:
        for u_det, v_det in detected_pixels:
            point = pixel_to_terrain(u_det, v_det, image_width, image_height,
                                   fov, altitude, pitch, roll, yaw, topo, x_vals, y_vals)
            if point is not None:
                detected_points.append(point)
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot terrain surface
    surf = ax.plot_surface(X, Y, topo, cmap='terrain', alpha=0.6)
    
    # Plot drone position
    ax.scatter(drone_pos[0], drone_pos[1], drone_pos[2],
              color='blue', s=100, label='Drone')
    
    # Plot perimeter points
    if perimeter:
        perim_x = [p[0] for p in perimeter]
        perim_y = [p[1] for p in perimeter]
        perim_z = [p[2] for p in perimeter]
        ax.scatter(perim_x, perim_y, perim_z,
                  color='orange', s=10, label='Image Perimeter')
    
    # Plot center point
    if center is not None:
        ax.scatter(center[0], center[1], center[2],
                  color='green', s=100, label='Center Pixel')
    
    # Plot detected points with different colors
    if detected_points:
        for i, point in enumerate(detected_points):
            ax.scatter(point[0], point[1], point[2]+5,
                      color="red", s=150, marker='*',
                      label=f'Cow {i+1}')
    
    # Connect perimeter points
    if perimeter:
        perim_x.append(perim_x[0])
        perim_y.append(perim_y[0])
        perim_z.append(perim_z[0])
        ax.plot(perim_x, perim_y, perim_z, 'k-', linewidth=2, label='Image Boundary')
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Projection of Drone Image onto Terrain Surface")
    ax.legend()
    plt.show()

# =========================
# PARAMETERS & RUN
# =========================
image_width = 4000
image_height = 4000
fov = 82.1
altitude = 100

pitch = -40  # Tilted camera to better show terrain intersection
roll = 0
yaw = -60

grid_size = 25
scale = 50
seed = 4

# Generate terrain
topo = generate_topography(grid_size, scale, seed)

# Define multiple detected pixels
detected_pixels = [
    (3000, 3000),  # First point
    (1000, 1000),  # Second point
    (2000, 3500),  # Third point
    (3500, 1500),   # Fourth point
    (500, 2500),    # Fifth point
    (3000, 1000),   # Sixth point
    (4000, 3000),   # Seventh point
    (1500, 3800)    # Eighth point
]

# Run the projection
plot_projection(image_width, image_height, fov, altitude, pitch, roll, yaw,
               topo, grid_size, scale, detected_pixels=detected_pixels)