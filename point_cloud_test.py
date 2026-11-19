import numpy as np
import open3d as o3d
from PIL import Image

def generate_point_cloud(rgb_path: str, depth_path: str, width: int = 1280, height: int = 960, fov_deg: float = 55.0):
    """
    Loads an RGB image and depth map, unprojects valid pixels into 3D space,
    and returns an Open3D PointCloud object.
    """
    # 1. Load Color and Depth Data
    print("Loading images...")
    rgb_img = Image.open(rgb_path).convert("RGB")
    rgb_np = np.asarray(rgb_img, dtype=np.float32) / 255.0  # Normalize color to [0, 1]
    
    depth_np = np.load(depth_path)  # Expected shape (H, W), dtype float32 (in meters)
    
    # 2. Compute Camera Intrinsics from FOV and Resolution
    # Assuming horizontal FOV is 55 degrees
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * np.tan(fov_rad / 2.0))
    fy = fx  # Square pixels assumption
    cx = width / 2.0
    cy = height / 2.0
    
    print(f"Calculated Intrinsics -> fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")

    # 3. Create Pixel Coordinate Grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Flatten grids and arrays for vectorized operations
    u = u.flatten()
    v = v.flatten()
    z = depth_np.flatten()
    colors = rgb_np.reshape(-1, 3)

    # 4. Filter Invalid Depth Pixels (e.g., zeros, background, or out-of-range)
    # In your sim, valid depth is only ~3.5% of pixels (cables only, range ~0.75m to 0.87m)
    valid_mask = (z > 0.1) & (z < 2.0)  # Adjust bounds based on your camera range
    
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = z[valid_mask]
    colors_valid = colors[valid_mask]

    # 5. Unproject 2D Pixels to 3D Camera Coordinates
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = Z
    x_valid = (u_valid - cx) * z_valid / fx
    y_valid = (v_valid - cy) * z_valid / fy
    
    # Stack into an (N, 3) point array
    points_camera = np.stack((x_valid, y_valid, z_valid), axis=-1)

    # 6. Optional: Transform from Camera Frame to World Frame
    # If your camera is fixed at (0, 0, 0.8) looking straight down (-Z camera into +Z world, or similar),
    # apply your T_world_cam extrinsics matrix here. 
    # For a simple downward-looking camera where Z is depth and you want standard world coords:
    # (Example transformation: rotate/translate depending on your simulator convention)
    
    # 7. Build Open3D Point Cloud Object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_camera)
    pcd.colors = o3d.utility.Vector3dVector(colors_valid)

    print(f"Successfully generated point cloud with {len(points_camera)} valid points.")
    return pcd

if __name__ == "__main__":
    # Example usage (update paths to match your output directory)
    rgb_file = "settled_color.png"
    depth_file = "settled_depth.npy"
    
    try:
        point_cloud = generate_point_cloud(rgb_file, depth_file)
        
        # Visualize the point cloud interactively
        o3d.visualization.draw_geometries([point_cloud], window_name="Cable Pile Point Cloud", width=1024, height=768)
        
        # To save to disk:
        # o3d.io.write_point_cloud("cable_pile.pcd", point_cloud)
        
    except FileNotFoundError:
        print(f"Could not find {rgb_file} or {depth_file}. Run your simulation capture script first!")