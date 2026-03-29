import numpy as np
import torch
import cv2
from typing import Dict


# ============================================
# Utility Functions (extracted from class)
# ============================================

def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Filter points by taking only points within workspace bounds.
    
    Args:
        points: (N, 3) array of 3D points
        bounds_min: (3,) array of minimum bounds
        bounds_max: (3,) array of maximum bounds
        strict: If False, expand bounds slightly
    
    Returns:
        Boolean mask of shape (N,) indicating which points are within bounds
    """
    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask


def extract_sensor_data(obs: Dict, camera_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Extract RGB, depth, segmentation, and camera parameters from observation.
    
    Args:
        obs: Observation dictionary
        camera_name: Name of the camera to use
        
    Returns:
        rgb: RGB image (H, W, 3), uint8
        depth: Depth image (H, W), float32 in meters
        segmentation: Segmentation mask (H, W), int16
        camera_params: Camera parameters dictionary
    """
    sensor_data = obs['sensor_data'][camera_name]
    camera_params = obs['sensor_param'][camera_name]
    
    # RGB image
    rgb = sensor_data['rgb']
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    # Remove batch dimension if present
    while len(rgb.shape) > 3:
        rgb = rgb.squeeze(0)
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    
    # Depth image
    depth = sensor_data['depth']
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    # Remove all singleton dimensions
    while len(depth.shape) > 2:
        depth = depth.squeeze()
    # Convert to meters
    depth_meters = depth.astype(np.float32) / 1000.0
    
    # Segmentation mask
    segmentation = sensor_data['segmentation']
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()
    
    # Force segmentation to be 2D
    if len(segmentation.shape) == 3:
        if segmentation.shape[2] == 1:
            segmentation = segmentation[:, :, 0]
        elif segmentation.shape[0] == 1:
            segmentation = segmentation[0, :, :]
        else:
            segmentation = segmentation[:, :, 0]
    elif len(segmentation.shape) != 2:
        segmentation = segmentation.squeeze()
        if len(segmentation.shape) == 3:
            segmentation = segmentation[:, :, 0] if segmentation.shape[2] == 1 else segmentation[:, :, 0]
    
    return rgb, depth_meters, segmentation, camera_params


def depth_to_pointcloud(depth: np.ndarray, camera_params: Dict) -> np.ndarray:
    """
    Convert depth image to 3D point cloud in world coordinates.
    
    Args:
        depth: Depth image (H, W) in meters
        camera_params: Camera parameters dictionary
        
    Returns:
        points: Point cloud (H, W, 3) in world coordinates
    """
    H, W = depth.shape
    
    # Get camera intrinsics
    intrinsic = camera_params['intrinsic_cv']
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.cpu().numpy()
    intrinsic = intrinsic.squeeze(0)
    
    # Get camera extrinsics (cam2world)
    cam2world = camera_params['cam2world_gl']
    if isinstance(cam2world, torch.Tensor):
        cam2world = cam2world.cpu().numpy()
    cam2world = cam2world.squeeze(0)
    
    # Handle different intrinsic formats
    if intrinsic.shape == (3, 3):
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    elif intrinsic.shape == (4, 4):
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    else:
        raise ValueError(f"Unexpected intrinsic shape: {intrinsic.shape}")
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Convert to camera space (x, y, z)
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack to get points in camera space
    points_cam = np.stack([x, y, z], axis=-1)  # [H, W, 3]
    
    # Filter out invalid points (depth == 0 or too far)
    valid_mask = (z > 0) & (z < 10.0)  # Adjust max distance as needed
    
    # Reshape to [H*W, 3]
    points_cam_flat = points_cam.reshape(-1, 3)
    valid_mask_flat = valid_mask.reshape(-1)
    
    # Convert to homogeneous coordinates
    points_cam_homo = np.concatenate([points_cam_flat, np.ones((points_cam_flat.shape[0], 1))], axis=-1)
    
    # Transform to world space
    if cam2world.shape == (4, 4):
        points_world_homo = (points_cam_homo @ cam2world.T)[:, :3]
    else:
        raise ValueError(f"Unexpected cam2world shape: {cam2world.shape}")
    
    # Create full point cloud array (H, W, 3) for keypoint detection
    points_full = np.zeros((H, W, 3), dtype=np.float32)
    points_world_2d = points_world_homo.reshape(H, W, 3)
    points_full[valid_mask] = points_world_2d[valid_mask]
    
    return points_full


def preprocess_for_dinov2(rgb, patch_size=14):
    """
    Preprocess RGB image to be compatible with DINOv2.
    
    Args:
        rgb: RGB image (H, W, 3)
        patch_size: Patch size of DINOv2 (default: 14)
    
    Returns:
        transformed_rgb: Resized and normalized RGB (new_H, new_W, 3)
        shape_info: Dictionary with shape information
    """
    H, W, _ = rgb.shape
    patch_h = int(H // patch_size)
    patch_w = int(W // patch_size)
    new_H = patch_h * patch_size
    new_W = patch_w * patch_size
    transformed_rgb = cv2.resize(rgb, (new_W, new_H))
    transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]
    
    shape_info = {
        'img_h': H,
        'img_w': W,
        'patch_h': patch_h,
        'patch_w': patch_w,
    }
    return transformed_rgb, shape_info


def preprocess_masks(masks):
    """
    Convert masks to binary masks and filter out background.
    
    Args:
        masks: Segmentation mask array (H, W)
    
    Returns:
        Tuple of (binary_masks, original_ids)
    """
    # Filter out background (ID 0) - we don't want keypoints on background
    unique_mask_ids = np.unique(masks)
    unique_mask_ids = unique_mask_ids[unique_mask_ids != 0]  # Remove background ID 0
    # Store both binary masks and their original IDs
    masks_binary = [masks == uid for uid in unique_mask_ids]
    return (masks_binary, unique_mask_ids)


def project_keypoints_to_img(rgb, candidate_pixels, candidate_rigid_group_ids, scale=1.0):
    """
    Project keypoints onto RGB image with labels.
    
    Args:
        rgb: RGB image (H, W, 3)
        candidate_pixels: Pixel coordinates (N, 2)
        candidate_rigid_group_ids: Mask IDs for each keypoint (N,)
        scale: Scale factor for label size (default 1.0)
    
    Returns:
        projected: RGB image with keypoints drawn on it
    """
    # Ensure contiguous uint8 array for OpenCV
    projected = np.ascontiguousarray(rgb, dtype=np.uint8)
    
    # Parameters (tuned for clear visibility)
    font_scale = 0.5 * scale
    box_width_base = int(18 * scale)
    box_width_per_char = int(8 * scale)
    box_height = int(18 * scale)
    border_thickness = max(1, int(1 * scale))
    font_thickness = max(1, int(2 * scale))
    
    for keypoint_idx, pixel in enumerate(candidate_pixels):
        displayed_text = str(keypoint_idx)
        text_length = len(displayed_text)
        
        # Calculate box size
        box_width = box_width_base + box_width_per_char * (text_length - 1)
        
        x, y = int(pixel[1]), int(pixel[0])
        
        # White filled rectangle
        cv2.rectangle(projected, 
                     (x - box_width // 2, y - box_height // 2), 
                     (x + box_width // 2, y + box_height // 2), 
                     (255, 255, 255), -1)
        # Black border
        cv2.rectangle(projected, 
                     (x - box_width // 2, y - box_height // 2), 
                     (x + box_width // 2, y + box_height // 2), 
                     (0, 0, 0), border_thickness)
        
        # Get text size for proper centering
        (text_w, text_h), baseline = cv2.getTextSize(displayed_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                      font_scale, font_thickness)
        # Center the text
        text_x = x - text_w // 2
        text_y = y + text_h // 2
        
        cv2.putText(projected, displayed_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 0, 0), font_thickness)
    
    return projected
