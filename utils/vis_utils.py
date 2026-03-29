"""
Draw predicted trajectories on images.

Uses matplotlib to draw 2D projections of 3D trajectories on camera images.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio
import os
import sys
import cv2
from typing import List, Dict, Optional, Union, Tuple

from core.env_adapters import BaseEnvAdapter

from utils.logging_utils import SteerLogger

log = SteerLogger("VisUtils")


def add_text_to_image(
    image: np.ndarray,
    text_lines: List[str],
    position: str = "top_left",
    font_scale: float = None,  # Auto-scale based on image size
    thickness: int = None,     # Auto-scale based on image size
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    padding: int = None        # Auto-scale based on image size
) -> np.ndarray:
    """
    Add text overlay to image.
    
    Args:
        image: Input image (H, W, 3) uint8
        text_lines: List of text strings to display
        position: "top_left", "top_right", "bottom_left", "bottom_right"
        font_scale: Font size scale (auto-scaled if None)
        thickness: Text thickness (auto-scaled if None)
        bg_color: Background color (B, G, R)
        text_color: Text color (B, G, R)
        padding: Padding around text (auto-scaled if None)
    
    Returns:
        Image with text overlay
    """
    image = image.copy()
    h, w = image.shape[:2]
    
    # Auto-scale parameters based on image size
    if font_scale is None:
        font_scale = 0.4 if w <= 256 else 0.6
    if thickness is None:
        thickness = 1 if w <= 256 else 2
    if padding is None:
        padding = 5 if w <= 256 else 10
    
    # Calculate text size for background box
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_text_width = 0
    total_text_height = 0
    line_heights = []
    
    for text in text_lines:
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        max_text_width = max(max_text_width, text_w)
        line_height = text_h + baseline + 5
        line_heights.append(line_height)
        total_text_height += line_height
    
    # Calculate background box position
    box_width = max_text_width + 2 * padding
    box_height = total_text_height + 2 * padding
    
    if position == "top_left":
        x_start, y_start = 0, 0
    elif position == "top_right":
        x_start, y_start = w - box_width, 0
    elif position == "bottom_left":
        x_start, y_start = 0, h - box_height
    elif position == "bottom_right":
        x_start, y_start = w - box_width, h - box_height
    else:
        x_start, y_start = 0, 0
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(overlay, 
                  (x_start, y_start), 
                  (x_start + box_width, y_start + box_height),
                  bg_color, 
                  -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw text lines
    y_offset = y_start + padding + line_heights[0] - 5
    for i, text in enumerate(text_lines):
        cv2.putText(image, text,
                    (x_start + padding, y_offset),
                    font, font_scale, text_color, thickness, cv2.LINE_AA)
        if i < len(line_heights) - 1:
            y_offset += line_heights[i]
    
    return image


def add_debug_chart_to_image(
    image: np.ndarray,
    history: dict,
    chart_height: int = 80,
    padding: int = 5,
) -> np.ndarray:
    """
    Add debug charts as extra rows below the image.
    
    Args:
        image: Input image (H, W, 3) uint8
        history: Dict with lists of 'grad_norm', 'factor', 'scale'
        chart_height: Height of each chart in pixels
        padding: Padding between charts
    
    Returns:
        Image with charts appended at bottom (taller than input)
    """
    h, w = image.shape[:2]
    
    # Check if we have data
    if not history or len(history.get('grad_norm', [])) == 0:
        return image
    
    num_points = len(history['grad_norm'])
    if num_points < 2:
        return image
    
    # Chart configs: (data_key, color_bgr, label, normalize)
    charts = [
        ('grad_norm', (0, 255, 255), 'Grad', False),  # Cyan
        ('factor', (0, 255, 0), 'Factor', True),       # Green
        ('scale', (255, 255, 0), 'Scale', False),      # Yellow
    ]
    
    num_charts = len(charts)
    total_chart_height = num_charts * chart_height + (num_charts + 1) * padding
    
    # Create chart area
    chart_area = np.zeros((total_chart_height, w, 3), dtype=np.uint8)
    chart_area[:] = (30, 30, 30)  # Dark gray background
    
    for chart_idx, (key, color, label, normalize) in enumerate(charts):
        data = history.get(key, [])
        if len(data) == 0:
            continue
            
        # Calculate chart position
        y_start = padding + chart_idx * (chart_height + padding)
        y_end = y_start + chart_height
        
        # Draw chart background
        cv2.rectangle(chart_area, (padding, y_start), (w - padding, y_end), (50, 50, 50), -1)
        
        # Normalize data
        data_arr = np.array(data)
        if normalize:
            data_min, data_max = 0.0, 1.0
        else:
            data_min = data_arr.min()
            data_max = data_arr.max()
            margin = (data_max - data_min) * 0.1 + 1e-6
            data_min -= margin
            data_max += margin
        
        # Map data to y coordinates
        x_coords = np.linspace(padding, w - padding, num_points).astype(int)
        y_coords = y_end - ((data_arr - data_min) / (data_max - data_min + 1e-8) * (chart_height - 10) + 5).astype(int)
        y_coords = np.clip(y_coords, y_start + 2, y_end - 2)
        
        # Draw line
        points = np.column_stack([x_coords, y_coords]).astype(np.int32)
        for i in range(len(points) - 1):
            cv2.line(chart_area, tuple(points[i]), tuple(points[i+1]), color, 2)
        
        # Draw current value marker
        cv2.circle(chart_area, tuple(points[-1]), 4, color, -1)
        
        # Draw label and current value
        current_val = data[-1]
        label_text = f"{label}: {current_val:.4f}"
        cv2.putText(chart_area, label_text, (padding + 5, y_start + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        
        # Draw min/max
        cv2.putText(chart_area, f"{data_max:.4f}", (w - 70, y_start + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(chart_area, f"{data_min:.4f}", (w - 70, y_end - 3),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    # Stack: image on top, charts below
    result = np.vstack([image, chart_area])
    
    return result


def project_3d_to_2d(
    points_3d: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    return_depth: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D world points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) array of 3D points in world frame
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: (4, 4) world-to-camera transformation matrix
        return_depth: If True, also return depth values
    
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates [u, v]
        valid_mask: (N,) boolean array indicating valid projections (in front of camera)
        (optional) depths: (N,) array of depth values if return_depth=True
    """
    points_3d = np.asarray(points_3d)
    if points_3d.ndim == 1:
        points_3d = points_3d.reshape(1, 3)
    
    N = points_3d.shape[0]
    
    # Convert to homogeneous coordinates
    points_homo = np.hstack([points_3d, np.ones((N, 1))])  # (N, 4)
    
    # Transform to camera frame
    points_cam = (extrinsics @ points_homo.T).T  # (N, 4)
    points_cam = points_cam[:, :3]  # (N, 3)
    
    # Check which points are in front of camera (positive Z in camera frame)
    # Note: Different conventions exist. OpenCV uses +Z forward, some use -Z.
    depths = points_cam[:, 2]
    
    # For OpenGL convention (camera looks at -Z), valid points have negative Z
    # For OpenCV convention (camera looks at +Z), valid points have positive Z
    # We'll handle both by checking absolute depth
    valid_mask = np.abs(depths) > 1e-6
    
    # Project to 2D
    points_2d_homo = (intrinsics @ points_cam.T).T  # (N, 3)
    
    # Normalize by depth (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        points_2d = points_2d_homo[:, :2] / np.abs(points_2d_homo[:, 2:3])
    
    # Mark invalid projections
    points_2d[~valid_mask] = np.nan
    
    if return_depth:
        return points_2d, valid_mask, depths
    
    return points_2d, valid_mask


def project_trajectory_to_image(
    trajectory_3d: np.ndarray,
    env,
    camera_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a 3D trajectory to 2D image coordinates.
    
    Convenience function that combines get_camera_intrinsics_extrinsics and project_3d_to_2d.
    
    Args:
        trajectory_3d: (N, 3) array of 3D trajectory points
        env: Environment instance
        camera_name: Camera name
    
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
        valid_mask: (N,) boolean array
    """
    intrinsics, extrinsics = get_camera_intrinsics_extrinsics(env, camera_name)
    return project_3d_to_2d(trajectory_3d, intrinsics, extrinsics)



def draw_action_trajectory_on_vlm_image(
    adapter: BaseEnvAdapter,
    action_chunk: Union[torch.Tensor, np.ndarray],
    num_steps: int = 8,
    line_width: int = 2,
    marker_size: int = 8,
    global_step: int = None,
    action_executed: int = None,
) -> np.ndarray:
    """
    Draw action trajectory on VLM camera image.
    
    Environment-agnostic function that uses adapter to get image and convert actions.
    Draws multiple trajectories with different colors.
    
    Args:
        adapter: BaseEnvAdapter instance
        action_chunk: Action chunk tensor, shape (T, action_dim) or (B, T, action_dim)
        num_steps: Number of steps to draw (default 8)
        line_width: Width of trajectory line
        marker_size: Size of start/end markers
        global_step: Current global step number (for display)
        action_executed: Number of actions executed in current chunk (for display)
    
    Returns:
        Image with trajectory drawn (numpy array, uint8, shape H,W,3)
    """
    # Colors for multiple trajectories (BGR format)
    COLORS = [
        (0, 255, 255),   # Cyan
        (255, 0, 255),   # Magenta  
        (255, 255, 0),   # Yellow
        (0, 255, 0),     # Green
        (255, 0, 0),     # Blue
        (0, 0, 255),     # Red
        (255, 128, 0),   # Orange
        (128, 0, 255),   # Purple
    ]
    
    # Get VLM image from adapter
    # For LIBERO: use raw (unflipped) image for projection, then flip result
    if hasattr(adapter, 'get_vlm_image_raw'):
        # LIBERO adapter has raw image method for projection
        image = np.array(adapter.get_vlm_image_raw())
        needs_flip = True
    else:
        # Other adapters (CALVIN) use get_vlm_image directly
        image = np.array(adapter.get_vlm_image())
        needs_flip = False
    
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    image_draw = image.copy()
    H, W = image_draw.shape[:2]
    
    # Handle action_chunk shape
    if action_chunk is None:
        return image_draw
    
    if torch.is_tensor(action_chunk):
        action_chunk = action_chunk.detach().cpu().numpy()
    
    # Ensure we have batch dimension: (B, T, action_dim)
    if action_chunk.ndim == 2:
        action_chunk = action_chunk[np.newaxis, ...]  # (1, T, action_dim)
    elif action_chunk.ndim != 3:
        print(f"Warning: Unexpected action_chunk shape: {action_chunk.shape}")
        return image_draw
    
    B, T, D = action_chunk.shape
    num_steps = min(num_steps, T)
    
    # Get camera parameters
    try:
        camera_params = adapter.get_camera_params(adapter.vlm_camera)
        intrinsics = camera_params.intrinsic
        extrinsics = camera_params.extrinsic
    except Exception as e:
        print(f"Warning: Failed to get camera params: {e}")
        return image_draw
    
    # Draw each trajectory
    num_drawn = 0
    for b in range(B):
        action_seq = action_chunk[b, :num_steps, :]
        
        try:
            # Convert to tensor for adapter
            action_tensor = torch.tensor(action_seq, dtype=torch.float32)
            
            # Convert delta actions to 3D TCP trajectory
            traj_3d = adapter.delta_actions_to_ee_trajectory(action_tensor)
            if torch.is_tensor(traj_3d):
                traj_3d = traj_3d.cpu().numpy()
            
            # Only take position (x, y, z)
            traj_3d_pos = traj_3d[:, :3]
            
            # Project 3D to 2D
            points_2d, valid_mask = project_3d_to_2d(traj_3d_pos, intrinsics, extrinsics)
            
            # Filter valid points within image bounds
            valid_in_image = (
                valid_mask &
                (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
                (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
            )
            
            valid_pts = points_2d[valid_in_image].astype(np.int32)
            
            if len(valid_pts) == 0:
                # Debug: print trajectory range if no valid points
                # print(f"Traj {b}: 3D range x=[{traj_3d_pos[:,0].min():.3f}, {traj_3d_pos[:,0].max():.3f}], "
                #       f"y=[{traj_3d_pos[:,1].min():.3f}, {traj_3d_pos[:,1].max():.3f}], "
                #       f"z=[{traj_3d_pos[:,2].min():.3f}, {traj_3d_pos[:,2].max():.3f}]")
                # print(f"  2D projection: {points_2d}")
                continue
            
            color = COLORS[b % len(COLORS)]
            
            # Draw trajectory line
            for i in range(len(valid_pts) - 1):
                cv2.line(image_draw, tuple(valid_pts[i]), tuple(valid_pts[i+1]), 
                        color, line_width, cv2.LINE_AA)
            
            # Draw start point (filled circle)
            cv2.circle(image_draw, tuple(valid_pts[0]), marker_size, color, -1)
            cv2.circle(image_draw, tuple(valid_pts[0]), marker_size, (255,255,255), 1)  # white border
            
            # Draw end point (square marker)
            if len(valid_pts) > 1:
                pt = valid_pts[-1]
                cv2.rectangle(image_draw, 
                             (pt[0]-marker_size//2, pt[1]-marker_size//2),
                             (pt[0]+marker_size//2, pt[1]+marker_size//2),
                             color, -1)
            
            num_drawn += 1
            
        except Exception as e:
            print(f"Warning: Failed to draw trajectory {b}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Flip result if we used raw image (LIBERO)
    if needs_flip:
        image_draw = np.flipud(image_draw)
    
    # Note: Text overlay is now handled by add_text_to_image in main.py
    # to avoid duplicate text rendering
    
    return image_draw


def draw_keypoints_on_image(
    adapter: BaseEnvAdapter,
    image: np.ndarray,
    keypoints: np.ndarray,
    mask_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw 3D keypoints on image by projecting them to 2D using the standard visualization format.
    Handles environment-specific image flipping (e.g., LIBERO) by adjusting coordinates.
    """
    if keypoints is None or len(keypoints) == 0:
        return image
        
    from utils.keypoint_utils import project_keypoints_to_img
    
    # Project 3D keypoints to 2D pixels [x, y] on the RAW frame coordinates
    pixels_xy = adapter.project_3d_to_2d(keypoints, adapter.vlm_camera)
    
    # If the environment uses flipped images (like LIBERO), 
    # we must adjust the coordinates to match the already-flipped 'image' 
    # instead of re-fetching and flipping the image itself.
    if hasattr(adapter, 'get_vlm_image_raw'):
        H = image.shape[0]
        # In LIBERO, get_vlm_image() is np.flipud(raw_rgb)
        # So flipped_y = H - 1 - raw_y
        pixels_xy[:, 1] = H - 1 - pixels_xy[:, 1]

    # Convert [x, y] to [y, x] for project_keypoints_to_img utility
    pixels_yx = pixels_xy[:, [1, 0]]
    
    # Use standard projection utility directly on the input image
    # This preserves any existing drawings (like trajectories) and keeps text upright
    dummy_mask_ids = mask_ids if mask_ids is not None else np.arange(len(keypoints))
    image_with_kps = project_keypoints_to_img(image, pixels_yx, dummy_mask_ids)
                        
    return image_with_kps


class TrajectoryVideoRecorder:
    """
    Class for recording trajectory visualization videos
    Generate videos for each episode and each camera
    """
    def __init__(self, output_dir: str, fps: int = 10):
        """
        Args:
            output_dir: output directory
            fps: video frame rate
        """
        self.output_dir = output_dir
        self.fps = fps
        self.frames: Dict[str, List[np.ndarray]] = {}  # camera_name -> list of frames
        # State for progressive trajectory drawing
        self.current_trajectory_2d: Optional[np.ndarray] = None
        self.current_trajectory_3d: Optional[np.ndarray] = None
        self.trajectory_color: Tuple[int, int, int] = (0, 255, 255)  # Default cyan
        self.executed_steps: int = 0
        self.total_steps: int = 0
        self.current_gripper_pos_3d: Optional[np.ndarray] = None
        os.makedirs(output_dir, exist_ok=True)

    def set_trajectory(self, adapter: BaseEnvAdapter, action_chunk: torch.Tensor, num_steps: int = 8, line_width: int = 2):
        """
        Set the trajectory to draw for the current chunk.

        Args:
            adapter: Environment adapter
            action_chunk: Action chunk tensor (T, action_dim) or (1, T, action_dim)
            num_steps: Number of steps to visualize
            line_width: Width of trajectory line
        """
        COLORS = [
            (0, 255, 255),   # Cyan
            (255, 0, 255),   # Magenta
            (255, 255, 0),   # Yellow
            (0, 255, 0),     # Green
            (255, 0, 0),     # Blue
            (0, 0, 255),     # Red
            (255, 128, 0),   # Orange
            (128, 0, 255),   # Purple
        ]

        if torch.is_tensor(action_chunk):
            action_chunk = action_chunk.detach().cpu().numpy()

        if action_chunk.ndim == 2:
            action_chunk = action_chunk[np.newaxis, ...]

        B, T, D = action_chunk.shape
        num_steps = min(num_steps, T)

        try:
            camera_params = adapter.get_camera_params(adapter.vlm_camera)
            intrinsics = camera_params.intrinsic
            extrinsics = camera_params.extrinsic
        except Exception as e:
            print(f"Warning: Failed to get camera params: {e}")
            return

        try:
            self.current_gripper_pos_3d = adapter.get_ee_position().copy()
        except:
            self.current_gripper_pos_3d = None

        action_tensor = torch.tensor(action_chunk[0, :num_steps], dtype=torch.float32)
        traj_3d = adapter.delta_actions_to_ee_trajectory(action_tensor)
        if torch.is_tensor(traj_3d):
            traj_3d = traj_3d.cpu().numpy()

        self.current_trajectory_3d = traj_3d[:, :3]

        points_2d, valid_mask = project_3d_to_2d(self.current_trajectory_3d, intrinsics, extrinsics)

        self.trajectory_2d_full = points_2d.copy()
        self.trajectory_valid_mask = valid_mask
        self.trajectory_color = COLORS[len(self.frames.get(adapter.vlm_camera, [])) % len(COLORS)]
        self.line_width = line_width

        image = np.array(adapter.get_vlm_image())
        self.H, self.W = image.shape[:2]

    def clear_trajectory(self):
        """Clear the current trajectory state."""
        self.current_trajectory_2d = None
        self.current_trajectory_3d = None
        self.current_gripper_pos_3d = None
        self.executed_steps = 0
        self.total_steps = 0

    def add_frame_with_progressive_trajectory(self, camera_name: str, image, progress: float = 1.0):
        """
        Add a frame with trajectory that progressively erases from start.

        Args:
            camera_name: camera name
            image: numpy array or torch.Tensor, shape (H, W, 3), uint8 format
            progress: Progress through current chunk (0.0 to 1.0). 1.0 means full trajectory.
        """
        if camera_name not in self.frames:
            self.frames[camera_name] = []

        if torch.is_tensor(image):
            image = image.cpu().numpy()
        if image.ndim == 4:
            image = image[0]
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        image_draw = image.copy()

        if hasattr(self, 'trajectory_2d_full') and self.trajectory_2d_full is not None:
            image_draw = self._draw_progressive_trajectory(image_draw, progress)

        self.frames[camera_name].append(image_draw)

    def _draw_progressive_trajectory(self, image: np.ndarray, progress: float = 1.0) -> np.ndarray:
        """
        Draw trajectory that progressively erases from start as execution progresses.
        """
        import cv2

        H, W = image.shape[:2]
        points_2d = self.trajectory_2d_full
        valid_mask = self.trajectory_valid_mask
        color = self.trajectory_color
        line_width = getattr(self, 'line_width', 2)

        if points_2d is None or len(points_2d) == 0:
            return image

        valid_in_image = (
            valid_mask &
            (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W) &
            (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
        )

        valid_indices = np.where(valid_in_image)[0]
        if len(valid_indices) == 0:
            return image

        num_points = len(valid_indices)
        show_count = max(0, int((1.0 - progress) * num_points))

        if show_count == 0:
            return image

        pts_to_show = points_2d[valid_indices[:show_count]].astype(np.int32)

        if len(pts_to_show) == 0:
            return image

        for i in range(len(pts_to_show) - 1):
            cv2.line(image, tuple(pts_to_show[i]), tuple(pts_to_show[i + 1]),
                    color, line_width, cv2.LINE_AA)

        cv2.circle(image, tuple(pts_to_show[0]), 8, color, -1)
        cv2.circle(image, tuple(pts_to_show[0]), 8, (255, 255, 255), 1)

        if len(pts_to_show) > 1:
            pt = pts_to_show[-1]
            cv2.rectangle(image,
                         (pt[0]-6, pt[1]-6),
                         (pt[0]+6, pt[1]+6),
                         color, -1)

        return image

    def add_frame(self, camera_name: str, image):
        """
        Add a frame image
        
        Args:
            camera_name: camera name
            image: numpy array or torch.Tensor, shape (H, W, 3) or (1, H, W, 3), uint8 format
        """
        if camera_name not in self.frames:
            self.frames[camera_name] = []
        
        # Convert tensor to numpy if needed
        if torch.is_tensor(image):
            image = image.cpu().numpy()
        
        # Squeeze batch dimension if present
        if image.ndim == 4:
            image = image[0]
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        self.frames[camera_name].append(image.copy())
    
    def save_video(self, episode_id: int = None, camera_name: str = None, save_path: str = None, success: bool = None,  behavior_name: str = None):
        """
        Save video
        
        Args:
            episode_id: episode ID (deprecated, use save_path instead)
            camera_name: if specified, only save the video for the specified camera; if None, save all cameras
            save_path: custom save path (without extension and camera name), if None, use default path
            success: whether the task is successfully completed, will add text label on the video
            behavior_name: behavior name, will add text label on the video
        """
        cameras_to_save = [camera_name] if camera_name is not None else list(self.frames.keys())
        
        for cam_name in cameras_to_save:
            if cam_name not in self.frames or len(self.frames[cam_name]) == 0:
                continue
            
            # decide the video path
            if save_path is not None:
                video_path = f"{save_path}_{cam_name}.mp4"
            else:
                video_path = os.path.join(
                    self.output_dir,
                    f"episode_{episode_id}_{cam_name}_trajectories.mp4"
                )
            
            # Use the LARGEST frame size (so charts are not cut off)
            frames = self.frames[cam_name]
            if len(frames) > 0:
                # Find max dimensions
                max_h = max(f.shape[0] for f in frames)
                max_w = max(f.shape[1] for f in frames)
                target_shape = (max_h, max_w, 3)
                normalized_frames = []
                
                for i, frame in enumerate(frames):
                    if frame.shape != target_shape:
                        # Pad smaller frames with black at bottom/right
                        padded = np.zeros(target_shape, dtype=np.uint8)
                        padded[:frame.shape[0], :frame.shape[1], :] = frame
                        frame = padded
                    
                    # add success/fail label on frame if success is provided
                    if success is not None:
                        frame = frame.copy()  # Don't modify original
                        label = "SUCCESS" if success else "FAIL"
                        color = (0, 255, 0) if success else (255, 0, 0)  # Green for success, red for fail
                        
                        # add text using cv2
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # Scale font based on image size (smaller for 256x256)
                        h, w = frame.shape[:2]
                        font_scale = 0.5 if w <= 256 else 1.0
                        thickness = 1 if w <= 256 else 2
                        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                        
                        # Position text at top-right corner
                        text_x = frame.shape[1] - text_size[0] - 5
                        text_y = text_size[1] + 5
                        
                        # Add background rectangle for better visibility
                        cv2.rectangle(frame, 
                                    (text_x - 3, text_y - text_size[1] - 3),
                                    (text_x + text_size[0] + 3, text_y + 3),
                                    (0, 0, 0), -1)
                        
                        # Add text
                        cv2.putText(frame, label, (text_x, text_y), 
                                  font, font_scale, color, thickness, cv2.LINE_AA)

                    # Removed behavior_name display
                    
                    normalized_frames.append(frame)
                
                # use imageio to create video
                imageio.mimsave(
                    video_path,
                    normalized_frames,
                    fps=self.fps,
                    codec='libx264',
                    quality=8
                )
                log.info(f"Saved trajectory video to {video_path} ({len(normalized_frames)} frames)")
    
    def clear(self):
        """clear all frames and trajectory state"""
        self.frames.clear()
        self.clear_trajectory()
    
    def reset(self):
        """reset (same as clear)"""
        self.clear()


def save_pointcloud_with_keypoints_ply(
    points: np.ndarray,
    rgb: np.ndarray,
    keypoints_3d: np.ndarray,
    save_path: str,
    keypoint_radius: float = 0.01,
    keypoint_num_points: int = 50
):
    """
    Save point cloud with keypoints to PLY file. Keypoints are marked in red.
    
    Args:
        points: Point cloud (H, W, 3) or (N, 3) in world coordinates
        rgb: RGB image (H, W, 3) uint8 for coloring the point cloud
        keypoints_3d: Keypoint positions (K, 3) in world coordinates
        save_path: Path to save the PLY file
        keypoint_radius: Radius of the sphere around each keypoint
        keypoint_num_points: Number of points to add around each keypoint
    """
    # Flatten points if needed
    if points.ndim == 3:
        H, W, _ = points.shape
        points_flat = points.reshape(-1, 3)
        rgb_flat = rgb.reshape(-1, 3)
    else:
        points_flat = points
        rgb_flat = rgb
    
    # Filter valid points (non-zero depth)
    valid_mask = np.linalg.norm(points_flat, axis=1) > 0.01
    valid_points = points_flat[valid_mask]
    valid_colors = rgb_flat[valid_mask]
    
    # Create keypoint markers (small spheres in red)
    keypoint_points = []
    keypoint_colors = []
    
    for kp in keypoints_3d:
        if np.any(np.isnan(kp)) or np.linalg.norm(kp) < 0.01:
            continue
        # Add random points around keypoint to form a visible sphere
        offsets = np.random.randn(keypoint_num_points, 3) * keypoint_radius
        sphere_points = kp + offsets
        keypoint_points.append(sphere_points)
        # Red color for keypoints
        keypoint_colors.append(np.full((keypoint_num_points, 3), [255, 0, 0], dtype=np.uint8))
    
    # Combine all points
    if keypoint_points:
        all_points = np.vstack([valid_points] + keypoint_points)
        all_colors = np.vstack([valid_colors] + keypoint_colors)
    else:
        all_points = valid_points
        all_colors = valid_colors
    
    # Write PLY file
    num_points = len(all_points)
    with open(save_path, 'w') as f:
        # PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(num_points):
            x, y, z = all_points[i]
            r, g, b = all_colors[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    log.info(f"Saved point cloud with {len(keypoints_3d)} keypoints to {save_path}")

