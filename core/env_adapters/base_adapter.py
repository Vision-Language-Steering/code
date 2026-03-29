"""
Base Environment Adapter - Abstract interface for different simulation backends.

This provides a unified API for:
- Getting end-effector pose
- Converting delta actions to target poses
- Getting robot base pose
- Getting camera parameters
- Tracking objects/actors in the scene
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from numba.core.types import none
import numpy as np
import torch
import gym


@dataclass
class Pose3D:
    """
    A simple 3D pose representation that works across all backends.
    
    Attributes:
        position: (3,) array of [x, y, z] in meters
        quaternion: (4,) array of [w, x, y, z] quaternion
    """
    position: np.ndarray  # (3,)
    quaternion: np.ndarray  # (4,) [w, x, y, z]
    
    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float32).reshape(3)
        self.quaternion = np.asarray(self.quaternion, dtype=np.float32).reshape(4)
    
    def to_transformation_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix."""
        from scipy.spatial.transform import Rotation
        mat = np.eye(4, dtype=np.float32)
        # quaternion is [w, x, y, z], scipy expects [x, y, z, w]
        quat_scipy = np.array([self.quaternion[1], self.quaternion[2], 
                               self.quaternion[3], self.quaternion[0]])
        mat[:3, :3] = Rotation.from_quat(quat_scipy).as_matrix()
        mat[:3, 3] = self.position
        return mat
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        from scipy.spatial.transform import Rotation
        # quaternion is [w, x, y, z], scipy expects [x, y, z, w]
        quat_scipy = np.array([self.quaternion[1], self.quaternion[2], 
                               self.quaternion[3], self.quaternion[0]])
        return Rotation.from_quat(quat_scipy).as_matrix().astype(np.float32)
    
    @classmethod
    def from_transformation_matrix(cls, mat: np.ndarray) -> "Pose3D":
        """Create from 4x4 transformation matrix."""
        from scipy.spatial.transform import Rotation
        position = mat[:3, 3].copy()
        quat_scipy = Rotation.from_matrix(mat[:3, :3]).as_quat()  # [x, y, z, w]
        quaternion = np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])  # [w, x, y, z]
        return cls(position=position, quaternion=quaternion)
    
    def __mul__(self, other: "Pose3D") -> "Pose3D":
        """Compose two poses: self * other."""
        mat = self.to_transformation_matrix() @ other.to_transformation_matrix()
        return Pose3D.from_transformation_matrix(mat)
    
    def inverse(self) -> "Pose3D":
        """Return the inverse pose."""
        mat = np.linalg.inv(self.to_transformation_matrix())
        return Pose3D.from_transformation_matrix(mat)


@dataclass
class CameraParams:
    """
    Camera parameters for projection.
    
    Attributes:
        intrinsic: (3, 3) camera intrinsic matrix
        extrinsic: (4, 4) camera extrinsic matrix (world to camera)
        width: image width in pixels
        height: image height in pixels
    """
    intrinsic: np.ndarray  # (3, 3)
    extrinsic: np.ndarray  # (4, 4)
    width: int
    height: int


@dataclass
class TrackedObject:
    """
    Represents a tracked object in the scene.
    
    Attributes:
        name: object name/identifier
        pose: current pose in world frame
        obj_ref: reference to the underlying object (Actor, etc.)
    """
    name: str
    pose: Pose3D
    obj_ref: Any = None


@dataclass
class InteractableObject:
    """
    Represents an interactable/touchable object in the scene for segmentation.
    
    Attributes:
        name: object name/identifier
        object_id: body unique ID
        link_index: Link index (-1 for base link)
        segment_index: Index in the interactable objects list (used in processed segmentation)
        obj_ref: Optional reference to the underlying object (actor/link/etc.)
    """
    name: str
    object_id: int
    link_index: int = -1
    segment_index: int = 0
    obj_ref: any = None


class BaseEnvAdapter(gym.Env):
    """
    Abstract base class for environment adapters.
    
    All simulation backends (CALVIN, LIBERO, real-world) should implement this interface.
    """
    
    def __init__(self, env_or_robot, env_config: dict, device: str = "cuda"):
        """
        Initialize the adapter.
        
        Args:
            env_or_robot: The underlying environment or robot interface
            env_config: The environment configuration
            device: Torch device for tensor operations
        """
        super().__init__()
        self._env = env_or_robot
        self._env_config = env_config
        self._device = device

        self.vlm_camera = env_config.get('vlm_camera', "")

        self.episode_max_steps = env_config.get('max_episode_steps', 200)
        self.episode_step = 0
    
    @property
    def device(self) -> str:
        return self._device
    
    @property
    def env(self):
        """Access the underlying environment."""
        return self._env
    
    @property
    def env_config(self) -> dict:
        """Access the environment configuration."""
        return self._env_config
    
    # ==================== Core Robot State ====================
    
    @abstractmethod
    def get_ee_pose(self) -> Pose3D:
        """
        Get current end-effector pose in robot base frame.
        
        Returns:
            Pose3D: End-effector pose (position + quaternion)
        """
        pass
    
    @abstractmethod
    def get_ee_pose_world(self) -> Pose3D:
        """
        Get current end-effector pose in world frame.
        
        Returns:
            Pose3D: End-effector pose in world frame
        """
        pass
    
    @abstractmethod
    def get_robot_base_pose(self) -> Pose3D:
        """
        Get robot base pose in world frame.
        
        Returns:
            Pose3D: Robot base pose
        """
        pass
    
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions.
        
        Returns:
            (N,) array of joint positions in radians
        """
        pass
    
    @abstractmethod
    def get_gripper_state(self) -> float:
        """
        Get current gripper state.
        
        Returns:
            Gripper opening width or normalized value
        """
        pass
    
    # ==================== Action Processing ====================
    @abstractmethod
    def delta_actions_to_ee_trajectory(
        self,
        action_sequence: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        transform delta action sequence to 3D trajectory of end-effector.
        
        Args:
            action_sequence: (T, action_dim) action sequence
            start_pose: start pose, if None then use current end-effector pose
            return_numpy: whether to return numpy array (default return tensor to support gradient)
            trajectory_scale: trajectory scale factor
        
        Returns:
            trajectory_3d: (T+1, 3) 3D position trajectory (including start point)
        """
        pass
    
    @abstractmethod
    def get_action_space_info(self) -> Dict[str, Any]:
        """
        Get information about the action space.
        
        Returns:
            Dict with keys like:
            - 'dim': action dimension
            - 'type': 'delta_ee_pose', 'absolute_ee_pose', 'joint_pos', etc.
            - 'bounds': (low, high) arrays
            - 'normalized': whether actions are normalized to [-1, 1]
        """
        pass
    
    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Unnormalize action from [-1, 1] to actual action space.
        
        Default implementation assumes action is already unnormalized.
        Override in subclasses if needed.
        
        Args:
            action: Normalized action
        
        Returns:
            Unnormalized action
        """
        return action
    
    # ==================== Camera & Perception ====================
    @abstractmethod
    def get_vlm_image(self) -> np.ndarray:
        """
        Get image for VLM.
        """
        pass
    
    @abstractmethod
    def get_camera_params(self, camera_name: str) -> CameraParams:
        """
        Get camera intrinsic and extrinsic parameters.
        
        Args:
            camera_name: Name of the camera
        
        Returns:
            CameraParams object
        """
        pass
    
    @abstractmethod
    def get_camera_names(self) -> List[str]:
        """
        Get list of available camera names.
        
        Returns:
            List of camera name strings
        """
        pass
    
    def project_3d_to_2d(
        self, 
        points_3d: np.ndarray, 
        camera_name: str
    ) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: (N, 3) array of 3D points in world frame
            camera_name: Name of the camera
        
        Returns:
            (N, 2) array of 2D pixel coordinates
        """
        params = self.get_camera_params(camera_name)
        
        # Transform to camera frame
        points_homo = np.hstack([points_3d, np.ones((len(points_3d), 1))])
        points_cam = (params.extrinsic @ points_homo.T).T[:, :3]
        
        # Project to image
        points_2d_homo = (params.intrinsic @ points_cam.T).T
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        return points_2d
    
    # ==================== Scene Objects ====================
    
    @abstractmethod
    def get_scene_objects(self, exclude_names: Optional[List[str]] = None) -> List[TrackedObject]:
        """
        Get all trackable objects in the scene.
        
        Args:
            exclude_names: List of object names to exclude
        
        Returns:
            List of TrackedObject instances
        """
        pass
    
    @abstractmethod
    def get_object_pose(self, object_name: str) -> Optional[Pose3D]:
        """
        Get pose of a specific object by name.
        
        Args:
            object_name: Name of the object
        
        Returns:
            Pose3D or None if object not found
        """
        pass
    
    @abstractmethod
    def get_object_pose_by_segment(self, segment_index: int) -> Optional[Pose3D]:
        """
        Get current world pose of an object/link by its segment index.
        
        This is used for keypoint tracking - given a segment_index from the 
        processed segmentation mask, get the current pose of that object/link.
        
        Args:
            segment_index: The segment index from processed segmentation
            
        Returns:
            Pose3D or None if not found
        """
        pass

    # ==================== Segmentation Processing ====================
    
    @abstractmethod
    def get_interactable_objects(self) -> List["InteractableObject"]:
        """
        Get list of interactable/touchable objects in the scene.
        
        Returns:
            List of InteractableObject instances with their names and IDs
        """
        pass
    
    @abstractmethod
    def process_segmentation(
        self, 
        seg_image: np.ndarray,
        camera_name: str = "static"
    ) -> Tuple[np.ndarray, List["InteractableObject"], Dict[int, str]]:
        """
        Process raw segmentation image to filter only interactable objects.
        
        Args:
            seg_image: Raw segmentation image from camera
            camera_name: Name of the camera
        
        Returns:
            processed_seg: Segmentation image where each pixel is either:
                          0 (background/non-interactable) or
                          index+1 (1-indexed) of the interactable object
            interactable_list: List of InteractableObject in the image
            segment_id_to_name: Dict mapping segment index to object name
        """
        pass
    
    @abstractmethod
    def get_keypoint_detection_inputs(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Get all inputs needed for keypoint detection.
        
        This provides a unified interface for keypoint detector across different backends.
        
        Returns:
            rgb: RGB image (H, W, 3), uint8
            depth: Depth image (H, W), float32 in meters
            points: Point cloud (H, W, 3) in world coordinates
            segmentation: Processed segmentation with interactable objects only
            segment_id_to_name: Dict mapping segment index to object name
        """
        pass
    
    @abstractmethod
    def get_policy_observation(self, sample_num: int = 1) -> Dict[str, Any]:
        """
        Get observation in the format expected by the policy.
        
        This converts raw environment observations to the format expected by
        diffusion policy or other learned policies.
        
        Args:
            sample_num: Number of samples to expand batch dimension
        
        Returns:
            Dict with keys like:
            - "observation.state": robot state tensor
            - "observation.images.<camera_name>": image tensors
        """
        pass

    # ==================== Behavior Analysis ====================
    @abstractmethod
    def check_success(self) -> Tuple[bool, str]:
        """Check if the task is successful."""
        pass

    @abstractmethod
    def get_behavior_static(self) -> Dict[str, int]:
        """Get static behavior information."""
        pass
    
    # ==================== Task-Specific Guidance ====================
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task-related information for guidance adjustment.
        
        This allows adapters to provide task-specific hints for steering.
        
        Returns:
            Dict with keys like:
            - 'instruction': task instruction string
            - 'recommended_guide_scale': suggested guide_scale value
            - 'task_type': e.g., 'manipulation', 'navigation'
            - 'requires_precision': bool for fine-grained tasks
        """
        return {
            'instruction': self._env_config.get('instruction', ''),
            'recommended_guide_scale': None,  # None = use default
            'task_type': 'manipulation',
            'requires_precision': False,
        }
    
    def get_instruction(self) -> str:
        """Get the current task instruction."""
        return self._env_config.get('instruction', '')
    
    # ==================== Environment Interface ====================
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Step the environment with an action.
        
        Args:
            action: Action to execute
        
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        pass
    
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """
        Reset the environment.
        
        Returns:
            (obs, info)
        """
        pass
    
    @abstractmethod
    def get_obs(self) -> Dict:
        """
        Get current observation.
        
        Returns:
            Observation dictionary
        """
        pass
    
    # ==================== Coordinate Transforms ====================
    
    def transform_pose_to_world(self, pose_base: Pose3D) -> Pose3D:
        """
        Transform a pose from robot base frame to world frame.
        
        Args:
            pose_base: Pose in robot base frame
        
        Returns:
            Pose in world frame
        """
        base_pose = self.get_robot_base_pose()
        return base_pose * pose_base
    
    def transform_pose_to_base(self, pose_world: Pose3D) -> Pose3D:
        """
        Transform a pose from world frame to robot base frame.
        
        Args:
            pose_world: Pose in world frame
        
        Returns:
            Pose in robot base frame
        """
        base_pose = self.get_robot_base_pose()
        return base_pose.inverse() * pose_world
    
    def transform_points_to_world(self, points_base: np.ndarray) -> np.ndarray:
        """
        Transform points from robot base frame to world frame.
        
        Args:
            points_base: (N, 3) points in robot base frame
        
        Returns:
            (N, 3) points in world frame
        """
        base_mat = self.get_robot_base_pose().to_transformation_matrix()
        points_homo = np.hstack([points_base, np.ones((len(points_base), 1))])
        points_world = (base_mat @ points_homo.T).T[:, :3]
        return points_world
    
    # ==================== Tensor Utilities ====================
    
    def to_tensor(self, arr: np.ndarray, requires_grad: bool = False) -> torch.Tensor:
        """Convert numpy array to torch tensor on the correct device."""
        tensor = torch.from_numpy(arr).float().to(self._device)
        if requires_grad:
            tensor.requires_grad_(True)
        return tensor
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert torch tensor to numpy array."""
        return tensor.detach().cpu().numpy()


