"""
Keypoint Tracker - A backend-agnostic class for tracking keypoints attached to objects.

Supports CALVIN, LIBERO, and real-world environments.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .env_adapters import BaseEnvAdapter, Pose3D, TrackedObject, create_adapter

from utils.logging_utils import SteerLogger
log = SteerLogger("KeypointTracker")

@dataclass
class KeypointInfo:
    """
    Store information for a single keypoint.
    """
    index: int
    initial_world_pos: np.ndarray  # initial world coordinates
    local_pos: np.ndarray  # local coordinates relative to the attached object (link frame)
    segment_index: int = 0  # segment index in the segmentation mask
    object_name: str = ""  # object name


class KeypointTracker:
    """
    Generic keypoint tracker.
    
    Supports CALVIN, LIBERO, and real-world environments.
    Uses EnvAdapter abstract layer to handle differences between different backends.
    
    Usage:
        from core.env_adapters import create_adapter
        
        adapter = create_adapter("calvin", env_config)
        tracker = KeypointTracker(adapter)
        tracker.register_keypoints(initial_keypoints)
        
        # ... objects move ...
        current_positions = tracker.get_keypoint_positions()
    """
    
    def __init__(
        self, 
        adapter: BaseEnvAdapter,
        exclude_objects: Optional[List[str]] = None
    ):
        """
        Initialize keypoint tracker.
        
        Args:
            adapter: EnvAdapter instance
            exclude_objects: list of object names to exclude
        """
        self.adapter = adapter
        self._keypoints: Optional[np.ndarray] = None
        self._keypoint_registry: Dict[int, KeypointInfo] = {}
    
    def register_keypoints(
        self, 
        keypoints: np.ndarray,
        mask_ids: np.ndarray,
        segment_id_to_name: Dict[int, str],
    ) -> Dict[int, str]:
        """
        Register keypoints and attach them to objects in the scene.
        
        Args:
            keypoints: (N, 3) keypoint positions in world frame
            mask_ids: (N,) segment index for each keypoint
            segment_id_to_name: dictionary mapping segment index to object name
        """
        keypoints = np.asarray(keypoints)
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(-1, 3)
        
        assert keypoints.shape[1] == 3, f"Keypoints must be (N, 3), got {keypoints.shape}"
        
        self._keypoints = keypoints.copy()
        self._keypoint_registry = {}
        
        # Attach each keypoint to the corresponding object
        for idx, keypoint in enumerate(keypoints):
            keypoint = keypoint.reshape(3)
            segment_index = int(mask_ids[idx])
            object_name = segment_id_to_name.get(segment_index, f"unknown_{segment_index}")
            
            log.debug(f"Keypoint {idx}: segment={segment_index}, object={object_name}")
            
            # Get the pose of the object corresponding to this segment from the adapter
            pose = self.adapter.get_object_pose_by_segment(segment_index)
            
            if pose is not None:
                # Compute the position of the keypoint in the local coordinate system of the object
                local_pos = self._compute_local_position(keypoint, pose)
                # Compute object position in robot base frame
                robot_base_pose = self.adapter.get_robot_base_pose()
                robot_to_world = robot_base_pose.to_transformation_matrix()
                world_to_robot = np.linalg.inv(robot_to_world)
                
                # Object position in robot frame
                obj_pos_homo = np.append(pose.position, 1.0)
                obj_pos_robot = (world_to_robot @ obj_pos_homo)[:3]
                
                log.debug(f"  Attached to {object_name}")
                log.debug(f"    Object world_pos={pose.position}")
                log.debug(f"    Object robot_pos={obj_pos_robot}")
                log.debug(f"    Keypoint local_pos={local_pos}")

               
            else:
                # Cannot get pose, use world coordinates (will not move with the object)
                local_pos = keypoint.copy()
                log.warning(f"  Could not get pose for segment {segment_index}, using world coords")
            self._keypoint_registry[idx] = KeypointInfo(
                index=idx,
                initial_world_pos=keypoint.copy(),
                local_pos=local_pos,
                segment_index=segment_index,
                object_name=object_name
            )

        key_points_objects_map = {
            idx: info.object_name for idx, info in self._keypoint_registry.items()
        }
        log.info(f"Registered {len(keypoints)} keypoints")
        return key_points_objects_map
    
    def _compute_local_position(self, world_pos: np.ndarray, object_pose: Pose3D) -> np.ndarray:
        """Compute the position of a point in the local coordinate system of an object."""
        inv_mat = np.linalg.inv(object_pose.to_transformation_matrix())
        pos_homo = np.append(world_pos, 1.0)
        local_pos = (inv_mat @ pos_homo)[:3]
        return local_pos.copy()
    
    def _compute_world_position(self, local_pos: np.ndarray, object_pose: Pose3D) -> np.ndarray:
        """Convert local coordinates to world coordinates."""
        mat = object_pose.to_transformation_matrix()
        pos_homo = np.append(local_pos, 1.0)
        world_pos = (mat @ pos_homo)[:3]
        return world_pos.copy()
    
    def get_keypoint_positions(self) -> np.ndarray:
        """
        Get the current world coordinates of all keypoints.
        
        Get the current world coordinates of all keypoints by getting the current pose of each object/link from the adapter, and then converting the local coordinates to world coordinates.
        
        Returns:
            (N, 3) array of current keypoint positions
        
        Raises:
            AssertionError: if keypoints have not been registered yet
        """
        assert self._keypoint_registry, \
            "Keypoints have not been registered yet. Call register_keypoints() first."
        
        positions = []
        
        for idx in range(len(self._keypoints)):
            info = self._keypoint_registry[idx]
            
            if info.segment_index == 0:
                # Background - return initial position
                positions.append(info.initial_world_pos)
            else:
                # Get the current pose of the object from the adapter
                current_pose = self.adapter.get_object_pose_by_segment(info.segment_index)
                
                if current_pose is not None:
                    # Convert local coordinates to world coordinates
                    world_pos = self._compute_world_position(info.local_pos, current_pose)
                    positions.append(world_pos)
                else:
                    # Cannot get pose, return initial position
                    log.warning(f"Could not get pose for keypoint {idx} (segment {info.segment_index})")
                    positions.append(info.initial_world_pos)
        
        return np.array(positions)
    
    def get_object_name_by_keypoint(self, keypoint_idx: int) -> Optional[str]:
        """
        Get the name of the object the keypoint is attached to.
        
        Args:
            keypoint_idx: keypoint index
        
        Returns:
            object name or None
        """
        if keypoint_idx not in self._keypoint_registry:
            return None
        return self._keypoint_registry[keypoint_idx].object_name
    
    def get_mask_ids(self) -> np.ndarray:
        """
        Get the mask IDs of all registered keypoints.
        """
        if not self._keypoint_registry:
            return np.array([])
        return np.array([info.segment_index for idx, info in self._keypoint_registry.items()])

    def reset(self):
        """Reset the tracker, clearing all registered keypoints."""
        self._keypoints = None
        self._keypoint_registry = {}
    
    def visualize_keypoints(self):
        """Print keypoint information (for debugging)."""
        if self._keypoints is None:
            log.info("No keypoints registered")
            return
        
        positions = self.get_keypoint_positions()
        
        log.info(f"Keypoint positions ({len(positions)} total):")
        for idx, pos in enumerate(positions):
            info = self._keypoint_registry[idx]
            log.debug(f"  Keypoint {idx}: {pos} (object: {info.object_name}, segment: {info.segment_index})")
