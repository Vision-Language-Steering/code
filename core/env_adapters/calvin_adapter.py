"""
CALVIN Environment Adapter

Adapter for CALVIN (PyBullet) environments.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import pybullet as p
from PIL import Image
from scipy.spatial.transform import Rotation as R
from .base_adapter import BaseEnvAdapter, Pose3D, CameraParams, TrackedObject, InteractableObject


BEHAVIOR_NAMES = [
    "no_behavior",
    "red_displace",
    "pink_displace",
    "blue_displace",
    "door_left",
    "door_right",
    "drawer_close",
    "drawer_open",
    "switch_off",
    "switch_on",
    "button_off",
    "button_on",
]

ADJUSTABLE_LIMITS = {"sliding_door" : [0, 0.27], "drawer" : [0, 0.16], "switch" : [0, 0.08], "green_light" : [0, 1]}
ADJUSTABLE_RATIO = {"sliding_door" : 0.9, "drawer" : 0.6, "switch" : 0.7, "green_light" : 0.5}

# ==================== Utility Functions ====================

@staticmethod
def _euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert euler angles (roll, pitch, yaw) to quaternion [w, x, y, z].
    """
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler('xyz', euler)
    quat_xyzw = r.as_quat()  # [x, y, z, w]
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

@staticmethod
def _quaternion_to_euler(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to euler angles (roll, pitch, yaw).
    """
    from scipy.spatial.transform import Rotation
    quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])  # [x, y, z, w]
    r = Rotation.from_quat(quat_xyzw)
    return r.as_euler('xyz')


class CalvinAdapter(BaseEnvAdapter):
    """
    Adapter for CALVIN environments.
    
    CALVIN uses PyBullet.
    
    """
    
    # CALVIN's default action scaling (from their controller)
    ACTION_SCALE_POS = 0.02  # meters per action unit
    ACTION_SCALE_ROT = 0.05  # radians per action unit
    
    def __init__(self, env, env_config: dict, device: str = "cuda"):
        """
        Initialize CALVIN adapter.
        
        Args:
            env: CALVIN environment (from calvin_env.envs.play_table_env)
            device: Torch device
        """
        super().__init__(env, env_config, device)

        self.init_scene_state_dict = None
        self.scene_state_dict = None
        self.robot_state_dict = None

        self.behavior_dict = {k: 0 for k in BEHAVIOR_NAMES}

        # Target behavior filter - only count success if this specific behavior happens
        # If None, any behavior counts as success (original behavior)
        self.target_behavior = env_config.get('target_behavior', None)
    
    # ==================== Core Robot State ====================
    def get_ee_pose(self) -> Pose3D:
        """
        Get end-effector pose in robot base frame.
        """
        return Pose3D(
            position=self.robot_state_dict["tcp_pos"],
            quaternion=_euler_to_quaternion(self.robot_state_dict["tcp_orn"])
        )

    def get_ee_pose_world(self) -> Pose3D:
        """
        Get end-effector pose in world frame.
        """
        return self.get_ee_pose().to_world_pose()
    
    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions.
        """
        return self.robot_state_dict["arm_joint_pos"]

    def get_gripper_state(self) -> float:
        """
        Get current gripper state.
        """
        return self.robot_state_dict["gripper_width"]
    
    def get_robot_base_pose(self) -> Pose3D:
        """
        Get robot base pose in world frame.
        """
        return self._robot_base_pose
    
    # ==================== Action Processing ====================
    def delta_actions_to_ee_trajectory(
        self,
        action_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        transform delta action sequence to 3D trajectory of end-effector.
        
        CALVIN uses relative end-effector control:
        - action[:3]: delta position (dx, dy, dz)
        - action[3:6]: delta rotation (euler angles)
        - action[6]: gripper action
        
        Args:
            action_sequence: (T, action_dim) action sequence, can be numpy or tensor
            start_pose: start pose, if None then use current end-effector pose
            return_numpy: whether to return numpy array (default return tensor to support gradient)
            trajectory_scale: trajectory scale factor
        
        Returns:
            trajectory_3d: (T+1, 3) 3D position trajectory (including start point)
        
        Note:
            Uses differentiable operations (cumsum, cat) to preserve gradient flow.
            Avoid in-place operations which would break the computation graph.
        """
        
        device = action_sequence.device
        dtype = action_sequence.dtype
        
        # Get starting position as tensor
        start_pos = torch.tensor(
            self.get_ee_pose().position, 
            device=device, 
            dtype=dtype
        )
        
        # Compute delta positions (T, 3) - this preserves gradient
        delta_positions = action_sequence[:, :3] * self.ACTION_SCALE_POS
        
        # Cumulative sum of deltas - differentiable operation
        cumsum_deltas = torch.cumsum(delta_positions, dim=0)  # (T, 3)
        
        # Build trajectory: [start_pos, start_pos + cumsum[0], start_pos + cumsum[1], ...]
        # trajectory[0] = start_pos
        # trajectory[t+1] = start_pos + cumsum_deltas[t] for t in 0..T-1
        trajectory = torch.cat([
            start_pos.unsqueeze(0),           # (1, 3)
            start_pos + cumsum_deltas         # (T, 3)
        ], dim=0)  # (T+1, 3)
    
        return trajectory

    
    def delta_to_target_pose(
        self, 
        current_pose: Pose3D, 
        delta_action: np.ndarray,
        action_scale: float = 1.0
    ) -> Pose3D:
        """
        Apply delta action to get target pose.
        
        CALVIN uses relative EE control with specific scaling.
        
        Args:
            current_pose: Current EE pose
            delta_action: Delta action [dx, dy, dz, droll, dpitch, dyaw] (6D)
            action_scale: Additional scale factor
        
        Returns:
            Target pose
        """
        delta_action = np.asarray(delta_action).reshape(-1)
        
        # Strip gripper if present
        if len(delta_action) == 7:
            delta_action = delta_action[:6]
        
        # Apply CALVIN's action scaling
        delta_pos = delta_action[:3] * self.ACTION_SCALE_POS * action_scale
        delta_rot = delta_action[3:6] * self.ACTION_SCALE_ROT * action_scale
        
        # Compute new position
        new_pos = current_pose.position + delta_pos
        
        # Compute new orientation (apply delta euler to current euler)
        current_euler = self._quaternion_to_euler(current_pose.quaternion)
        new_euler = current_euler + delta_rot
        new_quat = self._euler_to_quaternion(new_euler)
        
        return Pose3D(position=new_pos, quaternion=new_quat)
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """Get action space information."""
        return {
            'dim': 7,
            'type': 'delta_ee_pose',
            'bounds': (np.array([-1.0] * 7), np.array([1.0] * 7)),
            'normalized': True,
            'pos_scale': self.ACTION_SCALE_POS,
            'rot_scale': self.ACTION_SCALE_ROT,
        }
    
    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """
        Unnormalize action from normalized to actual delta values.
        
        CALVIN actions are typically in [-1, 1] range.
        """
        action = np.asarray(action).reshape(-1)
        unnorm = np.zeros_like(action)
        
        # Position: multiply by pos_scale
        unnorm[:3] = action[:3] * self.ACTION_SCALE_POS
        # Rotation: multiply by rot_scale
        unnorm[3:6] = action[3:6] * self.ACTION_SCALE_ROT
        # Gripper: keep as is
        if len(action) > 6:
            unnorm[6] = action[6]
        
        return unnorm
    
    # ==================== Camera & Perception ====================
    def get_vlm_image(self) -> np.ndarray:
        """
        Get image for VLM.
        """
        image = self._env.render(mode="rgb_array", height=640, width=640)
        return image
    
    def get_camera_params(self, camera_name: str) -> CameraParams:
        """
        Get camera parameters for CALVIN.
        
        CALVIN has 'static' and 'gripper' cameras.
        Computes proper intrinsic and extrinsic matrices from camera config.
        """
        if camera_name in ['static', 'rgb_static', "third_person"]:
            # Static camera parameters from config
            look_from = np.array([2.871459009488717, -2.166602199425597, 2.555159848480571])
            look_at = np.array([-0.026242351159453392, -0.0302329882979393, 0.3920000493526459])
            up_vector = np.array([0.4041403970338857, 0.22629790978217404, 0.8862616969685161])
            fov = 10  # degrees
            width, height = 640, 640
            
            # Compute intrinsic matrix from FOV
            fov_rad = np.radians(fov)
            focal_length = (height / 2) / np.tan(fov_rad / 2)
            cx, cy = width / 2, height / 2
            
            intrinsic = np.array([
                [focal_length, 0.0, cx],
                [0.0, focal_length, cy],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
            
            # Compute extrinsic matrix (world-to-camera transform)
            # Camera coordinate system: Z forward, X right, Y down
            forward = look_at - look_from
            forward = forward / np.linalg.norm(forward)
            
            right = np.cross(forward, up_vector)
            right = right / np.linalg.norm(right)
            
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            
            # Rotation matrix: camera axes in world frame
            # OpenCV convention: X-right, Y-down, Z-forward
            R_cam = np.array([
                right,      # X axis
                -up,        # Y axis (negated because Y points down in camera)
                forward     # Z axis
            ], dtype=np.float32)
            
            # Translation: camera position in world
            t = -R_cam @ look_from
            
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = R_cam
            extrinsic[:3, 3] = t
            
        else:
            raise ValueError(f"Unknown camera: {camera_name}. Available: static")
        
        return CameraParams(
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            width=width,
            height=height
        )
    
    def get_camera_names(self) -> List[str]:
        """Get available camera names."""
        return ['static']
    
    # ==================== Scene Objects ====================

    def get_link_pose(self, object_id: int, link_index: int) -> Pose3D:
        """
        Get the world pose of a specific link using PyBullet.
        
        Args:
            object_id: PyBullet body ID
            link_index: Link index (-1 for base link)
            
        Returns:
            Pose3D with world position and quaternion
        """
        cid = self._env.cid
        
        if link_index == -1:
            # Base link - use getBasePositionAndOrientation
            pos, orn = p.getBasePositionAndOrientation(object_id, physicsClientId=cid)
        else:
            # Other links - use getLinkState
            link_state = p.getLinkState(object_id, link_index, physicsClientId=cid)
            pos = link_state[0]  # World position of link frame
            orn = link_state[1]  # World orientation (quaternion xyzw)
        
        # Convert quaternion from PyBullet (xyzw) to (wxyz)
        quat_wxyz = np.array([orn[3], orn[0], orn[1], orn[2]])
        
        return Pose3D(position=np.array(pos), quaternion=quat_wxyz)
    
    def get_scene_objects(self) -> List[TrackedObject]:
        """
        Get trackable objects in CALVIN scene with their world positions.
        
        This includes:
        - Movable blocks (red, blue, pink) - from scene_state_dict
        - Articulated parts (drawer, slide, button, switch) - from PyBullet link poses
        """

        objects = []
        cid = self._env.cid
        
        # Get interactable objects (which already have object_id and link_index)
        interactables = self.get_interactable_objects()
        
        for interactable in interactables:
            # Get world pose from PyBullet
            pose = self.get_link_pose(interactable.object_id, interactable.link_index)
            
            objects.append(TrackedObject(
                name=interactable.name,
                pose=pose,
                obj_ref={
                    'object_id': interactable.object_id,
                    'link_index': interactable.link_index,
                    'segment_index': interactable.segment_index
                }
            ))
        
        return objects
    
    def get_object_pose_by_segment(self, segment_index: int) -> Optional[Pose3D]:
        """
        Get current world pose of an object by its segment index.
        
        This is useful for keypoint tracking - given a segment_id from the 
        segmentation mask, get the current pose of that object/link.
        
        Args:
            segment_index: The segment index from processed segmentation
            
        Returns:
            Pose3D or None if not found
        """
        interactables = self.get_interactable_objects()
        
        for interactable in interactables:
            if interactable.segment_index == segment_index:
                return self.get_link_pose(interactable.object_id, interactable.link_index)
        
        return None
    
    def get_object_pose(self, object_name: str) -> Optional[Pose3D]:
        """Get pose of a specific object."""
        objects = self.get_scene_objects()
        for obj in objects:
            if obj.name == object_name:
                return obj.pose
        return None

    # ==================== Segmentation Processing ====================
    
    def get_interactable_objects(self) -> List[InteractableObject]:
        """
        Get list of interactable/touchable objects in CALVIN scene.
        
        This includes:
        - Colored blocks (red, blue, pink)
        - Table components: button, switch, drawer, slide (sliding door)
        - Excludes: robot, ground plane, table base, lights
        """
        interactables = []
        cid = self._env.cid
        
        # Build mapping of all objects and links
        num_bodies = p.getNumBodies(physicsClientId=cid)
        segment_index = 1  # Start from 1 (0 is reserved for background)
        
        for body_id in range(num_bodies):
            body_info = p.getBodyInfo(body_id, physicsClientId=cid)
            body_name = body_info[1].decode('utf-8')
            
            # Skip robot
            if 'panda' in body_name.lower():
                continue
            
            # Skip ground plane
            if 'plane' in body_name.lower():
                continue
            
            num_joints = p.getNumJoints(body_id, physicsClientId=cid)
            
            # Check if it's a movable block
            if 'block' in body_name.lower():
                interactables.append(InteractableObject(
                    name=body_name,
                    object_id=body_id,
                    link_index=-1,  # Base link
                    segment_index=segment_index
                ))
                segment_index += 1
                continue
            
            # Check if it's the playtable with interactive components
            if 'playtable' in body_name.lower() or 'table' in body_name.lower():
                # Add interactive links (drawer, switch, button, slide)
                for joint_idx in range(num_joints):
                    joint_info = p.getJointInfo(body_id, joint_idx, physicsClientId=cid)
                    link_name = joint_info[12].decode('utf-8')
                    
                    # Only include interactive components
                    interactive_keywords = ['button', 'switch', 'drawer', 'slide']
                    if any(kw in link_name.lower() for kw in interactive_keywords):
                        interactables.append(InteractableObject(
                            name=link_name,
                            object_id=body_id,
                            link_index=joint_idx,
                            segment_index=segment_index
                        ))
                        segment_index += 1
        
        return interactables
    
    def process_segmentation(
        self, 
        seg_image: np.ndarray,
    ) -> Tuple[np.ndarray, List[InteractableObject], Dict[int, str]]:
        """
        Process raw segmentation image to filter only interactable objects.
        
        Args:
            seg_image: Raw segmentation image from camera (PyBullet encoded)
        
        Returns:
            processed_seg: Segmentation image with interactable object indices
            interactable_list: List of InteractableObject found in the image
            segment_id_to_name: Dict mapping segment index to object name
        """
        # Get interactable objects
        interactables = self.get_interactable_objects()
        
        # Build lookup table: (object_id, link_index) -> InteractableObject
        lookup = {}
        for obj in interactables:
            lookup[(obj.object_id, obj.link_index)] = obj
        
        # Decode PyBullet segmentation
        object_ids = seg_image & ((1 << 24) - 1)
        link_indices = (seg_image >> 24) - 1
        
        # Create processed segmentation image
        processed_seg = np.zeros_like(seg_image, dtype=np.int32)
        segment_id_to_name = {0: "background"}
        found_objects = []
        
        # Get unique (object_id, link_index) combinations
        unique_pairs = set()
        for obj_id, link_id in zip(object_ids.flatten(), link_indices.flatten()):
            if obj_id > 0:
                unique_pairs.add((int(obj_id), int(link_id)))
        
        # Process each unique pair
        for obj_id, link_id in unique_pairs:
            key = (obj_id, link_id)
            if key in lookup:
                interactable = lookup[key]
                # Create mask for this object+link
                mask = (object_ids == obj_id) & (link_indices == link_id)
                processed_seg[mask] = interactable.segment_index
                
                if interactable not in found_objects:
                    found_objects.append(interactable)
                    segment_id_to_name[interactable.segment_index] = interactable.name
        
        return processed_seg, found_objects, segment_id_to_name
    
    def get_keypoint_detection_inputs(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
        """
        Get all inputs needed for keypoint detection.
        
        Args:
            camera_name: Name of the camera to use
        
        Returns:
            rgb: RGB image (H, W, 3), uint8
            depth: Depth image (H, W), float32 in meters  
            points: Point cloud (H, W, 3) in world coordinates
            segmentation: Processed segmentation with interactable objects only
            segment_id_to_name: Dict mapping segment index to object name
        """

        rgb, depth, seg_raw = self._env.render(mode="rgb+depth+segmentation", height=640, width=640)
        
        # Process segmentation to get only interactable objects
        segmentation, interactable_list, segment_id_to_name = self.process_segmentation(
            seg_raw
        )   
        
        # Convert depth to point cloud
        camera_params = self.get_camera_params(self.vlm_camera)
        points = self._depth_to_pointcloud(depth, camera_params)
        
        return rgb, depth, points, segmentation, segment_id_to_name
    
    def _depth_to_pointcloud(self, depth: np.ndarray, camera_params) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.
        
        Args:
            depth: Depth image (H, W) in meters
            camera_params: CameraParams object
            
        Returns:
            points: Point cloud (H, W, 3) in world coordinates
        """
        H, W = depth.shape
        
        # Get camera intrinsics
        intrinsic = camera_params.intrinsic
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        
        # Get camera extrinsics and compute cam2world
        extrinsic = camera_params.extrinsic
        cam2world = np.linalg.inv(extrinsic)
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Convert to camera space
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Stack to get points in camera space
        points_cam = np.stack([x, y, z], axis=-1)
        
        # Filter invalid points
        valid_mask = (z > 0) & (z < 10.0)
        
        # Transform to world space
        points_cam_flat = points_cam.reshape(-1, 3)
        points_cam_homo = np.concatenate([points_cam_flat, np.ones((points_cam_flat.shape[0], 1))], axis=-1)
        points_world_homo = (points_cam_homo @ cam2world.T)[:, :3]
        
        # Create output array
        points_full = np.zeros((H, W, 3), dtype=np.float32)
        points_world_2d = points_world_homo.reshape(H, W, 3)
        points_full[valid_mask] = points_world_2d[valid_mask]
        
        return points_full
    
    def get_policy_observation(self, sample_num: int = 1) -> Dict[str, torch.Tensor]:
        """
        Get observation in the format expected by the policy.
        
        CALVIN observation structure:
        - robot_obs: [tcp_pos(3), tcp_orn(3), gripper(1), joints(7), gripper_action(1)]
        - rgb_obs: {'rgb_static': ..., 'rgb_gripper': ...}
        
        Args:
            sample_num: Number of samples to expand batch dimension
        
        Returns:
            Dict with policy-expected keys
        """
        obs = self.get_obs()

        robot_obs = obs.get('robot_obs', np.zeros(15))
        rgb_static = obs.get('rgb_obs', {}).get('rgb_static', np.zeros((200, 200, 3), dtype=np.uint8))
        rgb_gripper = obs.get('rgb_obs', {}).get('rgb_gripper', np.zeros((84, 84, 3), dtype=np.uint8))

        # Ensure uint8 for PIL
        if rgb_static.dtype != np.uint8:
            rgb_static = rgb_static.astype(np.uint8)
        if rgb_gripper.dtype != np.uint8:
            rgb_gripper = rgb_gripper.astype(np.uint8)

        # Resize images
        rgb_static_img = Image.fromarray(rgb_static).resize((256, 256), Image.BILINEAR)
        rgb_gripper_img = Image.fromarray(rgb_gripper).resize((256, 256), Image.BILINEAR)
        
        rgb_static = np.array(rgb_static_img)
        rgb_gripper = np.array(rgb_gripper_img)
        
        # Convert to tensor
        if isinstance(robot_obs, np.ndarray):
            robot_obs = torch.from_numpy(robot_obs).float()
        if isinstance(rgb_static, np.ndarray):
            rgb_static = torch.from_numpy(rgb_static).float()
            if rgb_static.max() > 1.0:
                rgb_static = rgb_static / 255.0
        if isinstance(rgb_gripper, np.ndarray):
            rgb_gripper = torch.from_numpy(rgb_gripper).float()
            if rgb_gripper.max() > 1.0:
                rgb_gripper = rgb_gripper / 255.0
        
        # Adjust dimension: (H, W, C) -> (C, H, W)
        if rgb_static.ndim == 3:
            rgb_static = rgb_static.permute(2, 0, 1)
        if rgb_gripper.ndim == 3:
            rgb_gripper = rgb_gripper.permute(2, 0, 1)
        
        observation = {
            "observation.state": robot_obs.cuda().unsqueeze(0),
            "observation.images.third_person": rgb_static.cuda().unsqueeze(0),
            "observation.images.eye_in_hand": rgb_gripper.cuda().unsqueeze(0),
        }
        
        # Expand batch dimension
        observation = {k: v.expand(sample_num, *v.shape[1:]) for k, v in observation.items()}
        return observation

    # ==================== Task Success and Static Analysis ====================

    def update_state_dict(self, obs):
        robot_obs = obs['robot_obs']
        scene_obs = obs['scene_obs']
        self.robot_state_dict = {
            "tcp_pos": robot_obs[:3],
            "tcp_orn": robot_obs[3:6],
            "gripper_width": robot_obs[6],
            "arm_joint_pos": robot_obs[7:14],
            "gripper_action": robot_obs[14]
        }
        self.scene_state_dict = {
            "sliding_door" : scene_obs[0],
            "drawer" : scene_obs[1],
            "button" : scene_obs[2],
            "switch" : scene_obs[3],
            "lightbulb" : scene_obs[4],
            "green_light" : scene_obs[5],
            "red_block": scene_obs[6:9], # we are ignoring rotations 
            "blue_block" : scene_obs[12 : 15],
            "pink_block" : scene_obs[18 :21],
            "red_rot": R.from_euler("XYZ", scene_obs[9:12]).as_matrix(), # we are ignoring rotations 
            "blue_rot" : R.from_euler("XYZ", scene_obs[15:18]).as_matrix(),
            "pink_rot" : R.from_euler("XYZ", scene_obs[21:24]).as_matrix()
        }

    def check_success(self) -> Tuple[bool, str]:
        """Check if the task is successful."""
        if self.init_scene_state_dict is None:
            self.init_scene_state_dict = self.scene_state_dict
            return False, "no_behavior"

        delta_state = {
            k : np.linalg.norm(self.scene_state_dict[k] - self.init_scene_state_dict[k]) for k in self.scene_state_dict.keys()
        }
        tcp_pos = self.robot_state_dict["tcp_pos"]
        gripper_action = self.robot_state_dict["gripper_action"]
        behavior_happened = False
        behavior_name = "no_behavior"
        # TODO: if multiple are touched at the same time 
        if np.linalg.norm(tcp_pos - self.scene_state_dict["red_block"]) < 0.06 \
            and delta_state["red_block"] > 0.005\
            and gripper_action < 0:
            behavior_name = "red_displace"
            behavior_happened = True 
            self.behavior_dict[behavior_name] += 1

        if np.linalg.norm(tcp_pos - self.scene_state_dict["blue_block"]) < 0.06 \
            and delta_state["blue_block"] > 0.005\
            and gripper_action < 0:
            behavior_name = "blue_displace"
            behavior_happened = True 
            self.behavior_dict[behavior_name] += 1

        if np.linalg.norm(tcp_pos - self.scene_state_dict["pink_block"]) < 0.06 \
            and delta_state["pink_block"] > 0.005\
            and gripper_action < 0:
            behavior_name = "pink_displace"
            behavior_happened = True 
            self.behavior_dict[behavior_name] += 1

        if delta_state["sliding_door"] > (ADJUSTABLE_LIMITS["sliding_door"][1] - ADJUSTABLE_LIMITS["sliding_door"][0]) * ADJUSTABLE_RATIO["sliding_door"]: 
            if self.scene_state_dict["sliding_door"] < self.init_scene_state_dict["sliding_door"]:
                behavior_name = "door_right"
            else:
                behavior_name = "door_left"
            behavior_happened = True
            self.behavior_dict[behavior_name] += 1

        if delta_state["drawer"] > (ADJUSTABLE_LIMITS["drawer"][1] - ADJUSTABLE_LIMITS["drawer"][0]) * ADJUSTABLE_RATIO["drawer"]: 
            if self.scene_state_dict["drawer"] > self.init_scene_state_dict["drawer"]:
                behavior_name = "drawer_open"
            else:
                behavior_name = "drawer_close"
            behavior_happened = True
            self.behavior_dict[behavior_name] += 1

        if delta_state["switch"] > (ADJUSTABLE_LIMITS["switch"][1] - ADJUSTABLE_LIMITS["switch"][0]) * ADJUSTABLE_RATIO["switch"]:
            if self.scene_state_dict["switch"] > self.init_scene_state_dict["switch"]:
                behavior_name = "switch_on"
            else:
                behavior_name = "switch_off"
            behavior_happened = True
            self.behavior_dict[behavior_name] += 1

        if delta_state["green_light"] > (ADJUSTABLE_LIMITS["green_light"][1] - ADJUSTABLE_LIMITS["green_light"][0]) * ADJUSTABLE_RATIO["green_light"]:
            if self.scene_state_dict["green_light"] > self.init_scene_state_dict["green_light"]:
                behavior_name = "button_on"
            else:
                behavior_name = "button_off"
            behavior_happened = True
            self.behavior_dict[behavior_name] += 1

        if behavior_happened:
            # If target_behavior is set, only count success if the specific behavior matches
            if self.target_behavior is not None:
                if behavior_name == self.target_behavior:
                    return True, behavior_name
                else:
                    # Behavior happened but not the target - don't count as success
                    return False, behavior_name
            return True, behavior_name
        else:
            return False, "no_behavior"

    def get_behavior_static(self) -> Dict[str, int]:
        """Get static behavior information."""
        return self.behavior_dict

    # ==================== Environment Interface ====================
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Step the CALVIN environment."""

        if torch.is_tensor(action):
            action = action.cpu().numpy().reshape(-1)

        action[-1] = 1 if action[-1] > 0 else -1

        # CALVIN step returns (obs, reward, done, info) - no truncated
        obs, reward, done, info = self._env.step(action)
        self.update_state_dict(obs)
        self.episode_step += 1
        info['success'], info["behavior_name"] = self.check_success()
        done = info['success']

        truncated = self.episode_step >= self.episode_max_steps
        if truncated:
            self.behavior_dict["no_behavior"] += 1

        return obs, reward, done, truncated, info

    
    def reset(self, **kwargs) -> Tuple[Dict, Dict]:
        """Reset the CALVIN environment."""
        result = self._env.reset()
        self.update_state_dict(result)
        self.init_scene_state_dict = None
        self.episode_step = 0
        # CALVIN reset may return just obs or (obs, info)
        if isinstance(result, tuple):
            return result
        else:
            return result, {}
    
    def get_obs(self) -> Dict:
        """Get current observation."""
        if hasattr(self._env, 'get_obs'):
            return self._env.get_obs()
        return {}
    
    # ==================== Task-Specific Guidance ====================
    # Task types that require higher guide_scale (block manipulation)
    PRECISION_TASKS = ['red_displace', 'blue_displace', 'pink_displace']
    PRECISION_GUIDE_SCALE = 120.0
    DEFAULT_GUIDE_SCALE = 80.0
    
    # Mapping from target_behavior to instruction
    BEHAVIOR_INSTRUCTIONS = {
        'drawer_open': 'open the drawer',
        'drawer_close': 'close the drawer',
        'door_left': 'slide the door to the left',
        'door_right': 'slide the door to the right',
        'switch_on': 'turn on the switch',
        'switch_off': 'turn off the switch',
        'button_on': 'press the button to turn on the light',
        'button_off': 'press the button to turn off the light',
        'red_displace': 'push the red block',
        'blue_displace': 'push the blue block',
        'pink_displace': 'push the pink block',
    }
    
    def get_task_info(self) -> Dict[str, Any]:
        """
        Get task-related information for guidance adjustment.
        
        Returns task-specific hints based on target_behavior.
        """
        target = self.target_behavior
        is_precision_task = target in self.PRECISION_TASKS
        
        return {
            'instruction': self.get_instruction(),
            'recommended_guide_scale': self.PRECISION_GUIDE_SCALE if is_precision_task else self.DEFAULT_GUIDE_SCALE,
            'task_type': 'precision_manipulation' if is_precision_task else 'manipulation',
            'requires_precision': is_precision_task,
            'target_behavior': target,
        }
    
    def get_instruction(self) -> str:
        """Get the current task instruction based on target_behavior."""
        # First check env_config
        instruction = self._env_config.get('instruction', '')
        if instruction:
            return instruction
        # Fall back to behavior-based instruction
        if self.target_behavior:
            return self.BEHAVIOR_INSTRUCTIONS.get(self.target_behavior, f'perform {self.target_behavior}')
        return ''


