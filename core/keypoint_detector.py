#!/usr/bin/env python3
"""
Keypoint Detector - A backend-agnostic class for detecting keypoints from observations

This module provides a KeypointDetector class that works with any simulation backend
through the adapter interface. It only depends on standard numpy arrays as input.

Supports both DINOv2 and DINOv3 feature extractors.

Usage:
    # Get data from adapter (any backend: CALVIN, LIBERO, real-world, etc.)
    rgb, depth, points, seg, seg_names = adapter.get_keypoint_detection_inputs("camera")

    # Detect keypoints (backend-agnostic)
    keypoints, projected, mask_ids = detector.get_keypoints_from_data(rgb, points, seg, seg_names)
"""

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
from sklearn.cluster import MeanShift

from utils.keypoint_utils import (
    preprocess_for_dinov2,
    preprocess_masks,
    filter_points_by_bounds,
    project_keypoints_to_img,
)

from utils.logging_utils import SteerLogger

logger = SteerLogger("KeypointDetector")

# DINOv3 model variants (HuggingFace)
DINOV3_MODELS = {
    'dinov3_vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'dinov3_vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'dinov3_vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'dinov3_vith16': 'facebook/dinov3-vith16-pretrain-lvd1689m',
    'dinov3_vit7b16': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
    'dinov3_convnext_tiny': 'facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
    'dinov3_convnext_small': 'facebook/dinov3-convnext-small-pretrain-lvd1689m',
    'dinov3_convnext_base': 'facebook/dinov3-convnext-base-pretrain-lvd1689m',
    'dinov3_convnext_large': 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
}

# DINOv2 model variants (torch.hub)
DINOV2_MODELS = {
    'dinov2_vits14': 'dinov2_vits14',
    'dinov2_vitb14': 'dinov2_vitb14',
    'dinov2_vitl14': 'dinov2_vitl14',
    'dinov2_vitg14': 'dinov2_vitg14',
}


class KeypointDetector:
    """
    Backend-agnostic keypoint detector using DINO features (v2 or v3).

    This class handles the core keypoint detection logic including:
    - Feature extraction using DINOv2 or DINOv3
    - Feature clustering in semantic space
    - Keypoint merging and filtering

    Usage:
        detector = KeypointDetector(config)

        # Get data from any adapter (CALVIN, LIBERO, real-world, etc.)
        rgb, depth, points, seg, seg_names = adapter.get_keypoint_detection_inputs("camera")

        # Detect keypoints
        keypoints, projected, mask_ids = detector.get_keypoints_from_data(rgb, points, seg, seg_names)

    Config options:
        - feature_extractor: 'dinov2_vits14', 'dinov3_vitb16', etc. (default: 'dinov3_vitb16')
        - See DINOV2_MODELS and DINOV3_MODELS for all options
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.config['device'])
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.mean_shift = MeanShift(bandwidth=self.config['min_dist_bt_keypoints'], bin_seeding=True, n_jobs=32)

        # Determine which feature extractor to use
        self.feature_extractor_name = config.get('feature_extractor', 'dinov3_vitb16')
        self._load_feature_extractor()

        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        logger.info(f"KeypointDetector initialized on device: {self.device}")

    def _load_feature_extractor(self):
        """Load the appropriate feature extractor (DINOv2 or DINOv3)."""
        name = self.feature_extractor_name

        if name in DINOV3_MODELS:
            self._load_dinov3(name)
        elif name in DINOV2_MODELS:
            self._load_dinov2(name)
        else:
            # Default fallback to DINOv2 vits14
            logger.warning(f"Unknown feature extractor '{name}', falling back to dinov2_vits14")
            self._load_dinov2('dinov2_vits14')

    def _load_dinov2(self, model_name: str):
        """Load DINOv2 model from torch.hub."""
        logger.info(f"Loading DINOv2 model: {model_name}")
        self.feature_extractor_type = 'dinov2'
        self.patch_size = 14  # DINOv2 uses patch size 14
        self.dino_model = torch.hub.load(
            'facebookresearch/dinov2',
            model_name
        ).eval().to(self.device)
        logger.info(f"DINOv2 model loaded: {model_name}")

    def _load_dinov3(self, model_name: str):
        """Load DINOv3 model from HuggingFace."""
        logger.info(f"Loading DINOv3 model: {DINOV3_MODELS[model_name]}")
        self.feature_extractor_type = 'dinov3'
        self.patch_size = 16  # DINOv3 uses patch size 16

        try:
            from transformers import AutoImageProcessor, AutoModel

            hf_model_name = DINOV3_MODELS[model_name]
            self.dinov3_processor = AutoImageProcessor.from_pretrained(hf_model_name)
            self.dino_model = AutoModel.from_pretrained(
                hf_model_name,
                torch_dtype=torch.float16 if 'cuda' in str(self.device) else torch.float32
            ).eval().to(self.device)

            logger.info(f"DINOv3 model loaded: {hf_model_name}")

        except ImportError as e:
            logger.error(
                "transformers>=4.56.0 required for DINOv3. "
                "Install with: pip install transformers>=4.56.0"
            )
            raise ImportError(f"Cannot load DINOv3: {e}")
        except Exception as e:
            logger.warning(f"Failed to load DINOv3 ({e}), falling back to DINOv2")
            self._load_dinov2('dinov2_vits14')
    
    def get_keypoints(
        self, 
        rgb: np.ndarray,
        points: np.ndarray,
        segmentation: np.ndarray,
        segment_id_to_name: Optional[Dict[int, str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect keypoints from provided data (works with any backend).
        
        This is the main entry point for keypoint detection when using adapters.
        Use adapter.get_keypoint_detection_inputs() to get the required inputs.
        
        Args:
            rgb: RGB image (H, W, 3), uint8
            points: Point cloud (H, W, 3) in world coordinates
            segmentation: Processed segmentation image (H, W), int
                         0 = background, >0 = object segment index
            segment_id_to_name: Optional dict mapping segment index to object name
        
        Returns:
            keypoints: (N, 3) array of 3D keypoint positions
            projected: RGB image with keypoints drawn on it
            mask_ids: (N,) array of segmentation mask IDs for each keypoint
        
        Example:
            # Get data from adapter
            rgb, depth, points, seg, seg_names = adapter.get_keypoint_detection_inputs("static")
            
            # Detect keypoints
            keypoints, projected, mask_ids = detector.get_keypoints_from_data(rgb, points, seg, seg_names)
        """
        # Print segment info if available
        if segment_id_to_name is not None:
            print(f"\n{'-'*50}")
            print(f"[KEYPOINT DETECTOR] Segments in image:")
            for seg_idx, name in segment_id_to_name.items():
                if seg_idx > 0:
                    pixel_count = np.sum(segmentation == seg_idx)
                    if pixel_count > 0:
                        print(f"  [{seg_idx}] {name}: {pixel_count} pixels")
            print(f"{'-'*50}\n")
        
        # Preprocess RGB and masks
        transformed_rgb, shape_info = preprocess_for_dinov2(rgb, self.patch_size)
        masks_data = preprocess_masks(segmentation)
        
        # Get DINOv2 features
        features_flat = self._get_features(transformed_rgb, shape_info)
        
        # Cluster features to get keypoint candidates
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(
            points, features_flat, masks_data
        )
        
        if len(candidate_keypoints) == 0:
            logger.warning("Warning: No keypoint candidates found!")
            return np.array([]), rgb.copy(), np.array([])
        
        # Exclude keypoints that are outside of the workspace
        within_space = filter_points_by_bounds(
            candidate_keypoints, self.bounds_min, self.bounds_max, strict=True
        )
        
        # Ensure at least one keypoint per mask category is kept
        unique_mask_ids = np.unique(candidate_rigid_group_ids)
        kept_indices_list = []
        
        for mask_id in unique_mask_ids:
            mask_keypoint_indices = np.where(candidate_rigid_group_ids == mask_id)[0]
            mask_kept_indices = mask_keypoint_indices[within_space[mask_keypoint_indices]]
            
            if len(mask_kept_indices) > 0:
                kept_indices_list.extend(mask_kept_indices.tolist())
            elif len(mask_keypoint_indices) > 0:
                kept_indices_list.append(mask_keypoint_indices[0])
        
        if len(kept_indices_list) == 0:
            logger.warning("No keypoints within workspace bounds!")
            return np.array([]), rgb.copy(), np.array([])
        
        kept_indices = np.array(kept_indices_list)
        candidate_keypoints = candidate_keypoints[kept_indices]
        candidate_pixels = candidate_pixels[kept_indices]
        candidate_rigid_group_ids = candidate_rigid_group_ids[kept_indices]
        
        # Merge close points
        merged_indices = self._merge_clusters(candidate_keypoints)
        merged_keypoints = candidate_keypoints[merged_indices]
        merged_pixels = candidate_pixels[merged_indices]
        merged_mask_ids = candidate_rigid_group_ids[merged_indices]
        
        # Ensure at least one keypoint per mask category after merging
        final_indices_list = []
        final_mask_ids_list = []
        used_mask_ids = set()
        
        for mask_id in unique_mask_ids:
            mask_merged_indices = np.where(merged_mask_ids == mask_id)[0]
            if len(mask_merged_indices) > 0:
                # Keep ALL keypoints per mask (not just one) so VLM can pick the right one
                final_indices_list.extend(mask_merged_indices.tolist())
                final_mask_ids_list.extend([mask_id] * len(mask_merged_indices))
                used_mask_ids.add(mask_id)
                
        for mask_id in unique_mask_ids:
            if mask_id not in used_mask_ids:
                mask_original_indices = np.where(candidate_rigid_group_ids == mask_id)[0]
                if len(mask_original_indices) > 0:
                    first_kp = candidate_keypoints[mask_original_indices[0]]
                    distances = np.linalg.norm(merged_keypoints - first_kp, axis=1)
                    closest_merged_idx = np.argmin(distances)
                    if closest_merged_idx not in final_indices_list:
                        final_indices_list.append(closest_merged_idx)
                        final_mask_ids_list.append(mask_id)
        
        final_indices = np.array(final_indices_list)
        candidate_keypoints = merged_keypoints[final_indices]
        candidate_pixels = merged_pixels[final_indices]
        candidate_rigid_group_ids = np.array(final_mask_ids_list)
        
        # Sort by mask ID for consistent ordering
        sort_idx = np.argsort(candidate_rigid_group_ids)
        candidate_keypoints = candidate_keypoints[sort_idx]
        candidate_pixels = candidate_pixels[sort_idx]
        candidate_rigid_group_ids = candidate_rigid_group_ids[sort_idx]

        
        
        # Project keypoints to image
        projected = project_keypoints_to_img(rgb, candidate_pixels, candidate_rigid_group_ids)
        
        return candidate_keypoints, projected, candidate_rigid_group_ids

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_features(self, transformed_rgb, shape_info):
        """
        Extract features using DINOv2 or DINOv3.

        Args:
            transformed_rgb: Preprocessed RGB image (H, W, 3) normalized to [0, 1]
            shape_info: Dict with img_h, img_w, patch_h, patch_w

        Returns:
            features_flat: (H*W, feature_dim) tensor of per-pixel features
        """
        img_h = shape_info['img_h']
        img_w = shape_info['img_w']
        patch_h = shape_info['patch_h']
        patch_w = shape_info['patch_w']

        if self.feature_extractor_type == 'dinov3':
            return self._get_features_dinov3(transformed_rgb, img_h, img_w)
        else:
            return self._get_features_dinov2(transformed_rgb, img_h, img_w, patch_h, patch_w)

    def _get_features_dinov2(self, transformed_rgb, img_h, img_w, patch_h, patch_w):
        """Extract features using DINOv2."""
        img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        assert img_tensors.shape[1] == 3, "unexpected image shape"

        features_dict = self.dino_model.forward_features(img_tensors)
        raw_feature_grid = features_dict['x_norm_patchtokens']  # [1, patch_h*patch_w, feature_dim]
        raw_feature_grid = raw_feature_grid.reshape(1, patch_h, patch_w, -1)

        # Interpolate to full resolution
        interpolated_feature_grid = interpolate(
            raw_feature_grid.permute(0, 3, 1, 2),
            size=(img_h, img_w),
            mode='bilinear'
        ).permute(0, 2, 3, 1).squeeze(0)  # [H, W, feature_dim]

        features_flat = interpolated_feature_grid.reshape(-1, interpolated_feature_grid.shape[-1])
        return features_flat

    def _get_features_dinov3(self, transformed_rgb, img_h, img_w):
        """Extract features using DINOv3 (HuggingFace transformers)."""
        from PIL import Image
        import math

        # Convert to uint8 PIL Image for the processor
        rgb_uint8 = (transformed_rgb * 255).astype(np.uint8)
        pil_image = Image.fromarray(rgb_uint8)

        # Process image with DINOv3 processor
        inputs = self.dinov3_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get features
        outputs = self.dino_model(**inputs)

        # DINOv3 returns last_hidden_state with shape [1, num_patches + 1, feature_dim]
        # The +1 is for the CLS token which we skip
        features = outputs.last_hidden_state[:, 1:, :]  # Skip CLS token

        num_patches = features.shape[1]
        feature_dim = features.shape[2]

        # Infer patch grid dimensions from actual number of patches
        processed_h = inputs['pixel_values'].shape[2]
        processed_w = inputs['pixel_values'].shape[3]

        # Calculate expected patches based on image size and patch size
        patch_h = processed_h // self.patch_size
        patch_w = processed_w // self.patch_size

        # If that doesn't match, try to infer from num_patches (assume square-ish)
        if patch_h * patch_w != num_patches:
            # Try to find factors that work
            patch_h = int(math.sqrt(num_patches))
            patch_w = num_patches // patch_h
            if patch_h * patch_w != num_patches:
                # If still doesn't work, try closest square
                patch_h = int(math.sqrt(num_patches))
                patch_w = patch_h
                # Truncate features if needed
                num_patches = patch_h * patch_w
                features = features[:, :num_patches, :]

        # Reshape to spatial grid
        raw_feature_grid = features.reshape(1, patch_h, patch_w, feature_dim)

        # Interpolate to original image size
        interpolated_feature_grid = interpolate(
            raw_feature_grid.permute(0, 3, 1, 2).float(),
            size=(img_h, img_w),
            mode='bilinear'
        ).permute(0, 2, 3, 1).squeeze(0)  # [H, W, feature_dim]

        features_flat = interpolated_feature_grid.reshape(-1, feature_dim)
        return features_flat

    def _cluster_features(self, points, features_flat, masks_data):
        candidate_keypoints = []
        candidate_pixels = []
        candidate_rigid_group_ids = []
        # Unpack masks_data: (binary_masks, original_ids)
       
        masks_binary, original_mask_ids = masks_data
        
        for idx, binary_mask in enumerate(masks_binary):
            rigid_group_id = original_mask_ids[idx]  # Use original mask ID, not enumerate index
            # ignore mask that is too large
            if np.mean(binary_mask) > self.config['max_mask_ratio']:
                continue
            # consider only foreground features
            obj_features_flat = features_flat[binary_mask.reshape(-1)]
            feature_pixels = np.argwhere(binary_mask)
            feature_points = points[binary_mask]

            # Filter out invalid depth points
            # Invalid points are set to (0,0,0) by the adapter when depth is invalid
            # Check if point is not at origin (all zeros) using L2 norm
            valid_depth_mask = np.linalg.norm(feature_points, axis=1) > 1e-6
            if valid_depth_mask.sum() < self.config['num_candidates_per_mask']:
                logger.warning(f"Mask {idx}: Too few valid depth points ({valid_depth_mask.sum()}), skipping")
                continue
            obj_features_flat = obj_features_flat[valid_depth_mask]
            feature_pixels = feature_pixels[valid_depth_mask]
            feature_points = feature_points[valid_depth_mask]

            # reduce dimensionality to be less sensitive to noise and texture
            obj_features_flat = obj_features_flat.double()
            
            # Skip if too few points for clustering
            num_points = obj_features_flat.shape[0]
            num_clusters = self.config['num_candidates_per_mask']
            if num_points < num_clusters:
                continue
            
            (u, s, v) = torch.pca_lowrank(obj_features_flat, center=False)
            features_pca = torch.mm(obj_features_flat, v[:, :3])
            
            # Safe normalization to avoid NaN (add epsilon to prevent div by zero)
            pca_min = features_pca.min(0)[0]
            pca_max = features_pca.max(0)[0]
            pca_range = pca_max - pca_min
            pca_range = torch.where(pca_range == 0, torch.ones_like(pca_range), pca_range)
            features_pca = (features_pca - pca_min) / pca_range
            X = features_pca
            
            # add feature_pixels as extra dimensions
            feature_points_torch = torch.tensor(feature_points, dtype=features_pca.dtype, device=features_pca.device)
            
            # Safe normalization for points
            pts_min = feature_points_torch.min(0)[0]
            pts_max = feature_points_torch.max(0)[0]
            pts_range = pts_max - pts_min
            pts_range = torch.where(pts_range == 0, torch.ones_like(pts_range), pts_range)
            feature_points_torch = (feature_points_torch - pts_min) / pts_range
            
            X = torch.cat([X, feature_points_torch], dim=-1)
            
            # Check for NaN values
            if torch.isnan(X).any():
                logger.warning(f"Warning: NaN detected in features for mask {idx}, skipping")
                continue
            
            # the feature dimension contains both feature and pixel coordinates
            # cluster features to get meaningful regions
            cluster_ids_x, cluster_centers = kmeans(
                X=X,
                num_clusters=self.config['num_candidates_per_mask'],
                distance='euclidean',
                device=self.device,
            )
            cluster_centers = cluster_centers.to(self.device)
            for cluster_id in range(self.config['num_candidates_per_mask']):
                cluster_center = cluster_centers[cluster_id][:3]
                member_idx = cluster_ids_x == cluster_id
                member_points = feature_points[member_idx]
                member_pixels = feature_pixels[member_idx]
                member_features = features_pca[member_idx]

                dist = torch.norm(member_features - cluster_center, dim=-1)
                closest_idx = torch.argmin(dist)
                candidate_keypoints.append(member_points[closest_idx])
                candidate_pixels.append(member_pixels[closest_idx])

                candidate_rigid_group_ids.append(rigid_group_id)

        candidate_keypoints = np.array(candidate_keypoints)
        candidate_pixels = np.array(candidate_pixels)
        candidate_rigid_group_ids = np.array(candidate_rigid_group_ids)

        return candidate_keypoints, candidate_pixels, candidate_rigid_group_ids

    def _merge_clusters(self, candidate_keypoints):
        self.mean_shift.fit(candidate_keypoints)
        cluster_centers = self.mean_shift.cluster_centers_
        merged_indices = []
        for center in cluster_centers:
            dist = np.linalg.norm(candidate_keypoints - center, axis=-1)
            merged_indices.append(np.argmin(dist))
        return merged_indices
    
    def visualize_dino_features(
        self,
        rgb: np.ndarray,
        segmentation: np.ndarray,
        keypoints_pixels: np.ndarray,
        keypoint_ids: np.ndarray,
        save_path: str = None,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Visualize DINOv2 features projected onto segmentation masks with keypoints.
        
        Creates a visualization similar to "SAM + DINOv2" style where:
        - Masked regions are colored by PCA of DINOv2 features
        - Keypoints are marked with numbered indices
        
        Args:
            rgb: RGB image (H, W, 3), uint8
            segmentation: Segmentation mask (H, W), int
            keypoints_pixels: Pixel coordinates of keypoints (N, 2) as [row, col]
            keypoint_ids: IDs/indices for each keypoint (N,)
            save_path: Optional path to save the visualization
            alpha: Blending alpha for feature overlay
            
        Returns:
            Visualization image (H, W, 3), uint8
        """
        # Get DINOv2 features
        transformed_rgb, shape_info = preprocess_for_dinov2(rgb, self.patch_size)
        features_flat = self._get_features(transformed_rgb, shape_info)
        
        H, W = rgb.shape[:2]
        
        # Reshape features to image shape
        features_img = features_flat.reshape(H, W, -1).cpu().numpy()
        
        # PCA to 3 channels for RGB visualization
        features_2d = features_img.reshape(-1, features_img.shape[-1])
        
        # Use SVD for PCA (faster for this use case)
        mean = features_2d.mean(axis=0)
        features_centered = features_2d - mean
        U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)
        
        # Project to 3 dimensions
        features_pca = features_centered @ Vt[:3].T
        
        # Normalize to [0, 255] for each channel
        features_pca = features_pca.reshape(H, W, 3)
        for c in range(3):
            channel = features_pca[:, :, c]
            min_val, max_val = channel.min(), channel.max()
            if max_val > min_val:
                features_pca[:, :, c] = (channel - min_val) / (max_val - min_val) * 255
            else:
                features_pca[:, :, c] = 128
        
        features_rgb = features_pca.astype(np.uint8)
        
        # Create output image
        output = rgb.copy()
        
        # Create mask for all segmented regions (non-background)
        mask = segmentation > 0
        
        # Blend DINOv2 features on masked regions
        output[mask] = (
            alpha * features_rgb[mask] + 
            (1 - alpha) * output[mask]
        ).astype(np.uint8)
        
        # Draw keypoints
        for i, (pixel, kp_id) in enumerate(zip(keypoints_pixels, keypoint_ids)):
            row, col = int(pixel[0]), int(pixel[1])
            
            # Draw circle
            cv2.circle(output, (col, row), 8, (255, 255, 255), -1)
            cv2.circle(output, (col, row), 8, (0, 0, 0), 2)
            
            # Draw index number
            text = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = col - text_size[0] // 2
            text_y = row + text_size[1] // 2
            cv2.putText(output, text, (text_x, text_y), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            print(f"Saved DINOv2 feature visualization to: {save_path}")
        
        return output
    
    def get_keypoints_with_visualization(
        self, 
        rgb: np.ndarray,
        points: np.ndarray,
        segmentation: np.ndarray,
        segment_id_to_name: Optional[Dict[int, str]] = None,
        save_dir: str = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect keypoints and generate DINOv2 feature visualization.
        
        Same as get_keypoints() but also returns/saves the DINOv2 feature visualization.
        
        Args:
            rgb: RGB image (H, W, 3), uint8
            points: Point cloud (H, W, 3) in world coordinates
            segmentation: Processed segmentation image (H, W), int
            segment_id_to_name: Optional dict mapping segment index to object name
            save_dir: Directory to save visualization (optional)
            
        Returns:
            keypoints: (N, 3) array of 3D keypoint positions
            projected: RGB image with keypoints drawn on it
            mask_ids: (N,) array of segmentation mask IDs for each keypoint
            dino_vis: DINOv2 feature visualization image
        """
        # Run normal keypoint detection
        keypoints, projected, mask_ids = self.get_keypoints(
            rgb, points, segmentation, segment_id_to_name
        )
        
        if len(keypoints) == 0:
            return keypoints, projected, mask_ids, rgb.copy()
        
        # Get pixel coordinates for visualization
        # We need to recompute since get_keypoints doesn't return pixels
        transformed_rgb, shape_info = preprocess_for_dinov2(rgb, self.patch_size)
        masks_data = preprocess_masks(segmentation)
        features_flat = self._get_features(transformed_rgb, shape_info)
        
        candidate_keypoints, candidate_pixels, candidate_rigid_group_ids = self._cluster_features(
            points, features_flat, masks_data
        )
        
        # Match returned keypoints to get their pixels
        keypoint_pixels = []
        for kp in keypoints:
            dists = np.linalg.norm(candidate_keypoints - kp, axis=1)
            closest_idx = np.argmin(dists)
            keypoint_pixels.append(candidate_pixels[closest_idx])
        keypoint_pixels = np.array(keypoint_pixels)
        
        # Generate visualization
        save_path = None
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "dino_features.png")
        
        dino_vis = self.visualize_dino_features(
            rgb, segmentation, keypoint_pixels, mask_ids, save_path
        )
        
        return keypoints, projected, mask_ids, dino_vis


