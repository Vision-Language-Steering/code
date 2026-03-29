#!/usr/bin/env python3
"""
SAM3 Segmenter - Text-prompted segmentation using Meta's Segment Anything Model 3

This module provides text-based segmentation for keypoint detection,
enabling real-world deployment without ground-truth masks.

Usage:
    segmenter = SAM3Segmenter(config)
    segmentation, segment_id_to_name = segmenter.segment(rgb, object_names=["cube", "gripper"])
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from PIL import Image

from utils.logging_utils import SteerLogger

logger = SteerLogger("SAM3Segmenter")


class SAM3Segmenter:
    """
    Text-prompted segmentation using SAM3.

    SAM3 can segment objects based on text descriptions (concept prompts),
    eliminating the need for clicks, boxes, or ground-truth masks.
    """

    def __init__(self, config: dict):
        """
        Initialize SAM3 segmenter.

        Args:
            config: Configuration dictionary with keys:
                - device: 'cuda' or 'cpu'
                - model_path: Optional path to local checkpoint
                - score_threshold: Minimum confidence score (default: 0.5)
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda'))
        self.score_threshold = config.get('score_threshold', 0.5)

        self._load_model()
        logger.info(f"SAM3Segmenter initialized on device: {self.device}")

    def _load_model(self):
        """Load SAM3 model and processor."""
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            logger.info("Loading SAM3 model...")
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            logger.info("SAM3 model loaded successfully")

        except ImportError as e:
            logger.error(
                "SAM3 not installed. Install with:\n"
                "  git clone https://github.com/facebookresearch/sam3.git\n"
                "  cd sam3 && pip install -e ."
            )
            raise ImportError(f"SAM3 not available: {e}")

    def segment(
        self,
        rgb: np.ndarray,
        object_names: List[str],
        return_all_masks: bool = False
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Segment objects in image using text prompts.

        Args:
            rgb: RGB image (H, W, 3), uint8
            object_names: List of object names to segment (e.g., ["red cube", "gripper"])
            return_all_masks: If True, return all detected instances; else best per class

        Returns:
            segmentation: (H, W) array where each pixel has segment ID (0=background)
            segment_id_to_name: Dict mapping segment ID to object name
        """
        H, W = rgb.shape[:2]

        # Ensure uint8 for PIL
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        # Convert to PIL Image
        image = Image.fromarray(rgb)

        # Initialize inference state
        inference_state = self.processor.set_image(image)

        # Output arrays
        segmentation = np.zeros((H, W), dtype=np.int32)
        segment_id_to_name = {0: "background"}
        current_id = 1

        for obj_name in object_names:
            try:
                # Query SAM3 with text prompt
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=obj_name
                )

                masks = output["masks"]  # (N, H, W) boolean masks
                scores = output["scores"]  # (N,) confidence scores

                if masks is None or len(masks) == 0:
                    logger.warning(f"No masks found for '{obj_name}'")
                    continue

                # Convert to numpy if needed
                if torch.is_tensor(masks):
                    masks = masks.cpu().numpy()
                if torch.is_tensor(scores):
                    scores = scores.cpu().numpy()

                # Handle different mask shapes - SAM3 may return various formats
                # Common shapes: (H, W), (N, H, W), (N, 1, H, W), (1, N, H, W)
                logger.debug(f"Raw mask shape for '{obj_name}': {masks.shape}, scores shape: {scores.shape}")

                # Squeeze out all singleton dimensions except the first (N) and last two (H, W)
                if masks.ndim == 2:
                    masks = masks[np.newaxis, ...]  # (H, W) -> (1, H, W)
                elif masks.ndim == 4:
                    # SAM3 returns (N, 1, H, W) where N=3 multimask outputs
                    # Squeeze the middle singleton dimension
                    masks = masks.squeeze(1) if masks.shape[1] == 1 else masks.squeeze(0)

                # Ensure masks is 3D: (N, H, W)
                while masks.ndim > 3:
                    masks = masks.squeeze(0)
                while masks.ndim < 3:
                    masks = masks[np.newaxis, ...]

                if scores.ndim == 0:
                    scores = np.array([scores.item()])  # scalar -> (1,)
                elif scores.ndim >= 2:
                    scores = scores.flatten()  # (1, N) or (B, N) -> flatten

                # Ensure scores matches masks
                if len(scores) != len(masks):
                    logger.warning(f"Score/mask mismatch for '{obj_name}': {len(scores)} scores vs {len(masks)} masks")
                    # Take min length
                    min_len = min(len(scores), len(masks))
                    scores = scores[:min_len]
                    masks = masks[:min_len]

                # Filter by score threshold
                valid_indices = scores >= self.score_threshold

                if not np.any(valid_indices):
                    logger.warning(f"No masks above threshold for '{obj_name}'")
                    continue

                if return_all_masks:
                    # Add all valid masks
                    for idx in np.where(valid_indices)[0]:
                        mask = masks[idx]
                        # Don't overwrite existing segments (first come, first served)
                        new_pixels = mask & (segmentation == 0)
                        segmentation[new_pixels] = current_id
                        segment_id_to_name[current_id] = f"{obj_name}_{idx}"
                        current_id += 1
                else:
                    # Take best mask per class
                    best_idx = np.argmax(scores)
                    if scores[best_idx] >= self.score_threshold:
                        mask = masks[best_idx]
                        new_pixels = mask & (segmentation == 0)
                        segmentation[new_pixels] = current_id
                        segment_id_to_name[current_id] = obj_name
                        current_id += 1

            except Exception as e:
                logger.warning(f"Error segmenting '{obj_name}': {e}")
                continue

        logger.info(f"SAM3 segmented {current_id - 1} objects: {list(segment_id_to_name.values())[1:]}")
        return segmentation, segment_id_to_name

    def segment_with_boxes(
        self,
        rgb: np.ndarray,
        boxes: np.ndarray,
        object_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Segment objects using bounding boxes (fallback mode).

        Args:
            rgb: RGB image (H, W, 3), uint8
            boxes: Bounding boxes (N, 4) as [x1, y1, x2, y2]
            object_names: Optional names for each box

        Returns:
            segmentation: (H, W) array with segment IDs
            segment_id_to_name: Dict mapping segment ID to object name
        """
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb)
        inference_state = self.processor.set_image(image)

        segmentation = np.zeros((H, W), dtype=np.int32)
        segment_id_to_name = {0: "background"}

        if object_names is None:
            object_names = [f"object_{i}" for i in range(len(boxes))]

        for idx, (box, name) in enumerate(zip(boxes, object_names)):
            try:
                output = self.processor.set_box_prompt(
                    state=inference_state,
                    box=box
                )

                masks = output["masks"]
                if masks is not None and len(masks) > 0:
                    if torch.is_tensor(masks):
                        masks = masks.cpu().numpy()
                    mask = masks[0]  # Best mask for this box
                    new_pixels = mask & (segmentation == 0)
                    segmentation[new_pixels] = idx + 1
                    segment_id_to_name[idx + 1] = name

            except Exception as e:
                logger.warning(f"Error segmenting box {idx}: {e}")
                continue

        return segmentation, segment_id_to_name


class SAM3SegmenterMock:
    """
    Mock SAM3 segmenter for testing without the actual model.
    Uses simple color-based segmentation as fallback.
    """

    def __init__(self, config: dict):
        self.config = config
        logger.warning("Using SAM3 MOCK - install SAM3 for real segmentation")

    def segment(
        self,
        rgb: np.ndarray,
        object_names: List[str],
        return_all_masks: bool = False
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """Simple color-based segmentation as fallback."""
        import cv2

        # Ensure uint8 for OpenCV
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)

        H, W = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        segmentation = np.zeros((H, W), dtype=np.int32)
        segment_id_to_name = {0: "background"}

        # Simple color ranges for common objects
        color_ranges = {
            "red": ([0, 100, 100], [10, 255, 255]),
            "blue": ([100, 100, 100], [130, 255, 255]),
            "green": ([40, 100, 100], [80, 255, 255]),
            "pink": ([140, 100, 100], [170, 255, 255]),
            "yellow": ([20, 100, 100], [40, 255, 255]),
        }

        current_id = 1
        for obj_name in object_names:
            # Check if any color keyword is in the object name
            for color, (lower, upper) in color_ranges.items():
                if color in obj_name.lower():
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    new_pixels = (mask > 0) & (segmentation == 0)
                    if np.any(new_pixels):
                        segmentation[new_pixels] = current_id
                        segment_id_to_name[current_id] = obj_name
                        current_id += 1
                    break

        return segmentation, segment_id_to_name


def create_segmenter(config: dict) -> SAM3Segmenter:
    """
    Factory function to create SAM3 segmenter.
    Falls back to mock if SAM3 is not installed.
    """
    try:
        return SAM3Segmenter(config)
    except ImportError:
        logger.warning("SAM3 not available, using mock segmenter")
        return SAM3SegmenterMock(config)
