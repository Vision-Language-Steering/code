#!/usr/bin/env python3
"""
Gemini Grounder - Use Gemini's visual grounding to detect objects via bounding boxes.

This uses Gemini 2.5's built-in spatial understanding capability to detect
objects without needing SAM3 or other segmentation models.
"""

import numpy as np
import json
import base64
import os
from typing import List, Dict, Tuple, Optional
from PIL import Image
import io

from utils.logging_utils import SteerLogger

logger = SteerLogger("GeminiGrounder")


class GeminiGrounder:
    """
    Use Gemini's visual grounding to detect objects and return bounding boxes.

    Gemini can detect objects based on text descriptions and return
    bounding boxes in [y_min, x_min, y_max, x_max] format (normalized 0-1000).
    """

    def __init__(self, config: dict):
        """
        Initialize Gemini grounder.

        Args:
            config: Configuration dict with:
                - api_key: Google API key
                - model: Model name (default: gemini-2.5-flash)
        """
        self.config = config
        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.model = config.get('model', 'gemini-2.5-flash')

        if not self.api_key:
            raise ValueError("Gemini API key required. Set 'api_key' in config or GOOGLE_API_KEY env var.")

        self._setup_client()
        logger.info(f"GeminiGrounder initialized with model: {self.model}")

    def _setup_client(self):
        """Setup Google Generative AI client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
            logger.info("Gemini client initialized")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Install with:\n"
                "  pip install google-generativeai"
            )

    def _image_to_base64(self, rgb: np.ndarray) -> str:
        """Convert numpy RGB image to base64."""
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        img = Image.fromarray(rgb)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def detect_objects(
        self,
        rgb: np.ndarray,
        object_names: List[str],
    ) -> List[Dict]:
        """
        Detect objects in image using Gemini's visual grounding.

        Args:
            rgb: RGB image (H, W, 3), uint8
            object_names: List of object names to detect

        Returns:
            List of dicts with 'label', 'box_2d' (normalized 0-1000), 'box_pixel' (actual pixels)
        """
        import google.generativeai as genai

        H, W = rgb.shape[:2]

        # Build prompt
        objects_str = ", ".join(object_names)
        prompt = f"""Detect these objects in the image and return bounding boxes: {objects_str}

Return a JSON array where each object has:
- "label": the object name
- "box_2d": [y_min, x_min, y_max, x_max] normalized to 0-1000

Example output:
[{{"label": "drawer handle", "box_2d": [500, 200, 600, 400]}}]

Only return the JSON array, no other text. If an object is not found, don't include it."""

        # Convert image
        if rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        img = Image.fromarray(rgb)

        try:
            response = self.client.generate_content([prompt, img])
            response_text = response.text.strip()

            # Parse JSON from response
            # Handle markdown code blocks
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]

            detections = json.loads(response_text)

            # Convert normalized coords to pixel coords
            for det in detections:
                box = det['box_2d']
                det['box_pixel'] = [
                    int(box[0] * H / 1000),  # y_min
                    int(box[1] * W / 1000),  # x_min
                    int(box[2] * H / 1000),  # y_max
                    int(box[3] * W / 1000),  # x_max
                ]

            logger.info(f"Gemini detected {len(detections)} objects: {[d['label'] for d in detections]}")
            return detections

        except Exception as e:
            logger.error(f"Gemini detection failed: {e}")
            return []

    def detect_to_segmentation(
        self,
        rgb: np.ndarray,
        object_names: List[str],
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Detect objects and convert bounding boxes to segmentation mask.

        This creates a simple segmentation where each bounding box region
        is filled with the segment ID.

        Args:
            rgb: RGB image (H, W, 3)
            object_names: List of object names to detect

        Returns:
            segmentation: (H, W) array with segment IDs
            segment_id_to_name: Dict mapping segment ID to object name
        """
        H, W = rgb.shape[:2]

        detections = self.detect_objects(rgb, object_names)

        segmentation = np.zeros((H, W), dtype=np.int32)
        segment_id_to_name = {0: "background"}

        for i, det in enumerate(detections):
            seg_id = i + 1
            box = det['box_pixel']
            y_min, x_min, y_max, x_max = box

            # Clamp to image bounds
            y_min = max(0, y_min)
            x_min = max(0, x_min)
            y_max = min(H, y_max)
            x_max = min(W, x_max)

            # Fill bounding box region (only if not already filled)
            mask = segmentation[y_min:y_max, x_min:x_max] == 0
            segmentation[y_min:y_max, x_min:x_max][mask] = seg_id
            segment_id_to_name[seg_id] = det['label']

        return segmentation, segment_id_to_name

    def detect_points(
        self,
        rgb: np.ndarray,
        object_names: List[str],
    ) -> Dict[str, Tuple[int, int]]:
        """
        Detect objects and return center points.

        Args:
            rgb: RGB image (H, W, 3)
            object_names: List of object names to detect

        Returns:
            Dict mapping object name to (x, y) center point in pixels
        """
        detections = self.detect_objects(rgb, object_names)

        points = {}
        for det in detections:
            box = det['box_pixel']
            y_min, x_min, y_max, x_max = box
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            points[det['label']] = (cx, cy)

        return points


def create_gemini_grounder(config: dict) -> GeminiGrounder:
    """Factory function to create Gemini grounder."""
    return GeminiGrounder(config)


class GeminiStageRecognizer:
    """
    Use Gemini to recognize current stage and decide whether guidance is needed.
    """
    
    def __init__(self, config: dict):
        """
        Initialize Gemini stage recognizer.
        
        Args:
            config: Configuration dict with:
                - api_key: Google API key
                - model: Model name (default: gemini-2.5-flash)
                - stage_template_path: Path to prompt template (optional)
        """
        self.config = config
        self.api_key = config.get('api_key') or os.environ.get('GOOGLE_API_KEY')
        self.model = config.get('model', 'gemini-2.5-flash')
        
        if not self.api_key:
            raise ValueError("Gemini API key required. Set 'api_key' in config or GOOGLE_API_KEY env var.")
        
        # Load prompt template
        template_path = config.get('stage_template_path')
        if template_path is None:
            import pathlib
            project_root = pathlib.Path(__file__).parent.parent
            template_path = project_root / "vlm_query" / "stage_template.txt"
        
        with open(template_path, 'r') as f:
            self.prompt_template = f.read()
        
        self._setup_client()
        logger.info(f"GeminiStageRecognizer initialized with model: {self.model}")
    
    def _setup_client(self):
        """Setup Google Generative AI client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Install with:\n"
                "  pip install google-generativeai"
            )
    
    def identify_stage_and_guidance(
        self, 
        current_rgb: np.ndarray, 
        instruction: str, 
        stage_descriptions: str,
        init_img_with_keypoints: np.ndarray = None,
        keypoint_id_to_object: Dict[int, str] = None,
        num_stages: int = None,
        trigger_reason: str = None,
    ) -> Tuple[int, bool]:
        """
        Identify current stage and whether guidance is needed.
        
        Args:
            trigger_reason: Why this query was triggered (e.g., "gripper closed", "reward rose above 80%")
        
        Returns:
            tuple: (stage_num: int, need_guidance: bool)
        """
        # Ensure correct format for current image
        if current_rgb.dtype != np.uint8:
            if current_rgb.max() <= 1.0:
                current_rgb = (current_rgb * 255).astype(np.uint8)
            else:
                current_rgb = current_rgb.astype(np.uint8)
        
        if current_rgb.ndim == 4:
            current_rgb = current_rgb[0]
        
        current_img = Image.fromarray(current_rgb)
        
        # Prepare init image if provided
        init_img = None
        if init_img_with_keypoints is not None:
            if init_img_with_keypoints.dtype != np.uint8:
                if init_img_with_keypoints.max() <= 1.0:
                    init_img_with_keypoints = (init_img_with_keypoints * 255).astype(np.uint8)
                else:
                    init_img_with_keypoints = init_img_with_keypoints.astype(np.uint8)
            if init_img_with_keypoints.ndim == 4:
                init_img_with_keypoints = init_img_with_keypoints[0]
            init_img = Image.fromarray(init_img_with_keypoints)
        
        # Format keypoint mapping
        keypoint_info = "(No keypoint info)"
        if keypoint_id_to_object:
            keypoint_lines = [f"  Keypoint {k}: {v}" for k, v in keypoint_id_to_object.items()]
            keypoint_info = "\n".join(keypoint_lines)
        
        # Build prompt from template
        prompt = self.prompt_template.format(
            trigger_reason=trigger_reason or "initial query",
            instruction=instruction,
            stage_descriptions=stage_descriptions,
            keypoint_info=keypoint_info,
        )
        
        stage_num = 1
        need_guidance = True
        evidence = ""
        
        try:
            # Build content list with images
            content = [prompt]
            if init_img is not None:
                content.append("**Initial Image (with keypoints):**")
                content.append(init_img)
            content.append("**Current Image:**")
            content.append(current_img)
            
            response = self.client.generate_content(content)
            output = response.text.strip()
            
            # Parse stage number
            import re
            stage_match = re.search(r'stage\s+(\d+)', output.lower())
            if stage_match:
                stage_num = int(stage_match.group(1))
                if num_stages is not None and stage_num > 0:
                    stage_num = min(stage_num, num_stages)
            
            # Parse guidance decision
            guidance_match = re.search(r'guidance:\s*(yes|no)', output.lower())
            if guidance_match:
                need_guidance = (guidance_match.group(1) == 'yes')
            
            # Parse evidence
            evidence_match = re.search(r'evidence:\s*(.+)', output, re.IGNORECASE)
            if evidence_match:
                evidence = evidence_match.group(1).strip()
            
            logger.info(f"[Gemini] Stage:{stage_num} Guide:{'ON' if need_guidance else 'OFF'} | {evidence[:80]}")
            return stage_num, need_guidance
            
        except Exception as e:
            logger.error(f"Gemini stage recognition failed: {e}")
            return 1, True


def create_gemini_stage_recognizer(config: dict) -> GeminiStageRecognizer:
    """Factory function to create Gemini stage recognizer."""
    return GeminiStageRecognizer(config)
