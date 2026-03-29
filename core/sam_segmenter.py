#!/usr/bin/env python3
"""
SAM Segmenter - 独立的 SAM 分割组件

提供多种分割方式：
1. 自动分割 (segment_auto)
2. 点提示分割 (segment_with_points)
3. 框提示分割 (segment_with_boxes)
4. 文本提示分割 (segment_with_text) - 使用 Grounded-SAM

Usage:
    segmenter = SAMSegmenter(device="cuda")
    
    # 自动分割
    masks = segmenter.segment_auto(rgb)
    
    # 文本提示分割 (Grounded-SAM)
    masks = segmenter.segment_with_text(rgb, ["drawer handle", "red block"])
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import sys
import os

# 添加 Grounded-SAM 路径
GSAM_PATH = os.path.join(os.path.dirname(__file__), "..", "moka", "Grounded-Segment-Anything")
sys.path.insert(0, os.path.join(GSAM_PATH, "segment_anything"))

# SAM model download URLs
SAM_MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


@dataclass
class SegmentResult:
    """分割结果"""
    mask: np.ndarray          # 二值 mask (H, W)
    label: str                # 标签/名称
    confidence: float         # 置信度
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    area: int = 0             # 像素面积


def download_sam_model(model_type: str, save_dir: str = None) -> str:
    """下载 SAM 模型权重"""
    import urllib.request
    
    if model_type not in SAM_MODEL_URLS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    url = SAM_MODEL_URLS[model_type]
    filename = url.split("/")[-1]
    
    if save_dir is None:
        save_dir = GSAM_PATH
    
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        return save_path
    
    print(f"[SAM] Downloading {model_type} model...")
    urllib.request.urlretrieve(url, save_path)
    print(f"[SAM] Model saved to: {save_path}")
    return save_path


class SAMSegmenter:
    """
    独立的 SAM 分割组件。
    
    支持多种分割方式，可以独立使用或与 VLM 配合。
    """
    
    def __init__(
        self,
        sam_checkpoint: str = None,
        sam_model_type: str = "vit_h",
        device: str = "cuda",
        # 自动分割参数
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        # Grounded-SAM 参数
        grounding_dino_config: str = None,
        grounding_dino_checkpoint: str = None,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        """
        初始化 SAM 分割器。
        
        Args:
            sam_checkpoint: SAM 模型权重路径
            sam_model_type: SAM 模型类型 (vit_h, vit_l, vit_b)
            device: 计算设备
            points_per_side: 自动分割的采样点密度
            grounding_dino_config: Grounding DINO 配置路径
            grounding_dino_checkpoint: Grounding DINO 权重路径
            box_threshold: 检测框置信度阈值
            text_threshold: 文本匹配阈值
        """
        self.device = torch.device(device)
        self.sam_model_type = sam_model_type
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        
        # 加载 SAM 模型
        if sam_checkpoint is None:
            expected_filenames = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth",
            }
            sam_checkpoint = os.path.join(GSAM_PATH, expected_filenames[sam_model_type])
            
            if not os.path.exists(sam_checkpoint):
                sam_checkpoint = download_sam_model(sam_model_type, GSAM_PATH)
        
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        
        print(f"[SAMSegmenter] Loading SAM model ({sam_model_type})...")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        sam.eval()
        
        self.sam = sam
        self.sam_predictor = SamPredictor(sam)
        
        # 创建自动 mask 生成器
        self.auto_mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            output_mode="binary_mask",
        )
        
        # Grounding DINO (延迟加载)
        self.grounding_dino = None
        self.grounding_dino_config = grounding_dino_config or os.path.join(
            GSAM_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        )
        self.grounding_dino_checkpoint = grounding_dino_checkpoint or os.path.join(
            GSAM_PATH, "groundingdino_swint_ogc.pth"
        )
        
        print(f"[SAMSegmenter] Initialized on device: {self.device}")
    
    def _load_grounding_dino(self):
        """延迟加载 Grounding DINO"""
        if self.grounding_dino is not None:
            return
        
        try:
            sys.path.insert(0, os.path.join(GSAM_PATH, "GroundingDINO"))
            from groundingdino.util.inference import Model
            
            if not os.path.exists(self.grounding_dino_checkpoint):
                print(f"[SAMSegmenter] Grounding DINO checkpoint not found: {self.grounding_dino_checkpoint}")
                print("[SAMSegmenter] Please download from: https://github.com/IDEA-Research/GroundingDINO")
                raise FileNotFoundError(f"Grounding DINO checkpoint not found")
            
            print("[SAMSegmenter] Loading Grounding DINO...")
            self.grounding_dino = Model(
                model_config_path=self.grounding_dino_config,
                model_checkpoint_path=self.grounding_dino_checkpoint
            )
            print("[SAMSegmenter] Grounding DINO loaded")
            
        except Exception as e:
            print(f"[SAMSegmenter] Failed to load Grounding DINO: {e}")
            raise
    
    def segment_auto(
        self, 
        rgb: np.ndarray,
        min_area: int = 100,
        max_area_ratio: float = 0.9,
    ) -> List[SegmentResult]:
        """
        自动分割整个图像。
        
        Args:
            rgb: RGB 图像 (H, W, 3), uint8
            min_area: 最小 mask 面积
            max_area_ratio: 最大 mask 面积比例
            
        Returns:
            分割结果列表
        """
        masks_data = self.auto_mask_generator.generate(rgb)
        
        H, W = rgb.shape[:2]
        max_area = H * W * max_area_ratio
        
        results = []
        for i, mask_data in enumerate(masks_data):
            mask = mask_data['segmentation']
            area = mask_data['area']
            
            if area < min_area or area > max_area:
                continue
            
            bbox = mask_data['bbox']  # [x, y, w, h]
            bbox_xyxy = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            
            results.append(SegmentResult(
                mask=mask,
                label=f"segment_{i}",
                confidence=mask_data['predicted_iou'],
                bbox=bbox_xyxy,
                area=area,
            ))
        
        # 按面积降序排序
        results.sort(key=lambda x: x.area, reverse=True)
        
        return results
    
    def segment_with_points(
        self,
        rgb: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray = None,
    ) -> List[SegmentResult]:
        """
        使用点提示分割。
        
        Args:
            rgb: RGB 图像 (H, W, 3), uint8
            points: 点坐标 (N, 2) as [x, y]
            labels: 点标签 (N,), 1=前景, 0=背景
            
        Returns:
            分割结果列表
        """
        self.sam_predictor.set_image(rgb)
        
        if labels is None:
            labels = np.ones(len(points), dtype=np.int32)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        results = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            results.append(SegmentResult(
                mask=mask,
                label=f"point_segment_{i}",
                confidence=float(score),
                area=int(np.sum(mask)),
            ))
        
        return results
    
    def segment_with_boxes(
        self,
        rgb: np.ndarray,
        boxes: np.ndarray,
        box_labels: List[str] = None,
    ) -> List[SegmentResult]:
        """
        使用框提示分割。
        
        Args:
            rgb: RGB 图像 (H, W, 3), uint8
            boxes: 边界框 (N, 4) as [x1, y1, x2, y2]
            box_labels: 每个框的标签
            
        Returns:
            分割结果列表
        """
        self.sam_predictor.set_image(rgb)
        
        results = []
        for i, box in enumerate(boxes):
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True,
            )
            
            # 选择最高分的 mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            score = scores[best_idx]
            
            label = box_labels[i] if box_labels else f"box_segment_{i}"
            
            results.append(SegmentResult(
                mask=mask,
                label=label,
                confidence=float(score),
                bbox=tuple(box),
                area=int(np.sum(mask)),
            ))
        
        return results
    
    def segment_with_text(
        self,
        rgb: np.ndarray,
        text_prompts: List[str],
        box_threshold: float = None,
        text_threshold: float = None,
    ) -> List[SegmentResult]:
        """
        使用文本提示分割 (Grounded-SAM)。
        
        这是与 VLM 配合的核心方法：VLM 输出对象描述，SAM 生成对应的 mask。
        
        Args:
            rgb: RGB 图像 (H, W, 3), uint8
            text_prompts: 文本描述列表，如 ["drawer handle", "red block"]
            box_threshold: 检测框置信度阈值
            text_threshold: 文本匹配阈值
            
        Returns:
            分割结果列表
        """
        self._load_grounding_dino()
        
        box_threshold = box_threshold or self.box_threshold
        text_threshold = text_threshold or self.text_threshold
        
        # 合并文本提示为单个 prompt
        prompt = ". ".join(text_prompts) + "."
        
        # 使用 Grounding DINO 检测
        # 注意：Grounding DINO 期望 BGR 格式
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        detections = self.grounding_dino.predict_with_classes(
            image=rgb_bgr,
            classes=text_prompts,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        
        if len(detections.xyxy) == 0:
            print(f"[SAMSegmenter] No objects detected for prompts: {text_prompts}")
            return []
        
        # 使用 SAM 对每个检测框进行分割
        self.sam_predictor.set_image(rgb)
        
        results = []
        for i, (box, confidence, class_id) in enumerate(zip(
            detections.xyxy, 
            detections.confidence,
            detections.class_id
        )):
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=True,
            )
            
            # 选择最高分的 mask
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            label = text_prompts[class_id] if class_id < len(text_prompts) else f"object_{i}"
            
            results.append(SegmentResult(
                mask=mask,
                label=label,
                confidence=float(confidence),
                bbox=tuple(map(int, box)),
                area=int(np.sum(mask)),
            ))
        
        return results
    
    def create_segmentation_image(
        self,
        rgb: np.ndarray,
        segments: List[SegmentResult],
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        从分割结果创建分割图像。
        
        Args:
            rgb: 原始 RGB 图像
            segments: 分割结果列表
            
        Returns:
            segmentation: 分割图像 (H, W), int
            segment_id_to_name: segment ID 到名称的映射
        """
        H, W = rgb.shape[:2]
        segmentation = np.zeros((H, W), dtype=np.int32)
        segment_id_to_name = {0: "background"}
        
        # 按面积降序排序，大的先放（小的会覆盖大的）
        segments_sorted = sorted(segments, key=lambda x: x.area, reverse=True)
        
        for i, seg in enumerate(segments_sorted):
            seg_id = i + 1
            segmentation[seg.mask] = seg_id
            segment_id_to_name[seg_id] = seg.label
        
        return segmentation, segment_id_to_name
    
    def visualize_segments(
        self,
        rgb: np.ndarray,
        segments: List[SegmentResult],
        save_path: str = None,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        可视化分割结果。
        
        Args:
            rgb: 原始 RGB 图像
            segments: 分割结果列表
            save_path: 保存路径
            alpha: 透明度
            
        Returns:
            可视化图像
        """
        output = rgb.copy()
        
        np.random.seed(42)
        colors = [np.random.randint(50, 255, 3).tolist() for _ in range(len(segments))]
        
        for seg, color in zip(segments, colors):
            mask = seg.mask
            output[mask] = (
                alpha * np.array(color) + 
                (1 - alpha) * output[mask]
            ).astype(np.uint8)
            
            # 绘制边界框和标签
            if seg.bbox:
                x1, y1, x2, y2 = seg.bbox
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{seg.label} ({seg.confidence:.2f})"
                cv2.putText(
                    output, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                )
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            print(f"[SAMSegmenter] Saved visualization to: {save_path}")
        
        return output


# ==================== VLM + SAM 协作接口 ====================

class VLMSAMPipeline:
    """
    VLM + SAM 协作流水线。
    
    使用 VLM 理解任务，SAM 生成任务相关对象的 mask。
    
    Usage:
        pipeline = VLMSAMPipeline(vlm_agent, sam_segmenter)
        segments = pipeline.segment_task_objects(rgb, "pick up the red block and place it in the drawer")
    """
    
    def __init__(
        self,
        vlm_agent,  # VLMAgent 实例
        sam_segmenter: SAMSegmenter,
    ):
        self.vlm = vlm_agent
        self.sam = sam_segmenter
    
    def identify_task_objects(
        self,
        rgb: np.ndarray,
        task_instruction: str,
    ) -> List[str]:
        """
        使用 VLM 识别任务相关对象。
        
        Args:
            rgb: RGB 图像
            task_instruction: 任务指令
            
        Returns:
            任务相关对象描述列表
        """
        # 构建 prompt
        prompt = f"""Given the task instruction: "{task_instruction}"

Please identify all objects in the image that are relevant to this task.
Return a JSON list of object descriptions that can be used for object detection.

Example format:
["drawer handle", "red cube", "table surface"]

Focus on:
1. Objects that need to be manipulated
2. Target locations
3. Reference objects for spatial relationships

Be specific and use visual descriptors (color, shape, material) when helpful.
"""
        
        # 调用 VLM
        response = self.vlm.query_with_image(rgb, prompt)
        
        # 解析响应
        import json
        try:
            # 尝试从响应中提取 JSON 列表
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                objects = json.loads(json_match.group())
                return objects
        except:
            pass
        
        # 如果解析失败，返回空列表
        print(f"[VLMSAMPipeline] Failed to parse VLM response: {response}")
        return []
    
    def segment_task_objects(
        self,
        rgb: np.ndarray,
        task_instruction: str,
    ) -> Tuple[List[SegmentResult], List[str]]:
        """
        分割任务相关对象。
        
        Args:
            rgb: RGB 图像
            task_instruction: 任务指令
            
        Returns:
            segments: 分割结果列表
            object_names: 识别的对象名称列表
        """
        # Step 1: VLM 识别任务相关对象
        object_names = self.identify_task_objects(rgb, task_instruction)
        
        if not object_names:
            print("[VLMSAMPipeline] No task objects identified, using auto segmentation")
            segments = self.sam.segment_auto(rgb)
            return segments, []
        
        print(f"[VLMSAMPipeline] Identified task objects: {object_names}")
        
        # Step 2: SAM 生成 mask
        try:
            segments = self.sam.segment_with_text(rgb, object_names)
        except Exception as e:
            print(f"[VLMSAMPipeline] Grounded-SAM failed: {e}")
            print("[VLMSAMPipeline] Falling back to auto segmentation")
            segments = self.sam.segment_auto(rgb)
        
        return segments, object_names
    
    def get_keypoint_detection_inputs(
        self,
        rgb: np.ndarray,
        task_instruction: str,
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        获取关键点检测所需的输入（分割图像）。
        
        Args:
            rgb: RGB 图像
            task_instruction: 任务指令
            
        Returns:
            segmentation: 分割图像 (H, W)
            segment_id_to_name: segment ID 到名称的映射
        """
        segments, _ = self.segment_task_objects(rgb, task_instruction)
        segmentation, segment_id_to_name = self.sam.create_segmentation_image(rgb, segments)
        
        return segmentation, segment_id_to_name

