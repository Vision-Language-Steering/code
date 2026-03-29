#!/usr/bin/env python3
"""
SAM Segment Refiner - 使用 SAM 模型细化分割结果

将 PyBullet 的 link 级别分割细化为更精细的部件级别分割。
例如：抽屉的把手和面板可以被分开识别。
"""

import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional
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


def download_sam_model(model_type: str, save_dir: str = None) -> str:
    """
    下载 SAM 模型权重。
    
    Args:
        model_type: 模型类型 (vit_h, vit_l, vit_b)
        save_dir: 保存目录，默认为 Grounded-SAM 目录
        
    Returns:
        模型文件路径
    """
    import urllib.request
    
    if model_type not in SAM_MODEL_URLS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(SAM_MODEL_URLS.keys())}")
    
    url = SAM_MODEL_URLS[model_type]
    filename = url.split("/")[-1]
    
    if save_dir is None:
        save_dir = GSAM_PATH
    
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        print(f"[SAM] Model already exists: {save_path}")
        return save_path
    
    print(f"[SAM] Downloading {model_type} model from {url}...")
    print(f"[SAM] This may take a while (~2.4GB for vit_h)...")
    
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"[SAM] Model saved to: {save_path}")
        return save_path
    except Exception as e:
        print(f"[SAM] Download failed: {e}")
        print(f"[SAM] Please manually download from: {url}")
        print(f"[SAM] And save to: {save_path}")
        raise


@dataclass
class RefinedSegment:
    """细化后的分割区域"""
    segment_id: int           # 新的 segment ID
    parent_link_id: int       # 原始 PyBullet link ID
    parent_link_name: str     # 原始 link 名称
    mask: np.ndarray          # 二值 mask
    area: int                 # 像素面积
    sub_part_index: int       # 在同一 link 内的子部件索引


class SAMSegmentRefiner:
    """
    使用 SAM 模型细化 PyBullet 分割结果。
    
    工作原理：
    1. 使用 SAM 自动分割整个图像
    2. 将 SAM 的细粒度 mask 与 PyBullet 的语义 mask 进行交集
    3. 每个 SAM mask 继承覆盖最多的 PyBullet link 的语义
    
    使用示例：
        refiner = SAMSegmentRefiner(device="cuda")
        refined_seg, new_seg_id_to_name = refiner.refine_segmentation(
            rgb_image, 
            pybullet_segmentation, 
            segment_id_to_name
        )
    """
    
    def __init__(
        self,
        sam_checkpoint: str = None,
        sam_model_type: str = "vit_h",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        min_overlap_ratio: float = 0.3,  # 最小重叠比例，低于此值的 SAM mask 会被过滤
    ):
        """
        初始化 SAM 分割细化器。
        
        Args:
            sam_checkpoint: SAM 模型权重路径
            sam_model_type: SAM 模型类型 (vit_h, vit_l, vit_b)
            device: 计算设备
            points_per_side: SAM 采样点密度
            pred_iou_thresh: IoU 预测阈值
            stability_score_thresh: 稳定性分数阈值
            min_mask_region_area: 最小 mask 区域面积
            min_overlap_ratio: 与 PyBullet mask 的最小重叠比例
        """
        self.device = torch.device(device)
        self.min_overlap_ratio = min_overlap_ratio
        
        # 默认 SAM checkpoint 路径
        if sam_checkpoint is None:
            # 尝试找到已存在的模型，或自动下载
            expected_filenames = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth",
            }
            sam_checkpoint = os.path.join(GSAM_PATH, expected_filenames[sam_model_type])
            
            if not os.path.exists(sam_checkpoint):
                print(f"[SAMSegmentRefiner] SAM model not found at {sam_checkpoint}")
                sam_checkpoint = download_sam_model(sam_model_type, GSAM_PATH)
        
        # 加载 SAM 模型
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        
        print(f"[SAMSegmentRefiner] Loading SAM model ({sam_model_type}) from {sam_checkpoint}")
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        sam.eval()
        
        # 创建自动 mask 生成器
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            output_mode="binary_mask",
        )
        
        print(f"[SAMSegmentRefiner] Initialized on device: {self.device}")
    
    def generate_sam_masks(self, rgb: np.ndarray) -> List[Dict]:
        """
        使用 SAM 自动分割生成 masks。
        
        Args:
            rgb: RGB 图像 (H, W, 3), uint8
            
        Returns:
            SAM masks 列表，每个元素包含 'segmentation', 'area', 'bbox' 等
        """
        # SAM 期望 RGB 格式
        masks = self.mask_generator.generate(rgb)
        
        # 按面积降序排序（大 mask 在前）
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def refine_segmentation(
        self,
        rgb: np.ndarray,
        pybullet_seg: np.ndarray,
        segment_id_to_name: Dict[int, str],
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        使用 SAM 细化 PyBullet 分割结果。
        
        Args:
            rgb: RGB 图像 (H, W, 3), uint8
            pybullet_seg: PyBullet 分割图像 (H, W), int
            segment_id_to_name: PyBullet segment ID 到名称的映射
            
        Returns:
            refined_seg: 细化后的分割图像 (H, W), int
            new_segment_id_to_name: 新的 segment ID 到名称的映射
        """
        H, W = pybullet_seg.shape
        
        # Step 1: 生成 SAM masks
        sam_masks = self.generate_sam_masks(rgb)
        print(f"[SAMSegmentRefiner] Generated {len(sam_masks)} SAM masks")
        
        if len(sam_masks) == 0:
            # 没有 SAM mask，返回原始分割
            return pybullet_seg, segment_id_to_name
        
        # Step 2: 获取所有可交互的 PyBullet segment IDs
        interactable_ids = [seg_id for seg_id in segment_id_to_name.keys() if seg_id > 0]
        
        if len(interactable_ids) == 0:
            return pybullet_seg, segment_id_to_name
        
        # Step 3: 对每个 SAM mask，计算与 PyBullet masks 的重叠
        refined_segments: List[RefinedSegment] = []
        new_segment_id = 1
        
        # 跟踪每个 PyBullet link 产生了多少个子部件
        link_sub_part_count: Dict[int, int] = {seg_id: 0 for seg_id in interactable_ids}
        
        for sam_mask_data in sam_masks:
            sam_mask = sam_mask_data['segmentation']  # (H, W) bool
            sam_area = sam_mask_data['area']
            
            # 计算与每个 PyBullet segment 的重叠
            best_overlap_ratio = 0
            best_link_id = None
            
            for link_id in interactable_ids:
                pybullet_mask = pybullet_seg == link_id
                
                # 计算交集
                intersection = np.logical_and(sam_mask, pybullet_mask)
                intersection_area = np.sum(intersection)
                
                # 重叠比例 = 交集面积 / SAM mask 面积
                overlap_ratio = intersection_area / sam_area if sam_area > 0 else 0
                
                if overlap_ratio > best_overlap_ratio:
                    best_overlap_ratio = overlap_ratio
                    best_link_id = link_id
            
            # 只保留与可交互对象有足够重叠的 SAM mask
            if best_overlap_ratio >= self.min_overlap_ratio and best_link_id is not None:
                # 裁剪 SAM mask 到 PyBullet mask 范围内
                pybullet_mask = pybullet_seg == best_link_id
                refined_mask = np.logical_and(sam_mask, pybullet_mask)
                
                if np.sum(refined_mask) > 50:  # 过滤太小的区域
                    link_sub_part_count[best_link_id] += 1
                    
                    refined_segments.append(RefinedSegment(
                        segment_id=new_segment_id,
                        parent_link_id=best_link_id,
                        parent_link_name=segment_id_to_name[best_link_id],
                        mask=refined_mask,
                        area=np.sum(refined_mask),
                        sub_part_index=link_sub_part_count[best_link_id]
                    ))
                    new_segment_id += 1
        
        # Step 4: 生成细化后的分割图像
        refined_seg = np.zeros((H, W), dtype=np.int32)
        new_segment_id_to_name = {0: "background"}
        
        # 按面积降序处理，确保小的 mask 覆盖大的
        refined_segments = sorted(refined_segments, key=lambda x: x.area, reverse=True)
        
        for seg in refined_segments:
            refined_seg[seg.mask] = seg.segment_id
            
            # 生成名称：如果有多个子部件，添加后缀
            if link_sub_part_count[seg.parent_link_id] > 1:
                name = f"{seg.parent_link_name}_part{seg.sub_part_index}"
            else:
                name = seg.parent_link_name
            
            new_segment_id_to_name[seg.segment_id] = name
        
        # Step 5: 处理未被 SAM 覆盖的区域
        # 对于 PyBullet 分割中有，但 SAM 没有覆盖的区域，保留原始分割
        for link_id in interactable_ids:
            pybullet_mask = pybullet_seg == link_id
            not_covered = np.logical_and(pybullet_mask, refined_seg == 0)
            
            if np.sum(not_covered) > 50:
                # 添加未覆盖区域作为新的 segment
                seg_id = max(new_segment_id_to_name.keys()) + 1
                refined_seg[not_covered] = seg_id
                new_segment_id_to_name[seg_id] = f"{segment_id_to_name[link_id]}_uncovered"
        
        print(f"[SAMSegmentRefiner] Refined to {len(new_segment_id_to_name) - 1} segments")
        for seg_id, name in new_segment_id_to_name.items():
            if seg_id > 0:
                pixel_count = np.sum(refined_seg == seg_id)
                print(f"  [{seg_id}] {name}: {pixel_count} pixels")
        
        return refined_seg, new_segment_id_to_name
    
    def refine_with_prompts(
        self,
        rgb: np.ndarray,
        pybullet_seg: np.ndarray,
        segment_id_to_name: Dict[int, str],
        prompts: Optional[Dict[int, List[Tuple[int, int]]]] = None,
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        使用点提示进行细化分割（更精确但需要提供提示点）。
        
        Args:
            rgb: RGB 图像
            pybullet_seg: PyBullet 分割
            segment_id_to_name: segment ID 到名称的映射
            prompts: 可选的点提示，格式 {link_id: [(x1, y1), (x2, y2), ...]}
                    提示点会被用来引导 SAM 分割
        
        Returns:
            refined_seg, new_segment_id_to_name
        """
        # 如果没有提示，使用自动分割
        if prompts is None:
            return self.refine_segmentation(rgb, pybullet_seg, segment_id_to_name)
        
        # TODO: 实现基于点提示的精确分割
        # 这可以用于需要更精确控制的场景
        raise NotImplementedError("Point-prompted refinement not yet implemented")
    
    def visualize_refinement(
        self,
        rgb: np.ndarray,
        original_seg: np.ndarray,
        refined_seg: np.ndarray,
        original_names: Dict[int, str],
        refined_names: Dict[int, str],
        save_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        可视化分割细化结果对比。
        
        Args:
            rgb: 原始 RGB 图像
            original_seg: 原始分割
            refined_seg: 细化后的分割
            original_names: 原始名称映射
            refined_names: 细化后的名称映射
            save_path: 保存路径
            
        Returns:
            可视化图像
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        H, W = rgb.shape[:2]
        
        # 创建随机颜色映射
        np.random.seed(42)
        max_id = max(max(original_seg.flatten()), max(refined_seg.flatten())) + 1
        colors = np.random.rand(max_id, 3)
        colors[0] = [0, 0, 0]  # 背景为黑色
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(rgb)
        axes[0].set_title("Original RGB")
        axes[0].axis('off')
        
        # 原始分割
        original_vis = np.zeros((H, W, 3))
        for seg_id in np.unique(original_seg):
            if seg_id > 0:
                mask = original_seg == seg_id
                original_vis[mask] = colors[seg_id % max_id]
        original_vis = (original_vis * 0.5 + rgb / 255.0 * 0.5)
        axes[1].imshow(original_vis)
        axes[1].set_title(f"Original Seg ({len(original_names)-1} objects)")
        axes[1].axis('off')
        
        # 细化分割
        refined_vis = np.zeros((H, W, 3))
        for seg_id in np.unique(refined_seg):
            if seg_id > 0:
                mask = refined_seg == seg_id
                refined_vis[mask] = colors[seg_id % max_id]
        refined_vis = (refined_vis * 0.5 + rgb / 255.0 * 0.5)
        axes[2].imshow(refined_vis)
        axes[2].set_title(f"SAM Refined ({len(refined_names)-1} parts)")
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[SAMSegmentRefiner] Saved visualization to {save_path}")
        
        # 转换为 numpy 图像 (兼容新版 matplotlib)
        fig.canvas.draw()
        # 使用 buffer_rgba() 替代已弃用的 tostring_rgb()
        buf = fig.canvas.buffer_rgba()
        vis_img = np.asarray(buf)[:, :, :3]  # RGBA -> RGB
        plt.close(fig)
        
        return vis_img.copy()

