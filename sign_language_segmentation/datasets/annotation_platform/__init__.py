from sign_language_segmentation.datasets.common import register_dataset
from sign_language_segmentation.datasets.annotation_platform.dataset import (
    AnnotationPlatformSegmentationDataset,
)

register_dataset("platform", AnnotationPlatformSegmentationDataset)

__all__ = ["AnnotationPlatformSegmentationDataset"]
