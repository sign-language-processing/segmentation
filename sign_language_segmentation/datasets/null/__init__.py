from sign_language_segmentation.datasets.common import register_dataset
from sign_language_segmentation.datasets.null.dataset import NullSegmentationDataset

register_dataset("null", NullSegmentationDataset)

__all__ = ["NullSegmentationDataset"]
