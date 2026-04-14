from sign_language_segmentation.datasets.common import register_dataset
from sign_language_segmentation.datasets.dgs.dataset import DGSSegmentationDataset

register_dataset("dgs", DGSSegmentationDataset)

__all__ = ["DGSSegmentationDataset"]
