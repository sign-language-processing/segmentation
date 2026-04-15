from sign_language_segmentation.datasets.common import register_dataset
from sign_language_segmentation.datasets.signtube.dataset import SignTubeSegmentationDataset

register_dataset("signtube", SignTubeSegmentationDataset)

__all__ = ["SignTubeSegmentationDataset"]
