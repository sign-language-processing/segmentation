import math

import numpy as np

BIO: dict[str, int] = {
    "UNK": 0,
    "O": 1,
    "B": 2,
    "I": 3,
}


def create_bio(annotations: list[dict[str, float]], num_frames: int, fps: float) -> np.ndarray:
    """create BIO labels from annotation spans assuming uniform frame spacing.

    annotations: list of {start, end} in milliseconds.
    """
    bio = np.full((num_frames,), fill_value=BIO["O"], dtype=np.uint8)
    for a in annotations:
        start_frame = min(math.floor(a["start"] * fps / 1000), num_frames - 1)
        end_frame = min(math.ceil(a["end"] * fps / 1000), num_frames - 1)
        bio[start_frame] = BIO["B"]
        bio[start_frame + 1:end_frame + 1] = BIO["I"]
    return bio


def create_bio_from_times(annotations: list[dict[str, float]], frame_times_ms: np.ndarray) -> np.ndarray:
    """create BIO labels using actual frame timestamps in ms — handles non-uniform frame spacing."""
    num_frames = len(frame_times_ms)
    bio = np.full(num_frames, fill_value=BIO["O"], dtype=np.uint8)
    for a in annotations:
        start_idx = int(np.searchsorted(frame_times_ms, a["start"], side="left"))
        end_idx = int(np.searchsorted(frame_times_ms, a["end"], side="right")) - 1
        start_idx = min(start_idx, num_frames - 1)
        end_idx = min(end_idx, num_frames - 1)
        if start_idx <= end_idx:
            bio[start_idx] = BIO["B"]
            if start_idx + 1 <= end_idx:
                bio[start_idx + 1:end_idx + 1] = BIO["I"]
    return bio
