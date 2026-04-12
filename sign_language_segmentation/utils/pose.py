import numpy as np
from pose_anonymization.data.normalization import normalize_mean_std
from pose_format import Pose
from pose_format.utils.generic import pose_hide_legs, reduce_holistic


def preprocess_pose(pose: Pose) -> Pose:
    """normalise and strip face landmarks from a holistic pose."""
    pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])
    pose_hide_legs(pose)
    pose = reduce_holistic(pose)
    return normalize_mean_std(pose)


def compute_velocity(pose_data: np.ndarray, frame_times_seconds: np.ndarray) -> np.ndarray:
    """compute fps-normalised velocity (units/second).

    frame_times_seconds: (T,) timestamps in seconds.
    Returns: (T, joints, dims) velocity array, zero at the first frame.
    """
    if len(pose_data) <= 1:
        return np.zeros_like(pose_data)
    dt = np.diff(frame_times_seconds)  # (T-1,) seconds
    vel_inner = np.diff(pose_data, axis=0) / dt[:, None, None]
    return np.concatenate([np.zeros_like(pose_data[:1]), vel_inner], axis=0)
