from pose_anonymization.data.normalization import normalize_mean_std
from pose_format import Pose
from pose_format.utils.generic import pose_hide_legs, reduce_holistic

BIO = {
    "UNK": 0,
    "O": 1,
    "B": 2,
    "I": 3,
}

def preprocess_pose(pose: Pose):
    pose = pose.get_components(["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS", "FACE_LANDMARKS"])
    pose_hide_legs(pose)
    pose = reduce_holistic(pose)
    return normalize_mean_std(pose)
