from pose_anonymization.data.normalization import normalize_mean_std
from pose_format import Pose
from pose_format.utils.generic import pose_hide_legs, reduce_holistic

BIO = {
    "UNK": 0,
    "O": 1,
    "B": 2,
    "I": 3,
}

def preprocess_pose(pose: Pose, normalize: bool = True, no_face: bool = False):
    components = ["POSE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"]
    if not no_face:
        components.append("FACE_LANDMARKS")
    pose = pose.get_components(components)
    pose_hide_legs(pose)
    pose = reduce_holistic(pose)
    if normalize:
        pose = normalize_mean_std(pose)
    return pose
