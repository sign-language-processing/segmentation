import importlib
from typing import Dict, List, TypedDict, Union

import tensorflow_datasets as tfds
from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus import DgsCorpusConfig
from tqdm import tqdm
import mediapipe as mp

from .pose_utils import pose_hide_legs, pose_normalization_info


mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]

class ProcessedPoseDatum(TypedDict):
    id: str
    pose: Union[Pose, Dict[str, Pose]]
    tf_datum: dict


def get_tfds_dataset(name,
                     poses="holistic",
                     fps=25,
                     split="train",
                     components: List[str] = None,
                     reduce_face=False,
                     data_dir=None,
                     version="1.0.0",
                     filter_func=None):
    dataset_module = importlib.import_module("sign_language_datasets.datasets." + name + "." + name)

    config_kwargs = dict(
        name=poses + "-" + str(fps),
        version=version,  # Specific version
        include_video=False,  # Download and load dataset videos
        fps=fps,  # Load videos at constant fps
        include_pose=poses)

    # Loading a dataset with custom configuration
    if name == "dgs_corpus":
        config = DgsCorpusConfig(**config_kwargs, split="3.0.0-uzh-document")
    else:
        config = SignDatasetConfig(**config_kwargs)

    tfds_dataset = tfds.load(name=name, builder_kwargs=dict(config=config), split=split, data_dir=data_dir)

    # pylint: disable=protected-access
    with open(dataset_module._POSE_HEADERS[poses], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    normalization_info = pose_normalization_info(pose_header)
    return [process_datum(datum, pose_header, normalization_info, components, reduce_face)
            for datum in tqdm(tfds_dataset, desc="Loading dataset")
            if filter_func is None or filter_func(datum)]


def process_datum(datum,
                  pose_header: PoseHeader,
                  normalization_info,
                  components: List[str] = None,
                  reduce_face=False) -> ProcessedPoseDatum:
    tf_poses = {"": datum["pose"]} if "pose" in datum else datum["poses"]
    poses = {}
    for key, tf_pose in tf_poses.items():
        fps = int(tf_pose["fps"].numpy())
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        pose = Pose(pose_header, pose_body)

        # Get subset of components if needed
        if reduce_face:
            pose = pose.get_components(components, {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})
        elif components and len(components) != len(pose_header.components):
            pose = pose.get_components(components)

        pose = pose.normalize(normalization_info)
        pose_hide_legs(pose)
        poses[key] = pose

    return {
        "id": datum["id"].numpy().decode('utf-8'),
        "pose": poses[""] if "pose" in datum else poses,
        "tf_datum": datum
    }