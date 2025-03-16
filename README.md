# Sign Language Segmentation

Pose segmentation model on both the sentence and sign level

Code for the paper [Linguistically Motivated Sign Language Segmentation](https://aclanthology.org/2023.findings-emnlp.846).

## Usage


```bash
pip install git+https://github.com/sign-language-processing/segmentation
```

To create an ELAN file with sign and sentence segments:
(To demo this on a longer file, you can download a large pose file from [here](https://firebasestorage.googleapis.com/v0/b/sign-language-datasets/o/poses%2Fholistic%2Fdgs_corpus%2F1413451-11105600-11163240_a.pose?alt=media&token=432f0b57-3fb9-45ad-a9a4-0b6fae4ffcf7))

```bash
pose_to_segments --pose="sign.pose" --elan="sign.eaf" [--video="sign.mp4"]
```

---

## Main Idea

We tag pose sequences with BIO (beginning/in/out) and try to classify each frame. 
Due to huge sequence sizes intended to work on (full videos), this is not done using a transformer.
Loss is heavily weighted in favor of "B" as it is a "rare" prediction compared to I and O.


#### Pseudo code:

```python
pose_embedding = embed_pose(pose)
pose_encoding = encoder(pose_embedding)
sign_bio = sign_bio_tagger(pose_encoding)
sentence_bio = sentence_bio_tagger(pose_encoding)
```

## Extra details

- Model tests, including overfitting, and continuous integration
- We remove the legs because they are not informative
- For experiment management we use WANDB
- Training works on CPU and GPU (90% util)
- Multiple-GPUs not tested

## Motivation

### Optical flow 
Optical flow is highly correlative to phrase boundaries. 

![Optical flow](sign_language_segmentation/figures/optical_fow/optical_flow_sentence_example.png)

### 3D Hand Normalization
3D hand normalization may assist the model with learning hand shape changes.

Watch [this video](https://youtu.be/pCKRWSNIaNQ?t=191) to see how it's done.

## Reproducing Experiments

### E0: Moryossef et al. (2020)
This is an attempt to reproduce the methodology of Moryossef et al. (2020) on the DGS corpus.
Since they used a different document split, and do not filter out wrong data, our results are not directly comparable. This model processes optical flow as input and outputs I (is signing) and O (not signing) tags.

```bash
python -m sign_language_segmentation.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=64 --encoder_depth=1 --encoder_bidirectional=false --optical_flow=true --only_optical_flow=true --weighted_loss=false --classes=io
```

### E1: Bidirectional BIO Tagger
We replace the IO tagging heads in E0 with BIO heads to form our baseline. Our preliminary experiments indicate that inputting only the 75 hand and body keypoints and making the LSTM layer bidirectional yields optimal results.
```bash
conda activate segmentation
export CUDA_VISIBLE_DEVICES=3
python -m sign_language_segmentation.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --no_wandb true
```
Or for the mediapi-skel dataset (only phrase segmentation)
```bash
# FPS is not relevant for mediapi-skel
export MEDIAPI_PATH=/shares/volk.cl.uzh/amoryo/datasets/mediapi/mediapi-skel.zip
export MEDIAPI_POSE_PATH=/shares/volk.cl.uzh/amoryo/datasets/mediapi/mediapipe_zips.zip
python -m sign_language_segmentation.src.train --dataset=mediapi_skel --pose=holistic --fps=0 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true
```

### E2: Adding Reduced Face Keypoints

Although the 75 hand and body keypoints serve as an efficient minimal set for sign language detection/segmentation models, we investigate the impact of other nonmanual sign language articulators, namely, the face. We introduce a reduced set of 128 face keypoints that signify the signer's face contour.
```bash
python -m sign_language_segmentation.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --pose_components POSE_LANDMARKS LEFT_HAND_LANDMARKS RIGHT_HAND_LANDMARKS FACE_LANDMARKS --pose_reduce_face=true
```

### E3: Adding Optical Flow

At every time step $t$ we append the optical flow between $t$ and $t-1$ to the current pose frame as an additional dimension after the $XYZ$ axes.
```bash
python -m sign_language_segmentation.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --optical_flow=true
```

### E4: Adding 3D Hand Normalization

At every time step, we normalize the hand poses and concatenate them to the current pose frame.
```bash
python -m sign_language_segmentation.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --optical_flow=true --hand_normalization=true
```

### E5: Autoregressive Encoder

We add autoregressive connections between time steps to encourage consistent output labels. The logits at time step $t$ are concatenated to the input of the next time step, $t+1$. This modification is implemented bidirectionally by stacking two autoregressive encoders and adding their output up before the Softmax operation. However, this approach is inherently slow, as we have to fully wait for the previous time step predictions before we can feed them to the next time step.
```bash
python -m sign_language_segmentation.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=4 --encoder_bidirectional=true --encoder_autoregressive=true --optical_flow=true --hand_normalization=true --epochs=50 --patience=10
```

CAUTION: this experiment does not improve the model as expected and runs very slowly.

## Test and Evaluation

To test and evaluate a model, add the `train=false` and `--checkpoint` flag. Take E1 as an example:

```bash
python -m sign_language_segmentation.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --train=false --checkpoint=./models/E1-1/best.ckpt
```

It's also possible to adjust the decoding algorithm by setting the `b_threshold` and the `o_threshold`:

```bash
python -m sign_language_segmentation.src.train --dataset=dgs_corpus --pose=holistic --fps=25 --hidden_dim=256 --encoder_depth=1 --encoder_bidirectional=true --train=false --checkpoint=./models/E1-1/best.ckpt --b_threshold=50 --o_threshold=50
```

To test on an external dataset, see [evaluate_mediapi.py](https://github.com/sign-language-processing/transcription/blob/main/sign_language_segmentation/src/evaluate_mediapi.py) for an example.

## Cite

```bibtex
@inproceedings{moryossef-etal-2023-linguistically,
    title = "Linguistically Motivated Sign Language Segmentation",
    author = {Moryossef, Amit  and Jiang, Zifan  and M{\"u}ller, Mathias  and Ebling, Sarah  and Goldberg, Yoav},
    editor = "Bouamor, Houda  and Pino, Juan  and Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.846",
    doi = "10.18653/v1/2023.findings-emnlp.846",
    pages = "12703--12724",
}
```