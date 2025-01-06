# Sign Language Segmentation

Pose segmentation model on both the sentence and sign level

## Usage

```bash
# Install the package
pip install git+https://github.com/sign-language-processing/segmentation

# Acquire a MediaPipe Holistic pose file
wget -O example.pose https://sign-lanugage-datasets.sign-mt.cloud/poses/holistic/dgs_corpus/1413451-11105600-11163240_a.pose

# Run the model!
pose_to_segments --pose="example.pose" --elan="example.eaf" [--video="example.mp4"]
```

## 2025 Version

The original version of the code supported many experimental model architectures.
In the current version, we simplify the code base, to allow continuous support.

### Summary of Improvements

| Category              | Original (2023)              | Current (2025)          |
|-----------------------|------------------------------|-------------------------|
| Reliability           | Unreliable in the first 2-3s | Should be more reliable |
| Inference Performance | Slow LSTM-based              | Fast CNN-based          |
| Training Efficiency   | Used wasteful padding        | Using packed sequences  |

### Development

<details>
<summary>Create the environment</summary>

```bash
conda create --name segmentation python=3.12 -y
conda activate segmentation
pip install ".[dev]"

# Confirm the environment
pylint sign_language_segmentation
pytest sign_language_segmentation
```

</details>

<details>
<summary>Prepare the dataset</summary>

Requires access to the annotations database.

```bash
# Prepare the entire dataset
python -m sign_language_segmentation.data.create_dataset \
  --poses="/Volumes/Echo/GCS/sign-mt-poses/" \
  --output="/tmp/segmentation/"
  
# Make sure you can load the dataset
python -m sign_language_segmentation.data.dataset \
  --dataset="/tmp/segmentation/"
```

</details>

<details>
<summary>Train the model</summary>

```bash
python -m sign_language_segmentation.src.train \
    --dataset=dgs_corpus \
    --pose=holistic \
    --fps=25 \
    --hidden_dim=256 \
    --encoder_depth=1 \
    --encoder_bidirectional=true
```

</details>

## 2023 Version ([v2023](https://github.com/sign-language-processing/segmentation/tree/v2023))

Exact code for the
paper [Linguistically Motivated Sign Language Segmentation](https://aclanthology.org/2023.findings-emnlp.846).

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
