# Data

We rely on https://tube.sign.mt to provide annotations of sign and sentence spans.
All annotation that start immediately, at time=0 are removed, as these are artifacts of the data creation process.

Captions with the language codes `Sgnw` (SignWriting) and `hns` (HamNoSys) are considered signs. 
`gloss` (Glosses) are considered signs only if there is no whitespace in the annotation.
All other captions are considered sentences.

## Split

In [split.json](./split.json), we extend the data split from 
[split.3.0.0-uzh-document](https://github.com/sign-language-processing/datasets/blob/master/sign_language_datasets/datasets/dgs_corpus/splits/split.3.0.0-uzh-document.json)
to remove the hard-coded train set.


## Storage

- Poses are pre-processed by normalizing the shoulders to be of size 1, 
the face mesh coordinates are reduced to only include the face contour,
and the legs are removed.

- BIO is stored as `uint8`, with the following mapping `{"O": 0, "B": 1, "I": 2}`.

We pack the data by introducing 100 empty frames between each pose file.
This should be the same as having CNN padding in inference.