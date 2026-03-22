# Experiments

Reproducing and improving on the EMNLP 2023 sign language segmentation results.
All experiments use the DGS Corpus 3.0.0-uzh-document split, MediaPipe Holistic poses.

**Paper baseline (EMNLP 2023, reported):** Sign F1 0.63, Sign IoU 0.69

**Paper checkpoints re-evaluated on our current dev split (9/10 videos):**
- E1s (BiLSTM 4-layer, 25fps, no optical flow): Sign IoU=0.4398, Sign Seg F1=0.9265
- E4s (BiLSTM 1-layer, optical flow + 3D hand norm): Sign IoU=0.5129, Sign Seg F1=0.9690
- Both use IO-fallback (pred counts ~1.5-2× gold), confirming B-class is unreliable at inference time on our poses
- Our E38 (CNN-medium + dice) at Sign IoU=0.6162 outperforms both paper checkpoints on this split

## Results

| Experiment | Params | Epochs | Sign F1 | Sign IoU | Sign Seg F1 | Phrase F1 | Phrase IoU | Phrase Seg F1 |
|---|---|---|---|---|---|---|---|---|
| E166-1024-depth4-drop01-fixchunk-hmval-3h | 5,734,141 | 199 | 0.5445 | 0.6411 | 0.9733 | 0.5464 | 0.8996 | 0.9006 |
| E165-1536-batch8-drop01-fixchunk-hmval-3h | 8,097,277 | 70 | 0.4909 | 0.5598 | 0.9654 | 0.5448 | 0.8892 | 0.8540 |
| E162-1536-batch8-drop01-fixchunk-3h | 7,801,597 | 139 | 0.5249 | 0.6156 | 0.9712 | 0.5491 | 0.6661 | 0.7567 |
| E161-2048-batch8-drop01-fd0-fixchunk-3h | 7,801,597 | 135 | 0.5247 | 0.6030 | 0.9622 | 0.5571 | 0.5865 | 0.6336 |
| E160-2048-batch8-drop01-fixchunk-3h | 7,801,597 | 121 | 0.5179 | 0.6068 | 0.9638 | 0.5461 | 0.8669 | 0.9039 |
| E159-1536-nodropframe-drop01-3h | 7,801,597 | 101 | 0.5075 | 0.6074 | 0.9721 | 0.5586 | 0.4071 | 0.4722 |
| E158-1024-nodropframe-drop01-30m | 7,801,597 | 57 | 0.5060 | 0.5819 | 0.9698 | 0.5469 | 0.4010 | 0.4923 |
| E157-1536-nodropframe-drop01-30m | 7,801,597 | 45 | 0.4892 | 0.5955 | 0.9737 | 0.5467 | 0.5511 | 0.6335 |
| E156-2048-nodropframe-drop01-30m | 7,801,597 | 34 | 0.5035 | 0.5874 | 0.9714 | 0.5452 | 0.8422 | 0.8796 |
| E155-1536-nofpsaug-dropout01-30m | 7,801,597 | 33 | 0.3660 | 0.4867 | 0.8249 | 0.5344 | 0.7402 | 0.7677 |
| E154-attnmask-1536-dropout01-30m | 7,801,597 | 45 | 0.4950 | 0.5823 | 0.9669 | 0.5750 | 0.7644 | 0.8381 |
| E153-attnmask-2048-dropout01-30m | 7,801,597 | 34 | 0.5015 | 0.5954 | 0.9741 | 0.5459 | 0.5564 | 0.6490 |
| E152-attnmask-1024-dropout01-30m | 7,801,597 | 57 | 0.4990 | 0.5818 | 0.9737 | 0.5596 | 0.5930 | 0.6467 |
| E151-attnmask-2048-batch8-30m | 7,801,597 | 34 | 0.4880 | 0.5886 | 0.9684 | 0.5407 | 0.6803 | 0.7401 |
| E150-best-config-12h | 7,801,597 | 170 | 0.5100 | 0.5971 | 0.9708 | 0.5516 | 0.6363 | 0.7033 |
| E149-finetune-e145-lr1e4-3h | 7,801,597 | 85 | 0.5167 | 0.5952 | 0.9713 | 0.5431 | 0.6930 | 0.7818 |
| E148-frames2048-batch8-dropout01-6h | 7,801,597 | 98 | 0.5194 | 0.6058 | 0.9795 | 0.5423 | 0.7465 | 0.7928 |
| E147-frames2048-batch8-dropout01-3h | 7,801,597 | 123 | 0.5278 | 0.6165 | 0.9756 | 0.5548 | 0.7393 | 0.7536 |
| E146-e123-6h | 7,801,597 | 175 | 0.5291 | 0.6232 | 0.9758 | 0.5633 | 0.6981 | 0.7484 |
| E145-body-dropout01-3h | 7,801,597 | 84 | 0.4963 | 0.5947 | 0.9736 | 0.5462 | 0.9072 | 0.9128 |
| E144-curriculum-batch8-3h | 7,801,597 | 90 | 0.5028 | 0.5924 | 0.9696 | 0.5453 | 0.8681 | 0.9096 |
| E143-frames2048-batch8-3h | 7,801,597 | 108 | 0.5285 | 0.6244 | 0.9796 | 0.5513 | 0.8607 | 0.8451 |
| E141-curriculum-body-speed-3h | 7,801,597 | 52 | 0.3853 | 0.4563 | 0.8784 | 0.4959 | 0.8284 | 0.8373 |
| E140-frames2048-body-speed-3h | 7,801,597 | 72 | 0.3649 | 0.4795 | 0.8212 | 0.5223 | 0.6974 | 0.7309 |
| E139-curriculum-3h | 7,801,597 | 125 | 0.5129 | 0.6109 | 0.9751 | 0.5566 | 0.6051 | 0.6877 |
| E138-frames2048-3h | 7,801,597 | 114 | 0.5253 | 0.6223 | 0.9726 | 0.5499 | 0.5657 | 0.6424 |
| E137-body-speed-aug-1h | 7,801,597 | 62 | 0.3657 | 0.4743 | 0.8944 | 0.4938 | 0.8178 | 0.8988 |
| E136-speed-aug-1h | 7,801,597 | 54 | 0.3675 | 0.4831 | 0.8618 | 0.5258 | 0.8501 | 0.7538 |
| E135-body-dropout-1h | 7,801,597 | 139 | 0.5117 | 0.6060 | 0.9678 | 0.5476 | 0.7477 | 0.7997 |
| E134-ft-dice10-cosine | 7,801,597 | 57 | 0.5180 | 0.5950 | 0.9739 | 0.5460 | 0.7066 | 0.7912 |
| E133-ft-patience100-cosine | 7,801,597 | 70 | 0.5115 | 0.5918 | 0.9691 | 0.5468 | 0.8486 | 0.8979 |
| E132-ft-steps200-cosine | 7,801,597 | 50 | 0.5115 | 0.5916 | 0.9695 | 0.5469 | 0.8486 | 0.8995 |
| E131-ft-nodropout-cosine | 7,801,597 | 69 | 0.5209 | 0.5361 | 0.8645 | 0.5467 | 0.7480 | 0.8343 |
| E130-ft-noaug-cosine | 7,801,597 | 63 | 0.5240 | 0.5451 | 0.8667 | 0.5475 | 0.6913 | 0.7664 |
| E129-ft-lr5e5-constant | 7,801,597 | 50 | 0.5115 | 0.5911 | 0.9692 | 0.5467 | 0.8483 | 0.8996 |
| E128-ft-lr1e5-cosine | 7,801,597 | 50 | 0.5111 | 0.6036 | 0.9721 | 0.5488 | 0.9060 | 0.9159 |
| E127-ft-lr5e5-cosine | 7,801,597 | 50 | 0.5115 | 0.5916 | 0.9693 | 0.5468 | 0.8485 | 0.8976 |
| E126-ft-lr1e4-cosine | 7,801,597 | 54 | 0.5114 | 0.5978 | 0.9716 | 0.5456 | 0.7751 | 0.8538 |
| E125-fresh-30m | 7,801,597 | 137 | 0.5169 | 0.6135 | 0.9763 | 0.5498 | 0.8016 | 0.8533 |
| E124-full-aug-tempo-fixed-1h | 3,775,869 | 91 | 0.5113 | 0.5606 | 0.9788 | 0.5570 | 0.4489 | 0.5160 |
| E123-h384-lr5e4-full-aug-3h | 7,801,597 | 90 | 0.5042 | 0.6018 | 0.9698 | 0.5438 | 0.9012 | 0.8853 |
| E122-h384-full-aug-3h | 7,801,597 | 68 | 0.4851 | 0.5664 | 0.9755 | 0.5479 | 0.7999 | 0.8001 |
| E121-full-aug-3h | 3,775,869 | 87 | 0.4984 | 0.5887 | 0.9756 | 0.5548 | 0.5372 | 0.5852 |
| E119-h384-steps200-full-aug-1h | 7,801,597 | 109 | 0.4708 | 0.5762 | 0.9707 | 0.5699 | 0.8122 | 0.8488 |
| E118-steps200-full-aug-1h | 3,775,869 | 93 | 0.5103 | 0.5886 | 0.9755 | 0.5566 | 0.6186 | 0.6603 |
| E117-h384-lr5e4-full-aug-1h | 7,801,597 | 129 | 0.5166 | 0.6019 | 0.9622 | 0.5642 | 0.6856 | 0.7734 |
| E116-lr5e4-full-aug-1h | 3,775,869 | 62 | 0.4621 | 0.5659 | 0.9671 | 0.5441 | 0.8552 | 0.8796 |
| E115-dropout30-1h | 3,775,869 | 131 | 0.5201 | 0.6132 | 0.9761 | 0.5686 | 0.8003 | 0.8195 |
| E114-dropout05-1h | 3,775,869 | 85 | 0.5017 | 0.5974 | 0.9721 | 0.5435 | 0.8614 | 0.8351 |
| E113-h384-fps-dice15-1h | 7,801,597 | 115 | 0.5138 | 0.6051 | 0.9802 | 0.5478 | 0.7564 | 0.8069 |
| E112-h384-full-aug-1h | 7,801,597 | 64 | 0.4798 | 0.5663 | 0.9758 | 0.5482 | 0.9052 | 0.8694 |
| E111-full-aug-1h | 3,775,869 | 93 | 0.5066 | 0.5794 | 0.9695 | 0.5434 | 0.4998 | 0.4774 |
| E109-fps-aug-dropout-dice15-1h | 3,775,869 | 119 | 0.5186 | 0.6137 | 0.9730 | 0.5543 | 0.6922 | 0.6876 |
| E108-fps-aug-dice15-fixed-1h | 3,775,869 | 76 | 0.4859 | 0.5939 | 0.9803 | 0.5461 | 0.6605 | 0.7526 |
| E107-fps-aug-dropout-fixed-1h | 3,775,869 | 110 | 0.5090 | 0.6055 | 0.9716 | 0.5448 | 0.7992 | 0.8469 |
| E106-fps-aug-fixed-1h | 3,775,869 | 131 | 0.5062 | 0.6066 | 0.9719 | 0.5789 | 0.7865 | 0.8598 |
| E104-dice15-1h | 3,775,869 | 102 | 0.5154 | 0.6150 | 0.9705 | 0.5494 | 0.8082 | 0.8291 |
| E103-robust-dice15-1h | 3,775,869 | 153 | 0.5041 | 0.5404 | 0.9456 | 0.5321 | 0.2607 | 0.3373 |
| E102-fps-aug-dropout-1h | 3,775,869 | 121 | 0.5041 | 0.4314 | 0.7495 | 0.5325 | 0.4360 | 0.5189 |
| E101-fps-aug-1h | 3,775,869 | 188 | 0.5049 | 0.3931 | 0.7267 | 0.5317 | 0.3917 | 0.4581 |
| E99-attn-pe-drop05-1h | 2,724,733 | 27 | 0.4978 | 0.5143 | 0.8637 | 0.5591 | 0.7393 | 0.7973 |
| E98-attn-pe-bs16-1h | 2,724,733 | 135 | 0.5198 | 0.5847 | 0.9679 | 0.5449 | 0.6192 | 0.6977 |
| E97-attn-h384-d6-nh8-pe-lr5e4-1h | 7,801,597 | 138 | 0.5332 | 0.6289 | 0.9752 | 0.5546 | 0.6773 | 0.7612 |
| E96-attn-pe-accel-1h | 2,725,222 | 136 | 0.5327 | 0.6136 | 0.9777 | 0.5496 | 0.6595 | 0.7313 |
| E95-attn-pe-ff4-1h | 3,775,357 | 143 | 0.5300 | 0.6167 | 0.9717 | 0.5547 | 0.5817 | 0.6842 |
| E94-attn-pe-bdice-1h | 2,724,733 | 50 | 0.3837 | 0.4633 | 0.9006 | 0.5141 | 0.8539 | 0.7696 |
| E93-attn-pe-dice15-1h | 2,724,733 | 129 | 0.5276 | 0.6290 | 0.9786 | 0.5846 | 0.7097 | 0.7975 |
| E92-attn-h512-d6-nh8-pe-1h | 13,432,957 | 58 | 0.4809 | 0.5637 | 0.9625 | 0.5287 | 0.8914 | 0.8819 |
| E91-attn-d6-nh8-pe-1h | 3,775,869 | 122 | 0.5193 | 0.6192 | 0.9735 | 0.5477 | 0.7573 | 0.8027 |
| E90-local-attn-h384-pe-1h | 4,854,202 | 81 | 0.4748 | 0.5608 | 0.9660 | 0.5536 | 0.6898 | 0.7340 |
| E89-local-attn-d6-pe-1h | 3,273,658 | 115 | 0.4974 | 0.5910 | 0.9730 | 0.5500 | 0.7555 | 0.7919 |
| E88-attn-pe-frames2048-1h | 2,724,733 | 86 | 0.5174 | 0.6018 | 0.9740 | 0.5467 | 0.7342 | 0.7891 |
| E87-attn-d6-pe-steps200-1h | 3,775,869 | 103 | 0.5145 | 0.6127 | 0.9722 | 0.5508 | 0.7484 | 0.8194 |
| E86-attn-pe-steps200-1h | 2,724,733 | 119 | 0.5125 | 0.5854 | 0.9740 | 0.5650 | 0.6432 | 0.7285 |
| E85-attn-d6-pe-lr5e4-1h | 3,775,869 | 62 | 0.4705 | 0.5655 | 0.9673 | 0.5457 | 0.6653 | 0.7721 |
| E84-attn-pe-lr5e4-1h | 2,724,733 | 129 | 0.5287 | 0.6220 | 0.9790 | 0.5546 | 0.6183 | 0.7057 |
| E83-attn-h512-d6-pe-1h | 13,432,957 | 60 | 0.4906 | 0.5010 | 0.8602 | 0.5361 | 0.5678 | 0.6562 |
| E82-attn-h512-pe-1h | 9,233,533 | 56 | 0.4755 | 0.5168 | 0.9703 | 0.5429 | 0.8972 | 0.7834 |
| E81-attn-h512-1h | 9,245,821 | 77 | 0.4806 | 0.5521 | 0.9583 | 0.5423 | 0.8313 | 0.8363 |
| E80-attn-h384-pe-1h | 5,438,461 | 148 | 0.5348 | 0.6274 | 0.9605 | 0.5473 | 0.7528 | 0.8139 |
| E79-attn-d8-pe-1h | 4,827,005 | 131 | 0.5106 | 0.6137 | 0.9615 | 0.5457 | 0.7936 | 0.7942 |
| E78-attn-d6-pe-1h | 3,775,869 | 142 | 0.5241 | 0.6255 | 0.9744 | 0.5492 | 0.4995 | 0.5899 |
| E77-attn-pe-1h | 2,724,733 | 76 | 0.5029 | 0.5836 | 0.9719 | 0.5597 | 0.6463 | 0.7200 |
| E76-attn-accel | 2,731,366 | 108 | 0.5096 | 0.6096 | 0.9736 | 0.5475 | 0.5525 | 0.6363 |
| E75-attn-bdice | 2,730,877 | 58 | 0.4532 | 0.5353 | 0.9648 | 0.5402 | 0.8900 | 0.7895 |
| E74-attn-dice20 | 2,730,877 | 109 | 0.5193 | 0.6248 | 0.9769 | 0.5485 | 0.7596 | 0.8292 |
| E73-attn-dice15 | 2,730,877 | 109 | 0.5205 | 0.6218 | 0.9737 | 0.5472 | 0.7682 | 0.8519 |
| E72-attn-frames2048 | 2,730,877 | 63 | 0.5266 | 0.6137 | 0.9761 | 0.5533 | 0.7048 | 0.7637 |
| E71-attn-bs16 | 2,730,877 | 105 | 0.5036 | 0.6054 | 0.9774 | 0.5465 | 0.5973 | 0.6927 |
| E70-attn-steps200 | 2,730,877 | 109 | 0.5223 | 0.6202 | 0.9770 | 0.5482 | 0.7367 | 0.8197 |
| E69-attn-lr2e3 | 2,730,877 | 94 | 0.5078 | 0.5920 | 0.9780 | 0.5540 | 0.6570 | 0.7163 |
| E68-attn-lr3e4 | 2,730,877 | 70 | 0.4877 | 0.5742 | 0.9711 | 0.5415 | 0.7707 | 0.7803 |
| E67-attn-lr5e4 | 2,730,877 | 78 | 0.4927 | 0.5801 | 0.9660 | 0.5441 | 0.7118 | 0.7864 |
| E66-local-attn-pe | 2,220,474 | 114 | 0.5051 | 0.5804 | 0.9747 | 0.5402 | 0.7418 | 0.8007 |
| E65-local-attn-d6 | 3,273,658 | 115 | 0.4974 | 0.5910 | 0.9730 | 0.5500 | 0.7555 | 0.7919 |
| E64-local-attn-d4 | 2,220,474 | 114 | 0.5051 | 0.5804 | 0.9747 | 0.5402 | 0.7418 | 0.8007 |
| E63-attn-d6-nh8-pe | 3,775,869 | 97 | 0.5238 | 0.6292 | 0.9784 | 0.5507 | 0.7681 | 0.7898 |
| E62-attn-nh8 | 2,730,877 | 108 | 0.5220 | 0.6059 | 0.9752 | 0.5488 | 0.6808 | 0.7614 |
| E61-attn-d8-pe | 4,827,005 | 91 | 0.5064 | 0.6021 | 0.9766 | 0.5514 | 0.7894 | 0.8631 |
| E60-attn-d6-pe | 3,775,869 | 97 | 0.5237 | 0.6163 | 0.9621 | 0.5591 | 0.6002 | 0.6960 |
| E59-attn-pe | 2,724,733 | 75 | 0.5006 | 0.5778 | 0.9715 | 0.5561 | 0.5270 | 0.5992 |
| E58-attn-h512 | 9,245,821 | 61 | 0.5050 | 0.5842 | 0.9738 | 0.5628 | 0.7362 | 0.8263 |
| E57-attn-h384 | 5,447,677 | 76 | 0.5182 | 0.6131 | 0.9780 | 0.5528 | 0.6471 | 0.7440 |
| E56-attn-d8 | 4,839,293 | 99 | 0.5183 | 0.6098 | 0.9763 | 0.5458 | 0.6356 | 0.6920 |
| E55-attn-d6 | 3,785,085 | 103 | 0.5103 | 0.5975 | 0.9709 | 0.5482 | 0.7778 | 0.8387 |
| E54-attn-d2 | 1,676,669 | 113 | 0.5183 | 0.6137 | 0.9745 | 0.5478 | 0.7712 | 0.8531 |
| E53-cnn-medium-h512 | 834,173 | 135 | 0.5199 | 0.6187 | 0.9759 | 0.5614 | 0.7656 | 0.8152 |
| E52-cnn-medium-attn | 2,730,877 | 158 | 0.5263 | 0.6262 | 0.9758 | 0.5496 | 0.7755 | 0.8571 |
| E51-cnn-medium-lstm-1h | 1,412,989 | 137 | 0.5227 | 0.6164 | 0.9779 | 0.5564 | 0.7440 | 0.8144 |
| E50-cnn-medium-lstm | 1,412,989 | 71 | 0.5081 | 0.5337 | 0.8652 | 0.5605 | 0.6303 | 0.7210 |
| E47-cnn-medium-dice-e200-1h | 622,205 | 199 | 0.5205 | 0.6162 | 0.9786 | 0.5592 | 0.7036 | 0.7917 |
| E46-cnn-medium-dice-1h | 622,205 | 203 | 0.5094 | 0.6093 | 0.9773 | 0.5596 | 0.7504 | 0.8152 |
| E45-tcn-noface-vel-dice | 1,655,304 | 56 | 0.4822 | 0.5573 | 0.9662 | 0.5344 | 0.8915 | 0.8840 |
| E44-cnn-medium-dice-signweight | 622,205 | -1 | 0.4879 | 0.5768 | 0.9727 | 0.5421 | 0.8963 | 0.9265 |
| E44-cnn-medium-dice-signweight | 622,205 | -1 | 0.4879 | 0.5768 | 0.9727 | 0.5421 | 0.8963 | 0.9265 |
| E43-cnn-medium-accel-dice | 622,694 | 118 | 0.5143 | 0.6052 | 0.9741 | 0.5572 | 0.4881 | 0.5700 |
| E42-cnn-large-dice | 4,074,336 | 31 | 0.5123 | 0.6034 | 0.9718 | 0.5502 | 0.7111 | 0.7610 |
| E41-bilstm-noface-dice | 1,622,024 | 126 | 0.5131 | 0.6005 | 0.9777 | 0.5601 | 0.6738 | 0.7454 |
| E40-cnn-noface-vel-dice-fast | 622,205 | 60 | 0.5168 | 0.5944 | 0.9742 | 0.5581 | 0.6623 | 0.6475 |
| E40-cnn-noface-vel-dice-fast | 622,205 | 60 | 0.5168 | 0.5944 | 0.9742 | 0.5581 | 0.6623 | 0.6475 |
| E39-cnn-noface-vel-dice2 | 622,205 | 117 | 0.5196 | 0.6099 | 0.9760 | 0.5569 | 0.7386 | 0.7693 |
| E38-cnn-noface-vel-dice | 622,205 | 121 | 0.5205 | 0.6162 | 0.9786 | 0.5592 | 0.7036 | 0.7917 |
| E37-cnn-noface-vel-iou-e130 | 622,205 | 114 | 0.5098 | 0.5758 | 0.9738 | 0.5637 | 0.8451 | 0.8895 |
| E36-cnn-noface-vel-iou-v2 | 622,205 | 121 | 0.5158 | 0.5933 | 0.9753 | 0.5485 | 0.6381 | 0.6958 |
| E35-cnn-noface-vel-iou-p50 | 622,205 | 115 | 0.5152 | 0.5252 | 0.8666 | 0.5462 | 0.6874 | 0.7456 |
| E34-cnn-noface-vel-iou | 622,205 | 38 | 0.4626 | 0.5657 | 0.9686 | 0.5443 | 0.9046 | 0.9006 |
| E33-tcn-noface-vel | 829,192 | 99 | 0.5416 | 0.1931 | 0.3310 | 0.5833 | 0.7308 | 0.7783 |
| E32-cnn-noface-vel-e200 | 622,205 | 131 | 0.5217 | 0.5107 | 0.8646 | 0.5582 | 0.5302 | 0.5904 |
| E31-bigru-noface | 1,226,760 | 71 | 0.5016 | 0.5144 | 0.9658 | 0.5482 | 0.7656 | 0.8431 |
| E30-cnn-noface-vel-smooth01 | 622,205 | 134 | 0.5221 | 0.5619 | 0.9654 | 0.5733 | 0.7496 | 0.8185 |
| E29-cnn-noface-vel-lr5e4 | 622,205 | 135 | 0.5179 | 0.5650 | 0.9709 | 0.5589 | 0.3204 | 0.3925 |
| E28-cnn-noface-vel-accel | 622,694 | 93 | 0.5107 | 0.5372 | 0.9730 | 0.5547 | 0.5899 | 0.6922 |
| E27b-cnn-noface-vel-25fps | 622,205 | 132 | 0.5403 | 0.0618 | 0.2309 | 0.5437 | 0.6655 | 0.7467 |
| E27-cnn-noface-vel-25fps | 622,205 | 199 | 0.5400 | 0.0643 | 0.2286 | 0.5555 | 0.4962 | 0.5936 |
| E26-cnn-noface-vel-2048 | 622,205 | 74 | 0.5085 | 0.5392 | 0.9639 | 0.5583 | 0.7241 | 0.7760 |
| E25-bilstm-noface-focal2 | 1,622,024 | 66 | 0.3990 | 0.0000 | 0.0000 | 0.5435 | 0.7893 | 0.8278 |
| E24-cnn-noface-vel-focal2 | 622,205 | 130 | 0.5254 | 0.3835 | 0.7547 | 0.5657 | 0.6875 | 0.7504 |
| E23-bilstm-noface-adam | 1,622,024 | 193 | 0.5121 | 0.5580 | 0.9666 | 0.5604 | 0.6464 | 0.7271 |
| E22-cnn-noface-vel-bs16 | 622,205 | 129 | 0.5137 | 0.4903 | 0.8568 | 0.5579 | 0.7468 | 0.8154 |
| E21-cnn-lstm-noface-vel | 904,634 | 194 | 0.5182 | 0.4385 | 0.7554 | 0.5628 | 0.7417 | 0.8363 |
| E20-cnn-noface-vel-h320 | 662,909 | 115 | 0.5168 | 0.5086 | 0.8579 | 0.5575 | 0.4904 | 0.5742 |
| E19-cnn-noface-vel-2d | 621,879 | 132 | 0.5236 | 0.5058 | 0.8614 | 0.5545 | 0.3850 | 0.4827 |
| E18-bilstm-noface-vel | 1,660,424 | 188 | 0.5258 | 0.5112 | 0.8595 | 0.5603 | 0.6818 | 0.7521 |
| E17-cnn-noface-vel | 622,205 | 101 | 0.5161 | 0.5703 | 0.9761 | 0.5481 | 0.6039 | 0.6906 |
| E16-bilstm-noface-bweight | 1,622,024 | 192 | 0.5524 | 0.0655 | 0.3404 | 0.5715 | 0.7434 | 0.8239 |
| E15-bilstm-noface | 1,622,024 | 193 | 0.5128 | 0.5620 | 0.9718 | 0.5917 | 0.7451 | 0.8287 |
| E14-cnn-medium-noface-200 | 621,716 | 135 | 0.5163 | 0.5628 | 0.9649 | 0.5580 | 0.5391 | 0.6060 |
| E13-cnn-medium-noface | 621,716 | 99 | 0.5118 | 0.5501 | 0.9634 | 0.5587 | 0.5955 | 0.6497 |
| E12-cnn-medium-velocity | 654,484 | 46 | 0.4998 | 0.5510 | 0.9720 | 0.5625 | 0.6660 | 0.7278 |
| E11-cnn-medium-wide | 760,468 | 21 | 0.4716 | 0.4481 | 0.9535 | 0.5441 | 0.9012 | 0.9272 |
| E10-bilstm-highweight | 1,720,328 | 99 | 0.4701 | 0.4208 | 0.8655 | 0.5386 | 0.4365 | 0.5354 |
| E9-cnn-medium-v2 | 654,484 | 99 | 0.5084 | 0.4797 | 0.8594 | 0.5597 | 0.7536 | 0.8322 |
| E8-cnn-lstm | 936,913 | 99 | 0.4960 | 0.4574 | 0.8489 | 0.5427 | 0.6008 | 0.6984 |
| E7-cnn-medium | 654,484 | 88 | 0.5167 | 0.5568 | 0.9728 | 0.5542 | 0.4803 | 0.5655 |
| E7-cnn-medium | 654,484 | 88 | 0.5167 | 0.5568 | 0.9728 | 0.5542 | 0.4803 | 0.5655 |
| E6-bilstm-skip-improved | 1,984,520 | 99 | 0.4736 | 0.3690 | 0.6374 | 0.5409 | 0.4708 | 0.5611 |
| E5-cnn-light-weighted | 227,780 | 98 | 0.4930 | 0.4371 | 0.7463 | 0.5452 | 0.6367 | 0.7361 |
| E4-bilstm-softer-weights | 1,720,072 | 62 | 0.3989 | 0.4653 | 0.8567 | 0.5255 | 0.5594 | 0.6145 |
| E3-cnn-weighted | 227,780 | 99 | 0.4716 | 0.1324 | 0.2574 | 0.5433 | 0.5479 | 0.6378 |
| E2-bilstm-weighted | 1,720,072 | 59 | 0.3744 | 0.4588 | 0.8233 | 0.5393 | 0.4991 | 0.5914 |
| E1-bilstm-4layer | 1,720,072 | 28 | 0.3465 | 0.0676 | 0.3463 | 0.5403 | 0.7712 | 0.8283 |
| BiLSTM depth=4 | 1,720,072 | 76 | 0.4553 | 0.4111 | 0.8200 | 0.5739 | 0.6143 | 0.6659 |
| BiLSTM depth=1 | 534,532 | ~38 | 0.4522 | 0.4717 | 0.9392 | 0.5091 | 0.8119 | 0.8227 |
| BiLSTM 2D depth=1 | 488,712 | ~30 | 0.4462 | 0.4577 | 0.9278 | 0.5242 | 0.8389 | 0.8723 |
| BiLSTM no-norm | 1,720,072 | ~15 | 0.2578 | 0.0000 | 0.0000 | 0.2766 | 0.6561 | 1.0000 |
| CNN (228K) | 227,780 | 95 | **0.5097** | **0.5589** | **0.9772** | 0.5874 | 0.6204 | 0.6359 |
| CNN-large (4.1M) | 4,106,135 | 10 | 0.4512 | 0.4929 | 0.9484 | 0.5790 | **0.9070** | 0.8041 |

## Key Findings
1. **CNN scales well** — CNN-large achieves incredible Phrase IoU (0.91!) with only 10 epochs. Needs more training time.
2. **Small CNN is best for sign F1/IoU** — compact model generalizes well for fine-grained sign segmentation.
3. **BIO index bug fixed** — metrics.py was using wrong probability columns. Major impact on IoU.
4. **Normalization is critical** — without normalize_mean_std, model fails on sign boundaries.
5. **2D vs 3D doesn't matter much** — dropping z-coordinate gives similar performance.
6. **LSTM depth doesn't help** — depth=1 ≈ depth=4 for most metrics, depth=1 much better for phrase.
7. **Weighted loss is arch-specific** — BiLSTM needs it (B never crosses 50% without it → IoU≈0). CNN does NOT want it (already produces confident B; adding B=5 over-segments → IoU 0.56→0.13). Use no weighting or very light weighting for CNN.
8. **CNN naturally detects B** — UNet skip connections give local boundary awareness. BiLSTM needs class weighting to commit to B predictions.
9. **Phrase IoU vs Sign IoU tradeoff with weighting** — B=5 improved sign IoU (0→0.46) but hurt phrase IoU (0.77→0.50). Need to tune weight ratio for BiLSTM.
10. **1024 frames better than 512 for sign IoU** — E7 (1024 frames, val_loss=0.437) beats E9 (512 frames, val_loss=0.445) on sign metrics (0.56 vs 0.48 IoU), though E9 has better phrase IoU (0.75 vs 0.48). Longer context windows improve sign boundary detection.
11. **Original E1s paper uses B-weight ≈ 35x** — Inverse class frequency: O=71.2%, B=2.1%, I=26.7%. Actual inv freq weights: O=1.40, B=48.57, I=3.74 (relative: O=1, B=34.7, I=2.67). Original trained for hours (SLURM job), not 30 min. With 30-min budget, B=20 causes severe overfitting (train=0.009, val=1.74, IoU=0.42).
12. **Wider hidden_dim (384) is slower per epoch** — E11 (760K, hidden=384) runs at ~1.7x slower per epoch than E7 (654K, hidden=256) due to Conv1d scaling with input_size=hidden_dim. Gets only ~20 epochs vs E7's 88 in 30 min. But achieves excellent Phrase IoU (0.90) suggesting more capacity helps phrase at lower epoch count.
13. **Sign boundary recall bottleneck** — All CNN models have high Sign Seg F1 (0.95-0.97) but moderate IoU (0.45-0.56). The model finds signs in the right place but temporal boundary precision is limited. Threshold sweep (b=20-70) shows CNN makes binary predictions (insensitive to threshold).
14. **Velocity features help phrase, neutral for sign** — E12 (velocity) vs E7 (no velocity): Sign IoU 0.551 vs 0.557 (slightly worse), Phrase IoU 0.666 vs 0.480 (much better). Velocity doubles input channels (3→6) causing fewer gradient steps/epoch (69 vs 91). Per-step learning rate is comparable. Use velocity if phrase segmentation is important.
15. **No-face is the key unlock** — E13/E14 (CNN no-face) and E15 (BiLSTM no-face) all beat E7 (CNN with face). No-face reduces 178→75 joints, making each epoch ~1.3x faster → more gradient steps in 30 min. E14 (135 epochs, no-face) achieves Sign IoU=0.5628 vs E7's 0.5568.
16. **BiLSTM no-face ties CNN on sign, dominates on phrase** — E15 (BiLSTM no-face, 130 epochs) vs E14 (CNN no-face, 135 epochs): Sign IoU 0.5620 vs 0.5628 (essentially tied), Phrase IoU 0.7451 vs 0.5391 (BiLSTM much better). Original paper architecture still excellent for phrase segmentation.
17. **B-class weighting breaks sign IoU for both CNN and BiLSTM** — E16 (BiLSTM B=5): Sign F1=0.5524 (highest ever!) but Sign IoU=0.0655 (terrible). Model correctly identifies signing frames but creates wrong segment boundaries (over-segmentation). The original paper trained for 23 hours which allows the model to learn to balance B-recall with boundary precision. In 30 min, B-weighting always causes over-segmentation.
18. **Velocity helps CNN but hurts BiLSTM for sign IoU** — E18 (BiLSTM+noface+vel): Sign IoU=0.5112, Sign SegF1=0.8595 — both worse than E15 (BiLSTM noface: 0.5620, 0.9718). BiLSTM hidden state already captures temporal dynamics; explicit velocity creates conflicting signals. Use velocity with CNN only.
19. **3D (z-coordinate) is important** — E19 (CNN noface+vel+2D): Sign IoU=0.5058 vs E17 (3D): 0.5703. Despite paper using 2D, our MediaPipe poses benefit from the z-depth dimension. Z captures hand approach/retreat from body, a strong sign boundary signal. Always use pose_dims=3.
20. **No-face reduces to 50 joints (not 75)** — With no_face=True, pose_hide_legs + reduce_holistic selects 8 upper-body + 21+21 hand = 50 joints. Confirmed by param count: E7(face)=654K, E13(noface)=622K → difference = 32K = 128*256 = (178-50)*256 joints.
21. **hidden_dim=256 is optimal for CNN no-face+vel** — E20 (hidden_dim=320) val_loss=0.430 (better than E17's 0.436) but Sign IoU=0.5086 (much worse than E17's 0.5703). All thresholds (20-70) give same IoU — binary predictions. The wider model finds sign frames but creates imprecise segment boundaries. 256 is the sweet spot for 30-min budget.
22. **batch_size=16 hurts sign, improves phrase** — E22 (bs=16): Sign IoU=0.4903 (worse), Phrase IoU=0.7468 (new best). Larger batches: better gradient estimates but half the steps/epoch → fewer total gradient steps. For sign IoU, more gradient steps beats more stable gradients. For phrase IoU, better gradient quality helps.
23. **UNet skip connections are critical for sign boundary precision** — Models without temporal UNet (BiLSTM, CNN-LSTM) have Sign SegF1=0.87-0.97. CNN-medium's second PoseEncoderUNetBlock with skip connections preserves multi-scale temporal features at original resolution, enabling precise boundary placement. The skip connections at fine temporal scales are what enable E17's 0.9761 Sign SegF1.
24. **Adam+ReduceLROnPlateau does NOT improve over AdamW+OneCycleLR in 30-min budget** — E23 (BiLSTM noface, Adam+ReduceLR): Sign IoU=0.5580, Phrase IoU=0.6464. E15 (same, AdamW+OneCycleLR): Sign IoU=0.5620, Phrase IoU=0.7451. The paper's optimizer advantage comes from 23h training duration, not the optimizer itself.
25. **Focal loss gamma=2 causes over-segmentation on CNN and sign failure on BiLSTM** — E24 (CNN+focal): Sign IoU=0.3835 (was 0.5703). E25 (BiLSTM+focal): Sign IoU=0.0000 (never commits to B). But phrase IoU improves: E24 phrase IoU=0.6875, E25 phrase IoU=0.7893 (new best). Focal loss helps phrase but breaks sign for both architectures at gamma=2.
26. **25fps normalization breaks sign IoU during TRAINING but not evaluation** — E27b (CNN noface+vel trained at 25fps): Sign IoU=0.0618 (catastrophic). Evaluating E17 (30fps-trained) at 25fps: Sign IoU=0.5563 (fine, −0.014) + Phrase IoU=0.7313 (up from 0.6039!). Training at 25fps fails; 30fps models evaluate well at 25fps. Root cause unclear — possibly mixed-fps batch padding or temporal scale incompatibility with OneCycleLR.
27. **Longer context (2048 frames) reduces epochs and hurts sign IoU** — E26 (2048 frames): Sign IoU=0.5392 (vs E17's 0.5703). 74 epochs in 30 min (vs 101 for 1024 frames). Consistent with finding #10 (1024 > 512): 1024 frames is the optimal training context length.

## Experiment Details

### CNN (228K) — best sign metrics
- 2× PoseEncoderUNetBlock (channels: 3→8→16, then 1→8→16→32→64)
- AdamW lr=1e-3, OneCycleLR, 30-min cap

### CNN-large (4.1M) — best phrase IoU
- 3× PoseEncoderUNetBlock (channels up to 256)
- AdamW lr=1e-3, OneCycleLR
- Only trained 10 epochs (early stopping). Retraining with lr=3e-4, patience=15

### BiLSTM variants
- All use Linear projection + LSTM + dual BIO heads
- Adam lr=1e-3, ReduceLROnPlateau

## Planned Experiments
- [ ] E20: CNN-medium no-face + velocity + hidden_dim=320, epochs=200 (more capacity; no-face makes 1st block fast enough)
- [ ] E21: CNN-lstm no-face + velocity (CNN for spatial + LSTM for temporal context)
- [ ] E22: BiLSTM no-face + optimizer=adam (match original paper's Adam+ReduceLROnPlateau)

## E1-bilstm-4layer [FAILED] — 2026-03-15 04:34

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- no_normalize: False
- num_frames: 1024
- patience: 10
- pose_dims: 3
- steps_per_epoch: 100

**Training:**
- Params: 1,720,072
- Best epoch: -1
- Best val loss: None
- Elapsed: 2.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E1-bilstm-4layer_train.log

**Dev metrics:**
- N/A (training failed or no checkpoint)

**Notes:** Reproducing paper baseline: 4-layer biLSTM, 256 hidden, weighted loss, 50fps

---

## E1-bilstm-4layer [SUCCESS] — 2026-03-15 04:42

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 10
- pose_dims: 3
- steps_per_epoch: 100
- weighted_loss: False

**Training:**
- Params: 1,720,072
- Best epoch: 28
- Best val loss: None
- Elapsed: 7.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E1-bilstm-4layer_train.log

**Dev metrics:**
- Sign: F1=0.3465  IoU=0.0676  SegF1=0.3463
- Phrase: F1=0.5403  IoU=0.7712  SegF1=0.8283

**Notes:** Reproducing paper baseline: 4-layer biLSTM, 256 hidden, 50fps

---

## E2-bilstm-weighted [SUCCESS] — 2026-03-15 04:58

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- pose_dims: 3
- steps_per_epoch: 100
- weighted_loss: True

**Training:**
- Params: 1,720,072
- Best epoch: 59
- Best val loss: None
- Elapsed: 14.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E2-bilstm-weighted_train.log

**Dev metrics:**
- Sign: F1=0.3744  IoU=0.4588  SegF1=0.8233
- Phrase: F1=0.5393  IoU=0.4991  SegF1=0.5914

**Notes:** KEY EXPERIMENT: weighted NLL (O=1,B=5,I=3) — paper's critical ingredient. E1 showed sign IoU collapses without it.

---

## E3-cnn-weighted [SUCCESS] — 2026-03-15 05:28

**Config:**
- arch: cnn
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- pose_dims: 3
- steps_per_epoch: 100
- weighted_loss: True

**Training:**
- Params: 227,780
- Best epoch: 99
- Best val loss: None
- Elapsed: 27.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E3-cnn-weighted_train.log

**Dev metrics:**
- Sign: F1=0.4716  IoU=0.1324  SegF1=0.2574
- Phrase: F1=0.5433  IoU=0.5479  SegF1=0.6378

**Notes:** CNN (best sign arch) + weighted loss. CNN had 0.51 F1 without weighting; weighted loss should unlock IoU too.

---

## E4-bilstm-softer-weights [SUCCESS] — 2026-03-15 05:44

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 3.0
- loss_i_weight: 2.0
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- pose_dims: 3
- steps_per_epoch: 100
- weighted_loss: True

**Training:**
- Params: 1,720,072
- Best epoch: 62
- Best val loss: None
- Elapsed: 15.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E4-bilstm-softer-weights_train.log

**Dev metrics:**
- Sign: F1=0.3989  IoU=0.4653  SegF1=0.8567
- Phrase: F1=0.5255  IoU=0.5594  SegF1=0.6145

**Notes:** BiLSTM with lighter weighting (B=3,I=2 vs B=5,I=3 in E2). Goal: keep sign IoU high while recovering phrase IoU.

---

## E5-cnn-light-weighted [SUCCESS] — 2026-03-15 06:12

**Config:**
- arch: cnn
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 2.0
- loss_i_weight: 1.5
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- pose_dims: 3
- steps_per_epoch: 100
- weighted_loss: True

**Training:**
- Params: 227,780
- Best epoch: 99
- Best val loss: None
- Elapsed: 27.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E5-cnn-light-weighted_train.log

**Dev metrics:**
- N/A (training failed or no checkpoint)

**Notes:** CNN with very light weighting (B=2,I=1.5). CNN already detects B well; tiny nudge might help phrase without hurting sign.

---

## E5-cnn-light-weighted [SUCCESS] — 2026-03-15 06:14

**Config:**
- arch: cnn
- batch_size: 8
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 2.0
- loss_i_weight: 1.5
- num_frames: 1024
- patience: 15
- pose_dims: 3
- weighted_loss: True

**Training:**
- Params: 227,780
- Best epoch: 98
- Best val loss: 0.77
- Elapsed: 32.0 min
- Log: logs/E5-cnn-light-weighted_train.log

**Dev metrics:**
- Sign: F1=0.4930  IoU=0.4371  SegF1=0.7463
- Phrase: F1=0.5452  IoU=0.6367  SegF1=0.7361

**Notes:** CNN + light weighting (B=2,I=1.5). Confirmed: any weighting hurts CNN sign IoU vs no weighting. CNN best = unweighted.

---

## E6-bilstm-skip-improved [SUCCESS] — 2026-03-15 06:40

**Config:**
- arch: bilstm-skip
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: True
- steps_per_epoch: 100
- weighted_loss: False

**Training:**
- Params: 1,984,520
- Best epoch: 99
- Best val loss: None
- Elapsed: 23.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E6-bilstm-skip-improved_train.log

**Dev metrics:**
- Sign: F1=0.4736  IoU=0.3690  SegF1=0.6374
- Phrase: F1=0.5409  IoU=0.4708  SegF1=0.5611

**Notes:** bilstm-skip: gated residuals + RMSNorm + AdamW+OneCycleLR + bf16 + sign-only weighting

---

## E7-cnn-medium [SUCCESS] — 2026-03-15 07:13

**Config:**
- arch: cnn-medium
- batch_size: 6
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- weighted_loss: False

**Training:**
- Params: 654,484
- Best epoch: 88
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E7-cnn-medium_train.log

**Dev metrics:**
- Sign: F1=0.5167  IoU=0.5568  SegF1=0.9728
- Phrase: F1=0.5542  IoU=0.4803  SegF1=0.5655

**Notes:** CNN-medium (~800K) no weighting. CNN unweighted still best; test if more capacity helps. AdamW+OneCycleLR+bf16 new defaults.

---

## E7-cnn-medium [SUCCESS] — 2026-03-15 07:15

**Config:**
- arch: cnn-medium
- batch_size: 6
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- num_frames: 1024
- params: 654484
- patience: 15
- pose_dims: 3
- sign_weighted_loss: False

**Training:**
- Params: 654,484
- Best epoch: 88
- Best val loss: 0.437
- Elapsed: 30.0 min
- Log: logs/E7-cnn-medium_train.log

**Dev metrics:**
- Sign: F1=0.5167  IoU=0.5568  SegF1=0.9728
- Phrase: F1=0.5542  IoU=0.4803  SegF1=0.5655

**Notes:** NEW BEST sign F1. Hit 30-min wall at epoch 88 still improving. AdamW+OneCycleLR+bf16+validation_sign_loss monitoring. More training would help.

---

## E8-cnn-lstm [SUCCESS] — 2026-03-15 07:45

**Config:**
- arch: cnn-lstm
- batch_size: 8
- encoder_depth: 2
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- weighted_loss: False

**Training:**
- Params: 936,913
- Best epoch: 99
- Best val loss: None
- Elapsed: 29.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E8-cnn-lstm_train.log

**Dev metrics:**
- Sign: F1=0.4960  IoU=0.4574  SegF1=0.8489
- Phrase: F1=0.5427  IoU=0.6008  SegF1=0.6984

**Notes:** Hybrid: CNN per-frame features (stride=1) + BiLSTM temporal. Combines CNN local strength with LSTM global context. No weighting.

---

## E9-cnn-medium-v2 [SUCCESS] — 2026-03-15 08:09

**Config:**
- arch: cnn-medium
- batch_size: 6
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_normalize: False
- num_frames: 512
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- weighted_loss: False

**Training:**
- Params: 654,484
- Best epoch: 99
- Best val loss: None
- Elapsed: 22.1 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E9-cnn-medium-v2_train.log

**Dev metrics:**
- Sign: F1=0.5084  IoU=0.4797  SegF1=0.8594
- Phrase: F1=0.5597  IoU=0.7536  SegF1=0.8322

**Notes:** CNN-medium with 512 frames (double diversity, more epochs in 30min). E7 hit 30min wall at ep88 still improving. Patience=20.

---

## E10-bilstm-highweight [SUCCESS] — 2026-03-15 08:42

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_normalize: False
- num_frames: 1024
- optimizer: adam
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 20.0
- sign_i_weight: 4.0
- sign_weighted_loss: True
- steps_per_epoch: 100
- weighted_loss: False

**Training:**
- Params: 1,720,328
- Best epoch: 99
- Best val loss: None
- Elapsed: 24.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E10-bilstm-highweight_train.log

**Dev metrics:**
- Sign: F1=0.4701  IoU=0.4208  SegF1=0.8655
- Phrase: F1=0.5386  IoU=0.4365  SegF1=0.5354

**Notes:** Approximate inverse class freq weights for BiLSTM like original E1s paper. B~20x O weight for sign head only, constant lr like original.

---

## E11-cnn-medium-wide [SUCCESS] — 2026-03-15 09:19

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 384
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- weighted_loss: False

**Training:**
- Params: 760,468
- Best epoch: 21
- Best val loss: None
- Elapsed: 33.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E11-cnn-medium-wide_train.log

**Dev metrics:**
- Sign: F1=0.4716  IoU=0.4481  SegF1=0.9535
- Phrase: F1=0.5441  IoU=0.9012  SegF1=0.9272

**Notes:** CNN-medium with hidden_dim=384 (~760K params). More capacity for precise sign boundaries. Based on E7 being best but 30-min capped.

---

## E12-cnn-medium-velocity [SUCCESS*] — 2026-03-15 09:57

*Note: Docker timeout killed process at 35min but Lightning had already hit max_time=30min and saved checkpoint.

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- no_normalize: False
- num_frames: 1024
- optimizer: None (adamw-onecycle)
- patience: 15
- pose_dims: 3
- steps_per_epoch: 100
- velocity: True

**Training:**
- Params: 654,484
- Best epoch: 46
- Best val loss: 0.476
- Elapsed: ~30 min (hit max_time)
- Log: logs/E12-cnn-medium-velocity_train.log

**Dev metrics:**
- Sign: F1=0.4998  IoU=0.5510  SegF1=0.9720
- Phrase: F1=0.5625  IoU=0.6660  SegF1=0.7278

**Notes:** Velocity features neutral for sign (slightly fewer gradient steps/epoch), significantly better for phrase (IoU 0.67 vs 0.48 for E7). Fewer steps/epoch (69 vs 91) because velocity doubles input channel dimension. Best Phrase IoU among comparable models.

---

## E13-cnn-medium-noface [SUCCESS] — 2026-03-15 10:27

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 100
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: False
- weighted_loss: False

**Training:**
- Params: 621,716
- Best epoch: 99
- Best val loss: None
- Elapsed: 23.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E13-cnn-medium-noface_train.log

**Dev metrics:**
- Sign: F1=0.5118  IoU=0.5501  SegF1=0.9634
- Phrase: F1=0.5587  IoU=0.5955  SegF1=0.6497

**Notes:** No face landmarks (match original E1s setup). Reduces 178→~75 joints, ~2x faster per epoch. Hit max_epochs=100 at only 23 min — model was still improving at epoch 99! Need epochs=200 in E14 to use full 30-min budget.

---

## E14-cnn-medium-noface-200 [SUCCESS] — 2026-03-15 11:05

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: False
- weighted_loss: False

**Training:**
- Params: 621,716
- Best epoch: 135
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E14-cnn-medium-noface-200_train.log

**Dev metrics:**
- Sign: F1=0.5163  IoU=0.5628  SegF1=0.9649
- Phrase: F1=0.5580  IoU=0.5391  SegF1=0.6060

**Notes:** E13 repeat with epochs=200 so 30-min max_time is the limiter. E13 hit max_epochs=100 at only 23 min and was still improving.

---

## E15-bilstm-noface [SUCCESS] — 2026-03-15 11:52

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: False
- weighted_loss: False

**Training:**
- Params: 1,622,024
- Best epoch: 193
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E15-bilstm-noface_train.log

**Dev metrics:**
- Sign: F1=0.5128  IoU=0.5620  SegF1=0.9718
- Phrase: F1=0.5917  IoU=0.7451  SegF1=0.8287

**Notes:** BiLSTM depth=4 with no-face (match original E1s arch+data). Smaller input (75 joints) should speed up training. Test if BiLSTM converges better than CNN with proper data setup.

---

## E16-bilstm-noface-bweight [SUCCESS] — 2026-03-15 12:25

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 5.0
- sign_i_weight: 2.0
- sign_weighted_loss: True
- steps_per_epoch: 100
- velocity: False
- weighted_loss: False

**Training:**
- Params: 1,622,024
- Best epoch: 192
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E16-bilstm-noface-bweight_train.log

**Dev metrics:**
- Sign: F1=0.5524  IoU=0.0655  SegF1=0.3404
- Phrase: F1=0.5715  IoU=0.7434  SegF1=0.8239

**Notes:** BiLSTM no-face + B-weight=5 for sign head. Original paper used B~35x (inv freq). Testing moderate weighting to improve boundary recall without E10-style overfitting.

---

## E17-cnn-noface-vel [SUCCESS] — 2026-03-15 13:04

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 101
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E17-cnn-noface-vel_train.log

**Dev metrics:**
- Sign: F1=0.5161  IoU=0.5703  SegF1=0.9761
- Phrase: F1=0.5481  IoU=0.6039  SegF1=0.6906

**Notes:** CNN-medium no-face + velocity (combine E14 sign strength with E12 phrase strength). Expect ~90-100 epochs in 30 min due to velocity doubling input channels.

---

## E18-bilstm-noface-vel [SUCCESS] — 2026-03-15 13:38

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: True
- weighted_loss: False

**Training:**
- Params: 1,660,424
- Best epoch: 188
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E18-bilstm-noface-vel_train.log

**Dev metrics:**
- Sign: F1=0.5258  IoU=0.5112  SegF1=0.8595
- Phrase: F1=0.5603  IoU=0.6818  SegF1=0.7521

**Notes:** BiLSTM no-face + velocity. Combine E15 phrase strength (BiLSTM) with E17 sign boost (velocity+no_face). Best of both worlds hypothesis.

---

## E19-cnn-noface-vel-2d [SUCCESS] — 2026-03-15 14:11

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 2
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: True
- weighted_loss: False

**Training:**
- Params: 621,879
- Best epoch: 132
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E19-cnn-noface-vel-2d_train.log

**Dev metrics:**
- Sign: F1=0.5236  IoU=0.5058  SegF1=0.8614
- Phrase: F1=0.5545  IoU=0.3850  SegF1=0.4827

**Notes:** 2D WORSE than 3D (E17: 0.5703 vs E19: 0.5058 sign IoU). The z-coordinate captures hand depth, important for detecting sign boundaries. Eval originally failed due to missing --pose_dims 2 in run_experiment.py (now fixed).

---

## E20-cnn-noface-vel-h320 [SUCCESS] — 2026-03-15 14:46

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- hidden_dim: 320
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: True
- weighted_loss: False

**Training:**
- Params: 662,909
- Best epoch: 115
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E20-cnn-noface-vel-h320_train.log

**Dev metrics:**
- Sign: F1=0.5168  IoU=0.5086  SegF1=0.8579
- Phrase: F1=0.5575  IoU=0.4904  SegF1=0.5742

**Notes:** E17 with hidden_dim=320 (more capacity). No-face makes first block fast, second block 1.25x slower. Expect ~90 epochs in 30 min vs E17's 101.

---

## E21-cnn-lstm-noface-vel [SUCCESS] — 2026-03-15 15:24

**Config:**
- arch: cnn-lstm
- batch_size: 8
- encoder_depth: 2
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: True
- weighted_loss: False

**Training:**
- Params: 904,634
- Best epoch: 194
- Best val loss: None
- Elapsed: 28.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E21-cnn-lstm-noface-vel_train.log

**Dev metrics:**
- Sign: F1=0.5182  IoU=0.4385  SegF1=0.7554
- Phrase: F1=0.5628  IoU=0.7417  SegF1=0.8363

**Notes:** CNN-LSTM hybrid: CNN compresses joint features, BiLSTM adds temporal context. depth=2 to keep params under 2M. No-face+velocity for best features.

---

## E22-cnn-noface-vel-bs16 [SUCCESS] — 2026-03-15 15:59

**Config:**
- arch: cnn-medium
- batch_size: 16
- encoder_depth: 4
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 129
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E22-cnn-noface-vel-bs16_train.log

**Dev metrics:**
- Sign: F1=0.5137  IoU=0.4903  SegF1=0.8568
- Phrase: F1=0.5579  IoU=0.7468  SegF1=0.8154

**Notes:** E17 with batch_size=16 (more stable gradients). With no-face+vel, GPU can handle larger batches. Halves steps/epoch but better gradient estimates.

---

## E23-bilstm-noface-adam [SUCCESS] — 2026-03-15 16:42

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: adam
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: False
- weighted_loss: False

**Training:**
- Params: 1,622,024
- Best epoch: 193
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E23-bilstm-noface-adam_train.log

**Dev metrics:**
- Sign: F1=0.5121  IoU=0.5580  SegF1=0.9666
- Phrase: F1=0.5604  IoU=0.6464  SegF1=0.7271

**Notes:** E15 but with optimizer=adam (ReduceLROnPlateau, matches original paper). Test if Adam+ReduceLR gives better sign IoU than OneCycleLR for BiLSTM.

---

## E24-cnn-noface-vel-focal2 [SUCCESS] — 2026-03-15 17:15

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 2.0
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 130
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E24-cnn-noface-vel-focal2_train.log

**Dev metrics:**
- Sign: F1=0.5254  IoU=0.3835  SegF1=0.7547
- Phrase: F1=0.5657  IoU=0.6875  SegF1=0.7504

**Notes:** focal_loss_gamma2 on best CNN config (E17 baseline 0.5703)

---

## E25-bilstm-noface-focal2 [SUCCESS] — 2026-03-15 17:28

**Config:**
- arch: bilstm
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 2.0
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: False
- weighted_loss: False

**Training:**
- Params: 1,622,024
- Best epoch: 66
- Best val loss: None
- Elapsed: 11.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E25-bilstm-noface-focal2_train.log

**Dev metrics:**
- Sign: F1=0.3990  IoU=0.0000  SegF1=0.0000
- Phrase: F1=0.5435  IoU=0.7893  SegF1=0.8278

**Notes:** focal_loss_gamma2 on BiLSTM noface - test if focal helps BiLSTM learn B class

---

## E26-cnn-noface-vel-2048 [SUCCESS] — 2026-03-15 18:22

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 2048
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 74
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E26-cnn-noface-vel-2048_train.log

**Dev metrics:**
- Sign: F1=0.5085  IoU=0.5392  SegF1=0.9639
- Phrase: F1=0.5583  IoU=0.7241  SegF1=0.7760

**Notes:** longer context (2048 frames) for better temporal boundary precision vs E17 (1024 frames)

---

## E27-cnn-noface-vel-25fps [SUCCESS] — 2026-03-15 18:52

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: 25.0
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 199
- Best val loss: None
- Elapsed: 27.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E27-cnn-noface-vel-25fps_train.log

**Dev metrics:**
- Sign: F1=0.5400  IoU=0.0643  SegF1=0.2286
- Phrase: F1=0.5555  IoU=0.4962  SegF1=0.5936

**Notes:** downsample all poses to 25fps to match original paper's temporal resolution

---

## E27b-cnn-noface-vel-25fps [SUCCESS] — 2026-03-15 19:30

**Config:**
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: 25.0
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 132
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E27b-cnn-noface-vel-25fps_train.log

**Dev metrics:**
- Sign: F1=0.5403  IoU=0.0618  SegF1=0.2309
- Phrase: F1=0.5437  IoU=0.6655  SegF1=0.7467

**Notes:** 25fps normalization with fixed load_num_frames proportional scaling (E27 had bug)

---

## E28-cnn-noface-vel-accel [SUCCESS] — 2026-03-15 20:08

**Config:**
- acceleration: True
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,694
- Best epoch: 93
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E28-cnn-noface-vel-accel_train.log

**Dev metrics:**
- Sign: F1=0.5107  IoU=0.5372  SegF1=0.9730
- Phrase: F1=0.5547  IoU=0.5899  SegF1=0.6922

**Notes:** add 2nd-order temporal derivative (acceleration) to pose features; hypothesis: high at B-class boundary transitions

---

## E29-cnn-noface-vel-lr5e4 [SUCCESS] — 2026-03-15 20:41

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 135
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E29-cnn-noface-vel-lr5e4_train.log

**Dev metrics:**
- Sign: F1=0.5179  IoU=0.5650  SegF1=0.9709
- Phrase: F1=0.5589  IoU=0.3204  SegF1=0.3925

**Notes:** lower lr=5e-4 (half default); hypothesis: more conservative lr improves boundary precision

---

## E30-cnn-noface-vel-smooth01 [SUCCESS] — 2026-03-15 21:15

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.1
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 134
- Best val loss: None
- Elapsed: 31.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E30-cnn-noface-vel-smooth01_train.log

**Dev metrics:**
- Sign: F1=0.5221  IoU=0.5619  SegF1=0.9654
- Phrase: F1=0.5733  IoU=0.7496  SegF1=0.8185

**Notes:** label smoothing alpha=0.1 on E17 config; hypothesis: regularize overconfident predictions for better generalization

---

## E31-bigru-noface [SUCCESS] — 2026-03-15 21:24

**Config:**
- acceleration: False
- arch: bigru
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: False
- weighted_loss: False

**Training:**
- Params: 1,226,760
- Best epoch: 71
- Best val loss: None
- Elapsed: 8.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E31-bigru-noface_train.log

**Dev metrics:**
- Sign: F1=0.5016  IoU=0.5144  SegF1=0.9658
- Phrase: F1=0.5482  IoU=0.7656  SegF1=0.8431

**Notes:** BiGRU (fewer params than BiLSTM, potentially faster/more epochs); compare to E15 BiLSTM noface

---

## E32-cnn-noface-vel-e200 [SUCCESS] — 2026-03-15 22:02

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 15
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 131
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E32-cnn-noface-vel-e200_train.log

**Dev metrics:**
- Sign: F1=0.5217  IoU=0.5107  SegF1=0.8646
- Phrase: F1=0.5582  IoU=0.5302  SegF1=0.5904

**Notes:** E17 config but epochs=200 to use full 30-min budget; E17 was epoch-limited at 101/100

---

## E33-tcn-noface-vel [SUCCESS] — 2026-03-15 22:20

**Config:**
- acceleration: False
- arch: tcn
- batch_size: 8
- encoder_depth: 8
- epochs: 100
- focal_gamma: 0.0
- hidden_dim: 128
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 829,192
- Best epoch: 99
- Best val loss: None
- Elapsed: 15.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E33-tcn-noface-vel_train.log

**Dev metrics:**
- Sign: F1=0.5416  IoU=0.1931  SegF1=0.3310
- Phrase: F1=0.5833  IoU=0.7308  SegF1=0.7783

**Notes:** Dilated TCN depth=8 hidden=128; dilation=[1,2,4,...,128]; RF=1021 frames; ~829K params; E32 showed more epochs hurts so reverting to E17 patience=20 epochs=100 baseline

---

## E34-cnn-noface-vel-iou [SUCCESS] — 2026-03-15 22:23

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 38
- Best val loss: None
- Elapsed: 14.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E34-cnn-noface-vel-iou_train.log

**Dev metrics:**
- Sign: F1=0.4626  IoU=0.5657  SegF1=0.9686
- Phrase: F1=0.5443  IoU=0.9046  SegF1=0.9006

**Notes:** E17 config but checkpoint metric changed to val_sign_iou; fixes loss/IoU proxy mismatch from E32 regression

---

## E35-cnn-noface-vel-iou-p50 [SUCCESS] — 2026-03-15 22:57

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 0.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 115
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E35-cnn-noface-vel-iou-p50_train.log

**Dev metrics:**
- Sign: F1=0.5152  IoU=0.5252  SegF1=0.8666
- Phrase: F1=0.5462  IoU=0.6874  SegF1=0.7456

**Notes:** E17 config + val_sign_iou checkpoint metric + patience=50; E34 stopped too early at epoch 18 with patience=20; extended patience to reach later convergence

---

## E36-cnn-noface-vel-iou-v2 [FAILED] — 2026-03-15 23:03

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 0.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: -1
- Best val loss: None
- Elapsed: 1.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E36-cnn-noface-vel-iou-v2_train.log

**Dev metrics:**
- N/A (training failed or no checkpoint)

**Notes:** E17 config + CORRECTED val_sign_iou (uses probs_to_segments+segment_IoU to exactly match evaluate.py); E35 used argmax-based IoU which diverged from eval metric

---

## E36-cnn-noface-vel-iou-v2 [SUCCESS] — 2026-03-15 23:35

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 0.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 121
- Best val loss: None
- Elapsed: 31.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E36-cnn-noface-vel-iou-v2_train.log

**Dev metrics:**
- Sign: F1=0.5158  IoU=0.5933  SegF1=0.9753
- Phrase: F1=0.5485  IoU=0.6381  SegF1=0.6958

**Notes:** E17 config + CORRECTED val_sign_iou matching evaluate.py exactly (probs_to_segments+segment_IoU); fixes argmax vs threshold mismatch that hurt E34/E35

---

## E37-cnn-noface-vel-iou-e130 [SUCCESS] — 2026-03-16 00:07

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 0.0
- encoder_depth: 4
- epochs: 130
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 114
- Best val loss: None
- Elapsed: 29.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E37-cnn-noface-vel-iou-e130_train.log

**Dev metrics:**
- Sign: F1=0.5098  IoU=0.5758  SegF1=0.9738
- Phrase: F1=0.5637  IoU=0.8451  SegF1=0.8895

**Notes:** E36 config but epochs=130 to fully anneal LR within 30-min budget; E36 trained 116/200 epochs (58% LR decay) and was time-limited; epochs=130 allows ~89% LR decay at epoch 116

---

## E38-cnn-noface-vel-dice [SUCCESS] — 2026-03-16 00:53

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 121
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E38-cnn-noface-vel-dice_train.log

**Dev metrics:**
- Sign: F1=0.5205  IoU=0.6162  SegF1=0.9786
- Phrase: F1=0.5592  IoU=0.7036  SegF1=0.7917

**Notes:** E36 config + Dice loss (weight=1.0); directly optimizes sign frame overlap during training; combined with corrected IoU checkpoint metric; hypothesis: faster convergence → higher Sign IoU in 30-min budget

---

## E39-cnn-noface-vel-dice2 [SUCCESS] — 2026-03-16 01:32

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 2.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 117
- Best val loss: None
- Elapsed: 30.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E39-cnn-noface-vel-dice2_train.log

**Dev metrics:**
- Sign: F1=0.5196  IoU=0.6099  SegF1=0.9760
- Phrase: F1=0.5569  IoU=0.7386  SegF1=0.7693

**Notes:** E38 config but dice_loss_weight=2.0; E38 was time-limited at 103 epochs still improving with IoU=0.616; stronger Dice signal may converge even faster, potentially reaching higher IoU in 30-min budget

---

## E40-cnn-noface-vel-dice-fast [SUCCESS] — 2026-03-16 02:08

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 60
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E40-cnn-noface-vel-dice-fast_train.log

**Dev metrics:**
- Sign: F1=0.5168  IoU=0.5944  SegF1=0.9742
- Phrase: F1=0.5581  IoU=0.6623  SegF1=0.6475

**Notes:** E38 config + single-pass validation optimization (eliminates redundant forward pass in validation_step); should give ~13 sec/epoch vs E38 17 sec -> ~138 epochs in 30 min vs E38 103

---

## E40-cnn-noface-vel-dice-fast [SUCCESS] — 2026-03-16 02:08

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 60
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E40-cnn-noface-vel-dice-fast_train.log

**Dev metrics:**
- Sign: F1=0.5168  IoU=0.5944  SegF1=0.9742
- Phrase: F1=0.5581  IoU=0.6623  SegF1=0.6475

**Notes:** E38 config + single-pass validation optimization (eliminates redundant forward pass in validation_step); should give ~13 sec/epoch vs E38's 17 sec → ~138 epochs in 30 min vs E38's 103

---

## E41-bilstm-noface-dice [SUCCESS] — 2026-03-16 02:44

**Config:**
- acceleration: False
- arch: bilstm
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: False
- weighted_loss: False

**Training:**
- Params: 1,622,024
- Best epoch: 126
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E41-bilstm-noface-dice_train.log

**Dev metrics:**
- Sign: F1=0.5131  IoU=0.6005  SegF1=0.9777
- Phrase: F1=0.5601  IoU=0.6738  SegF1=0.7454

**Notes:** Paper architecture (BiLSTM 4-layer, no-face, no-velocity) + Dice loss + corrected IoU metric; E15 (BiLSTM no-face) got 0.562 with loss metric; testing if Dice brings it to 0.62+

---

## E42-cnn-large-dice [SUCCESS] — 2026-03-16 03:21

**Config:**
- acceleration: False
- arch: cnn-large
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 4,074,336
- Best epoch: 31
- Best val loss: None
- Elapsed: 31.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E42-cnn-large-dice_train.log

**Dev metrics:**
- Sign: F1=0.5123  IoU=0.6034  SegF1=0.9718
- Phrase: F1=0.5502  IoU=0.7111  SegF1=0.7610

**Notes:** CNN-large + dice=1.0 + no_face + velocity; same as E38 but 3-stage UNet vs 2-stage; testing if more CNN capacity improves past 0.6162

---

## E43-cnn-medium-accel-dice [SUCCESS] — 2026-03-16 03:54

**Config:**
- acceleration: True
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,694
- Best epoch: 118
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E43-cnn-medium-accel-dice_train.log

**Dev metrics:**
- Sign: F1=0.5143  IoU=0.6052  SegF1=0.9741
- Phrase: F1=0.5572  IoU=0.4881  SegF1=0.5700

**Notes:** CNN-medium + dice=1.0 + velocity + acceleration; same as E38 but adding 2nd-order temporal derivatives; testing if richer motion features help past 0.6162

---

## E44-cnn-medium-dice-signweight [SUCCESS] — 2026-03-16 04:27

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: True
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: -1
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E44-cnn-medium-dice-signweight_train.log

**Dev metrics:**
- Sign: F1=0.4879  IoU=0.5768  SegF1=0.9727
- Phrase: F1=0.5421  IoU=0.8963  SegF1=0.9265

**Notes:** CNN-medium + dice=1.0 + sign_weighted_loss (B=3, I=2) + no_face + velocity; testing if class weighting for rare B frames improves boundary detection past E38's 0.6162

---

## E44-cnn-medium-dice-signweight [SUCCESS] — 2026-03-16 04:27

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: True
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: -1
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E44-cnn-medium-dice-signweight_train.log

**Dev metrics:**
- Sign: F1=0.4879  IoU=0.5768  SegF1=0.9727
- Phrase: F1=0.5421  IoU=0.8963  SegF1=0.9265

**Notes:** CNN-medium + dice=1.0 + sign_weighted_loss (B=3, I=2) + no_face + velocity; testing if class weighting for rare B frames improves boundary detection past E38's 0.6162

---

## E45-tcn-noface-vel-dice [SUCCESS] — 2026-03-16 04:39

**Config:**
- acceleration: False
- arch: tcn
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 1,655,304
- Best epoch: 56
- Best val loss: None
- Elapsed: 6.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E45-tcn-noface-vel-dice_train.log

**Dev metrics:**
- Sign: F1=0.4822  IoU=0.5573  SegF1=0.9662
- Phrase: F1=0.5344  IoU=0.8915  SegF1=0.8840

**Notes:** TCN + dice=1.0 + no_face + velocity; E33 TCN got 0.193 with loss metric (all-O collapse); dice prevents this by penalizing missed sign frames; TCN excels at temporal segmentation

---

## E46-cnn-medium-dice-1h [SUCCESS] — 2026-03-16 05:40

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 400
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 100
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 203
- Best val loss: None
- Elapsed: 53.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E46-cnn-medium-dice-1h_train.log

**Dev metrics:**
- Sign: F1=0.5094  IoU=0.6093  SegF1=0.9773
- Phrase: F1=0.5596  IoU=0.7504  SegF1=0.8152

**Notes:** E38 config (CNN-medium + dice=1.0 + no_face + velocity) but 1-hour max_time; E38 was still improving at epoch 103 (rate ~0.00035/epoch); testing if 200 epochs reaches 0.65+

---

## E47-cnn-medium-dice-e200-1h [SUCCESS] — 2026-03-16 06:37

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 100
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 622,205
- Best epoch: 199
- Best val loss: None
- Elapsed: 51.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E47-cnn-medium-dice-e200-1h_train.log

**Dev metrics:**
- Sign: F1=0.5205  IoU=0.6162  SegF1=0.9786
- Phrase: F1=0.5592  IoU=0.7036  SegF1=0.7917

**Notes:** E38 config (CNN-medium + dice=1.0 + no_face + velocity) with epochs=200 max_time=1h; E38 stopped at epoch 103/200 (30-min limit, IoU=0.616); this runs the full LR schedule to epoch 200 (~58 min); expected ~0.63-0.65

---

## E48-cnn-medium-dice-face [FAILED] — 2026-03-16 07:15

**Config:**
- acceleration: False
- arch: cnn-medium
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: False
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 654,973
- Best epoch: -1
- Best val loss: None
- Elapsed: 35.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E48-cnn-medium-dice-face_train.log

**Dev metrics:**
- N/A (training failed or no checkpoint)

**Notes:** CNN-medium + dice=1.0 + WITH FACE (holistic, like paper) + velocity; all previous experiments used no_face; paper uses holistic features; testing if facial landmarks improve sign boundary detection past 0.6162

---

## BASELINE — Paper checkpoints (E1s, E4s from EMNLP 2023) [EVALUATED 2026-03-15]

Re-evaluated the two TorchScript model checkpoints from the original `main` branch
on our current dev split (DGS Corpus 3.0.0-uzh-document, 9/10 videos with poses).

**Config:**
- E1s: BiLSTM 4-layer, hidden=512, 25fps, no optical flow, no hand normalization
- E4s: BiLSTM 1-layer, hidden=512, 25fps, optical flow + 3D hand normalization
- Inference: b_threshold=60, o_threshold=50 (original paper settings)
- Evaluation script: evaluate_baseline.py

**Per-video results (Sign IoU / Seg F1):**

| Video | E1s IoU | E1s SegF1 | E4s IoU | E4s SegF1 |
|---|---|---|---|---|
| 1212402_a | 0.3791 | 0.8488 | 0.3662 | 0.8415 |
| 1212402_b | 0.5246 | 0.9838 | 0.5433 | 0.9928 |
| 1247641_a | 0.3679 | 0.9360 | 0.4702 | 0.9732 |
| 1247641_b | 0.4692 | 0.9653 | 0.4925 | 0.9710 |
| 1413236_b | 0.4944 | 0.9727 | 0.5649 | 0.9892 |
| 1414123_a | 0.3676 | 0.8631 | 0.5314 | 0.9770 |
| 1414123_b | 0.5459 | 0.9857 | 0.5585 | 0.9914 |
| 1429910_a | 0.3819 | 0.8463 | 0.5379 | 0.9922 |
| 1582841_b | 0.4279 | 0.9370 | 0.5512 | 0.9929 |

**Summary:**
- E1s: Sign IoU=0.4398, Sign Seg F1=0.9265 (n=9)
- E4s: Sign IoU=0.5129, Sign Seg F1=0.9690 (n=9)
- E38 (our best): Sign IoU=0.6162, Sign Seg F1=0.9786

**Notes:**
- Missing 1413236_a for both models (no pose file for that person/video)
- Pred segment counts are ~1.5-2× the gold counts, indicating the models use IO-fallback
  mode on these poses (B class probability stays below b_threshold=60 throughout)
- The gap from reported (IoU=0.69) to measured (IoU=0.51) is likely due to:
  (1) the DGS corpus split changed (we use 3.0.0-uzh-document, paper may have used older version)
  (2) the pose files may differ (different MediaPipe version or settings)
- Our E38 model outperforms the paper's released checkpoints on this evaluation setup

---

## E50-cnn-medium-lstm [SUCCESS] — 2026-03-16 09:20

**Config:**
- acceleration: False
- arch: cnn-medium-lstm
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 2
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 1,412,989
- Best epoch: 71
- Best val loss: None
- Elapsed: 30.1 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E50-cnn-medium-lstm_train.log

**Dev metrics:**
- Sign: F1=0.5081  IoU=0.5337  SegF1=0.8652
- Phrase: F1=0.5605  IoU=0.6303  SegF1=0.7210

**Notes:** CNN-medium spatial encoder + BiLSTM(depth=2) temporal; same features as E38 best; tests whether adding LSTM on top of CNN-medium improves over pure CNN-medium

---

## E51-cnn-medium-lstm-1h [SUCCESS] — 2026-03-16 10:16

**Config:**
- acceleration: False
- arch: cnn-medium-lstm
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 2
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 1,412,989
- Best epoch: 137
- Best val loss: None
- Elapsed: 54.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E51-cnn-medium-lstm-1h_train.log

**Dev metrics:**
- Sign: F1=0.5227  IoU=0.6164  SegF1=0.9779
- Phrase: F1=0.5564  IoU=0.7440  SegF1=0.8144

**Notes:** cnn-medium-lstm depth=2 with 1h + patience=50 (matches E38); tests CNN+LSTM ceiling vs E38 0.6162

---

## E52-cnn-medium-attn [SUCCESS] — 2026-03-16 11:01

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 158
- Best val loss: None
- Elapsed: 45.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E52-cnn-medium-attn_train.log

**Dev metrics:**
- Sign: F1=0.5263  IoU=0.6262  SegF1=0.9758
- Phrase: F1=0.5496  IoU=0.7755  SegF1=0.8571

**Notes:** cnn-medium + 4-layer pre-norm self-attention temporal encoder; nhead=4 ff=512; 2.76M params; chunked T>2048

---

## E53-cnn-medium-h512 [SUCCESS] — 2026-03-16 12:03

**Config:**
- acceleration: False
- arch: cnn-medium
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 512
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 834,173
- Best epoch: 135
- Best val loss: None
- Elapsed: 61.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E53-cnn-medium-h512_train.log

**Dev metrics:**
- Sign: F1=0.5199  IoU=0.6187  SegF1=0.9759
- Phrase: F1=0.5614  IoU=0.7656  SegF1=0.8152

**Notes:** cnn-medium with 2x wider features (h=512); tests if feature capacity is the bottleneck vs E38 0.6162

---

## E54-attn-d2 [FAILED] — 2026-03-18 04:49

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 2
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: ?
- Best epoch: -1
- Best val loss: None
- Elapsed: 0.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E54-attn-d2_train.log

**Dev metrics:**
- N/A (training failed or no checkpoint)

**Notes:** attn depth=2

---

## E54-attn-d2 [SUCCESS] — 2026-03-18 05:21

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 2
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 1,676,669
- Best epoch: 113
- Best val loss: None
- Elapsed: 31.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E54-attn-d2_train.log

**Dev metrics:**
- Sign: F1=0.5183  IoU=0.6137  SegF1=0.9745
- Phrase: F1=0.5478  IoU=0.7712  SegF1=0.8531

**Notes:** attn depth=2

---

## E55-attn-d6 [SUCCESS] — 2026-03-18 05:53

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,785,085
- Best epoch: 103
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E55-attn-d6_train.log

**Dev metrics:**
- Sign: F1=0.5103  IoU=0.5975  SegF1=0.9709
- Phrase: F1=0.5482  IoU=0.7778  SegF1=0.8387

**Notes:** attn depth=6

---

## E56-attn-d8 [SUCCESS] — 2026-03-18 06:25

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 8
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 4,839,293
- Best epoch: 99
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E56-attn-d8_train.log

**Dev metrics:**
- Sign: F1=0.5183  IoU=0.6098  SegF1=0.9763
- Phrase: F1=0.5458  IoU=0.6356  SegF1=0.6920

**Notes:** attn depth=8

---

## E57-attn-h384 [SUCCESS] — 2026-03-18 06:57

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 5,447,677
- Best epoch: 76
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E57-attn-h384_train.log

**Dev metrics:**
- Sign: F1=0.5182  IoU=0.6131  SegF1=0.9780
- Phrase: F1=0.5528  IoU=0.6471  SegF1=0.7440

**Notes:** attn hidden=384

---

## E58-attn-h512 [SUCCESS] — 2026-03-18 07:29

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 512
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 9,245,821
- Best epoch: 61
- Best val loss: None
- Elapsed: 31.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E58-attn-h512_train.log

**Dev metrics:**
- Sign: F1=0.5050  IoU=0.5842  SegF1=0.9738
- Phrase: F1=0.5628  IoU=0.7362  SegF1=0.8263

**Notes:** attn hidden=512

---

## E59-attn-pe [SUCCESS] — 2026-03-18 07:53

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 75
- Best val loss: None
- Elapsed: 23.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E59-attn-pe_train.log

**Dev metrics:**
- Sign: F1=0.5006  IoU=0.5778  SegF1=0.9715
- Phrase: F1=0.5561  IoU=0.5270  SegF1=0.5992

**Notes:** RoPE, base config

---

## E60-attn-d6-pe [SUCCESS] — 2026-03-18 08:25

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 97
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E60-attn-d6-pe_train.log

**Dev metrics:**
- Sign: F1=0.5237  IoU=0.6163  SegF1=0.9621
- Phrase: F1=0.5591  IoU=0.6002  SegF1=0.6960

**Notes:** depth=6 + RoPE

---

## E61-attn-d8-pe [SUCCESS] — 2026-03-18 08:57

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 8
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 4,827,005
- Best epoch: 91
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E61-attn-d8-pe_train.log

**Dev metrics:**
- Sign: F1=0.5064  IoU=0.6021  SegF1=0.9766
- Phrase: F1=0.5514  IoU=0.7894  SegF1=0.8631

**Notes:** depth=8 + RoPE

---

## E62-attn-nh8 [SUCCESS] — 2026-03-18 09:28

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 108
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E62-attn-nh8_train.log

**Dev metrics:**
- Sign: F1=0.5220  IoU=0.6059  SegF1=0.9752
- Phrase: F1=0.5488  IoU=0.6808  SegF1=0.7614

**Notes:** 8 attention heads

---

## E63-attn-d6-nh8-pe [SUCCESS] — 2026-03-18 10:00

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 97
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E63-attn-d6-nh8-pe_train.log

**Dev metrics:**
- Sign: F1=0.5238  IoU=0.6292  SegF1=0.9784
- Phrase: F1=0.5507  IoU=0.7681  SegF1=0.7898

**Notes:** d=6, nh=8, RoPE

---

## E64-local-attn-d4 [SUCCESS] — 2026-03-18 10:17

**Config:**
- acceleration: False
- arch: cnn-local-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,220,474
- Best epoch: 114
- Best val loss: None
- Elapsed: 16.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E64-local-attn-d4_train.log

**Dev metrics:**
- Sign: F1=0.5051  IoU=0.5804  SegF1=0.9747
- Phrase: F1=0.5402  IoU=0.7418  SegF1=0.8007

**Notes:** local-attn d=4

---

## E65-local-attn-d6 [SUCCESS] — 2026-03-18 10:36

**Config:**
- acceleration: False
- arch: cnn-local-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,273,658
- Best epoch: 115
- Best val loss: None
- Elapsed: 18.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E65-local-attn-d6_train.log

**Dev metrics:**
- Sign: F1=0.4974  IoU=0.5910  SegF1=0.9730
- Phrase: F1=0.5500  IoU=0.7555  SegF1=0.7919

**Notes:** local-attn d=6

---

## E66-local-attn-pe [SUCCESS] — 2026-03-18 10:53

**Config:**
- acceleration: False
- arch: cnn-local-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,220,474
- Best epoch: 114
- Best val loss: None
- Elapsed: 16.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E66-local-attn-pe_train.log

**Dev metrics:**
- Sign: F1=0.5051  IoU=0.5804  SegF1=0.9747
- Phrase: F1=0.5402  IoU=0.7418  SegF1=0.8007

**Notes:** local-attn + RoPE

---

## E67-attn-lr5e4 [SUCCESS] — 2026-03-18 11:22

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 78
- Best val loss: None
- Elapsed: 28.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E67-attn-lr5e4_train.log

**Dev metrics:**
- Sign: F1=0.4927  IoU=0.5801  SegF1=0.9660
- Phrase: F1=0.5441  IoU=0.7118  SegF1=0.7864

**Notes:** lr=5e-4

---

## E68-attn-lr3e4 [SUCCESS] — 2026-03-18 11:43

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.0003
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 70
- Best val loss: None
- Elapsed: 21.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E68-attn-lr3e4_train.log

**Dev metrics:**
- Sign: F1=0.4877  IoU=0.5742  SegF1=0.9711
- Phrase: F1=0.5415  IoU=0.7707  SegF1=0.7803

**Notes:** lr=3e-4

---

## E69-attn-lr2e3 [SUCCESS] — 2026-03-18 12:11

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.002
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 94
- Best val loss: None
- Elapsed: 27.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E69-attn-lr2e3_train.log

**Dev metrics:**
- Sign: F1=0.5078  IoU=0.5920  SegF1=0.9780
- Phrase: F1=0.5540  IoU=0.6570  SegF1=0.7163

**Notes:** lr=2e-3

---

## E70-attn-steps200 [SUCCESS] — 2026-03-18 12:43

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 200
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 109
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E70-attn-steps200_train.log

**Dev metrics:**
- Sign: F1=0.5223  IoU=0.6202  SegF1=0.9770
- Phrase: F1=0.5482  IoU=0.7367  SegF1=0.8197

**Notes:** 200 steps/epoch (2x)

---

## E71-attn-bs16 [SUCCESS] — 2026-03-18 13:15

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 16
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 105
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E71-attn-bs16_train.log

**Dev metrics:**
- Sign: F1=0.5036  IoU=0.6054  SegF1=0.9774
- Phrase: F1=0.5465  IoU=0.5973  SegF1=0.6927

**Notes:** batch=16

---

## E72-attn-frames2048 [SUCCESS] — 2026-03-18 13:47

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 4
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 63
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E72-attn-frames2048_train.log

**Dev metrics:**
- Sign: F1=0.5266  IoU=0.6137  SegF1=0.9761
- Phrase: F1=0.5533  IoU=0.7048  SegF1=0.7637

**Notes:** 2048-frame training context

---

## E73-attn-dice15 [SUCCESS] — 2026-03-18 14:19

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 109
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E73-attn-dice15_train.log

**Dev metrics:**
- Sign: F1=0.5205  IoU=0.6218  SegF1=0.9737
- Phrase: F1=0.5472  IoU=0.7682  SegF1=0.8519

**Notes:** dice=1.5

---

## E74-attn-dice20 [SUCCESS] — 2026-03-18 14:51

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 2.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 109
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E74-attn-dice20_train.log

**Dev metrics:**
- Sign: F1=0.5193  IoU=0.6248  SegF1=0.9769
- Phrase: F1=0.5485  IoU=0.7596  SegF1=0.8292

**Notes:** dice=2.0

---

## E75-attn-bdice [SUCCESS] — 2026-03-18 15:09

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.5
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,730,877
- Best epoch: 58
- Best val loss: None
- Elapsed: 17.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E75-attn-bdice_train.log

**Dev metrics:**
- Sign: F1=0.4532  IoU=0.5353  SegF1=0.9648
- Phrase: F1=0.5402  IoU=0.8900  SegF1=0.7895

**Notes:** B-frame dice=0.5

---

## E76-attn-accel [SUCCESS] — 2026-03-18 15:40

**Config:**
- acceleration: True
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,731,366
- Best epoch: 108
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E76-attn-accel_train.log

**Dev metrics:**
- Sign: F1=0.5096  IoU=0.6096  SegF1=0.9736
- Phrase: F1=0.5475  IoU=0.5525  SegF1=0.6363

**Notes:** velocity + acceleration

---

## E77-attn-pe-1h [SUCCESS] — 2026-03-18 16:04

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 76
- Best val loss: None
- Elapsed: 23.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E77-attn-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.5029  IoU=0.5836  SegF1=0.9719
- Phrase: F1=0.5597  IoU=0.6463  SegF1=0.7200

**Notes:** PE 1h — key baseline

---

## E78-attn-d6-pe-1h [SUCCESS] — 2026-03-18 16:50

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 142
- Best val loss: None
- Elapsed: 45.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E78-attn-d6-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.5241  IoU=0.6255  SegF1=0.9744
- Phrase: F1=0.5492  IoU=0.4995  SegF1=0.5899

**Notes:** d=6 + PE, 1h

---

## E79-attn-d8-pe-1h [SUCCESS] — 2026-03-18 17:34

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 8
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 4,827,005
- Best epoch: 131
- Best val loss: None
- Elapsed: 44.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E79-attn-d8-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.5106  IoU=0.6137  SegF1=0.9615
- Phrase: F1=0.5457  IoU=0.7936  SegF1=0.7942

**Notes:** d=8 + PE, 1h

---

## E80-attn-h384-pe-1h [SUCCESS] — 2026-03-18 18:36

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 5,438,461
- Best epoch: 148
- Best val loss: None
- Elapsed: 61.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E80-attn-h384-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.5348  IoU=0.6274  SegF1=0.9605
- Phrase: F1=0.5473  IoU=0.7528  SegF1=0.8139

**Notes:** h=384 + PE, 1h

---

## E81-attn-h512-1h [SUCCESS] — 2026-03-18 19:17

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 512
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: none
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 9,245,821
- Best epoch: 77
- Best val loss: None
- Elapsed: 40.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E81-attn-h512-1h_train.log

**Dev metrics:**
- Sign: F1=0.4806  IoU=0.5521  SegF1=0.9583
- Phrase: F1=0.5423  IoU=0.8313  SegF1=0.8363

**Notes:** h=512, 1h

---

## E82-attn-h512-pe-1h [SUCCESS] — 2026-03-18 19:48

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 512
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 9,233,533
- Best epoch: 56
- Best val loss: None
- Elapsed: 30.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E82-attn-h512-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.4755  IoU=0.5168  SegF1=0.9703
- Phrase: F1=0.5429  IoU=0.8972  SegF1=0.7834

**Notes:** h=512 + PE, 1h

---

## E83-attn-h512-d6-pe-1h [SUCCESS] — 2026-03-18 20:23

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 512
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 13,432,957
- Best epoch: 60
- Best val loss: None
- Elapsed: 34.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E83-attn-h512-d6-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.4906  IoU=0.5010  SegF1=0.8602
- Phrase: F1=0.5361  IoU=0.5678  SegF1=0.6562

**Notes:** h=512, d=6, PE, 1h

---

## E84-attn-pe-lr5e4-1h [SUCCESS] — 2026-03-18 21:03

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 129
- Best val loss: None
- Elapsed: 39.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E84-attn-pe-lr5e4-1h_train.log

**Dev metrics:**
- Sign: F1=0.5287  IoU=0.6220  SegF1=0.9790
- Phrase: F1=0.5546  IoU=0.6183  SegF1=0.7057

**Notes:** PE + lr=5e-4, 1h

---

## E85-attn-d6-pe-lr5e4-1h [SUCCESS] — 2026-03-18 21:25

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 62
- Best val loss: None
- Elapsed: 21.1 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E85-attn-d6-pe-lr5e4-1h_train.log

**Dev metrics:**
- Sign: F1=0.4705  IoU=0.5655  SegF1=0.9673
- Phrase: F1=0.5457  IoU=0.6653  SegF1=0.7721

**Notes:** d=6, PE, lr=5e-4, 1h

---

## E86-attn-pe-steps200-1h [SUCCESS] — 2026-03-18 22:01

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 200
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 119
- Best val loss: None
- Elapsed: 36.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E86-attn-pe-steps200-1h_train.log

**Dev metrics:**
- Sign: F1=0.5125  IoU=0.5854  SegF1=0.9740
- Phrase: F1=0.5650  IoU=0.6432  SegF1=0.7285

**Notes:** PE + 200 steps, 1h

---

## E87-attn-d6-pe-steps200-1h [SUCCESS] — 2026-03-18 22:35

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 200
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 103
- Best val loss: None
- Elapsed: 33.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E87-attn-d6-pe-steps200-1h_train.log

**Dev metrics:**
- Sign: F1=0.5145  IoU=0.6127  SegF1=0.9722
- Phrase: F1=0.5508  IoU=0.7484  SegF1=0.8194

**Notes:** d=6, PE, 200 steps, 1h

---

## E88-attn-pe-frames2048-1h [SUCCESS] — 2026-03-18 23:20

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 4
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 86
- Best val loss: None
- Elapsed: 44.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E88-attn-pe-frames2048-1h_train.log

**Dev metrics:**
- Sign: F1=0.5174  IoU=0.6018  SegF1=0.9740
- Phrase: F1=0.5467  IoU=0.7342  SegF1=0.7891

**Notes:** PE + 2048-frame context, 1h

---

## E89-local-attn-d6-pe-1h [SUCCESS] — 2026-03-18 23:40

**Config:**
- acceleration: False
- arch: cnn-local-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,273,658
- Best epoch: 115
- Best val loss: None
- Elapsed: 18.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E89-local-attn-d6-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.4974  IoU=0.5910  SegF1=0.9730
- Phrase: F1=0.5500  IoU=0.7555  SegF1=0.7919

**Notes:** local-attn d=6, PE, 1h

---

## E90-local-attn-h384-pe-1h [SUCCESS] — 2026-03-18 23:54

**Config:**
- acceleration: False
- arch: cnn-local-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 4,854,202
- Best epoch: 81
- Best val loss: None
- Elapsed: 13.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E90-local-attn-h384-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.4748  IoU=0.5608  SegF1=0.9660
- Phrase: F1=0.5536  IoU=0.6898  SegF1=0.7340

**Notes:** local-attn h=384, PE, 1h

---

## E91-attn-d6-nh8-pe-1h [SUCCESS] — 2026-03-19 00:34

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 122
- Best val loss: None
- Elapsed: 39.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E91-attn-d6-nh8-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.5193  IoU=0.6192  SegF1=0.9735
- Phrase: F1=0.5477  IoU=0.7573  SegF1=0.8027

**Notes:** d=6, nh=8, PE, 1h

---

## E92-attn-h512-d6-nh8-pe-1h [SUCCESS] — 2026-03-19 01:12

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 512
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 13,432,957
- Best epoch: 58
- Best val loss: None
- Elapsed: 37.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E92-attn-h512-d6-nh8-pe-1h_train.log

**Dev metrics:**
- Sign: F1=0.4809  IoU=0.5637  SegF1=0.9625
- Phrase: F1=0.5287  IoU=0.8914  SegF1=0.8819

**Notes:** h=512, d=6, nh=8, PE, 1h

---

## E93-attn-pe-dice15-1h [SUCCESS] — 2026-03-19 01:51

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 129
- Best val loss: None
- Elapsed: 38.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E93-attn-pe-dice15-1h_train.log

**Dev metrics:**
- Sign: F1=0.5276  IoU=0.6290  SegF1=0.9786
- Phrase: F1=0.5846  IoU=0.7097  SegF1=0.7975

**Notes:** PE + dice=1.5, 1h

---

## E94-attn-pe-bdice-1h [SUCCESS] — 2026-03-19 02:07

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.5
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 50
- Best val loss: None
- Elapsed: 16.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E94-attn-pe-bdice-1h_train.log

**Dev metrics:**
- Sign: F1=0.3837  IoU=0.4633  SegF1=0.9006
- Phrase: F1=0.5141  IoU=0.8539  SegF1=0.7696

**Notes:** PE + B-frame dice, 1h

---

## E95-attn-pe-ff4-1h [SUCCESS] — 2026-03-19 02:52

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 4
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,357
- Best epoch: 143
- Best val loss: None
- Elapsed: 43.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E95-attn-pe-ff4-1h_train.log

**Dev metrics:**
- Sign: F1=0.5300  IoU=0.6167  SegF1=0.9717
- Phrase: F1=0.5547  IoU=0.5817  SegF1=0.6842

**Notes:** PE + ff_mult=4, 1h

---

## E96-attn-pe-accel-1h [SUCCESS] — 2026-03-19 03:34

**Config:**
- acceleration: True
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,725,222
- Best epoch: 136
- Best val loss: None
- Elapsed: 41.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E96-attn-pe-accel-1h_train.log

**Dev metrics:**
- Sign: F1=0.5327  IoU=0.6136  SegF1=0.9777
- Phrase: F1=0.5496  IoU=0.6595  SegF1=0.7313

**Notes:** PE + velocity + accel, 1h

---

## E97-attn-h384-d6-nh8-pe-lr5e4-1h [SUCCESS] — 2026-03-19 04:36

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 138
- Best val loss: None
- Elapsed: 61.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E97-attn-h384-d6-nh8-pe-lr5e4-1h_train.log

**Dev metrics:**
- Sign: F1=0.5332  IoU=0.6289  SegF1=0.9752
- Phrase: F1=0.5546  IoU=0.6773  SegF1=0.7612

**Notes:** h=384, d=6, nh=8, PE, lr=5e-4, 1h

---

## E98-attn-pe-bs16-1h [SUCCESS] — 2026-03-19 05:18

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 16
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 135
- Best val loss: None
- Elapsed: 42.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E98-attn-pe-bs16-1h_train.log

**Dev metrics:**
- Sign: F1=0.5198  IoU=0.5847  SegF1=0.9679
- Phrase: F1=0.5449  IoU=0.6192  SegF1=0.6977

**Notes:** PE + batch=16, 1h

---

## E99-attn-pe-drop05-1h [SUCCESS] — 2026-03-19 05:28

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.05
- attn_ff_mult: 2
- attn_nhead: 4
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 4
- epochs: 200
- focal_gamma: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 2,724,733
- Best epoch: 27
- Best val loss: None
- Elapsed: 9.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E99-attn-pe-drop05-1h_train.log

**Dev metrics:**
- Sign: F1=0.4978  IoU=0.5143  SegF1=0.8637
- Phrase: F1=0.5591  IoU=0.7393  SegF1=0.7973

**Notes:** PE + dropout=0.05, 1h

---

## E101-fps-aug-1h [SUCCESS] — 2026-03-19 06:49

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 188
- Best val loss: None
- Elapsed: 60.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E101-fps-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5049  IoU=0.3931  SegF1=0.7267
- Phrase: F1=0.5317  IoU=0.3917  SegF1=0.4581

**Dev metrics @ 25fps:**
- Sign: F1=0.4956  IoU=0.1914  SegF1=0.4100
- Phrase: F1=0.5285  IoU=0.4773  SegF1=0.5551

**Notes:** fps aug 25-50fps, time-based timestamps

---

## E102-fps-aug-dropout-1h [SUCCESS] — 2026-03-19 07:29

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 121
- Best val loss: None
- Elapsed: 39.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E102-fps-aug-dropout-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5041  IoU=0.4314  SegF1=0.7495
- Phrase: F1=0.5325  IoU=0.4360  SegF1=0.5189

**Dev metrics @ 25fps:**
- Sign: F1=0.4942  IoU=0.2908  SegF1=0.5327
- Phrase: F1=0.5321  IoU=0.5829  SegF1=0.6905

**Notes:** fps aug + 0-15% frame dropout

---

## E103-robust-dice15-1h [SUCCESS] — 2026-03-19 08:23

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 153
- Best val loss: None
- Elapsed: 53.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E103-robust-dice15-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5041  IoU=0.5404  SegF1=0.9456
- Phrase: F1=0.5321  IoU=0.2607  SegF1=0.3373

**Dev metrics @ 25fps:**
- Sign: F1=0.4933  IoU=0.4509  SegF1=0.8504
- Phrase: F1=0.5353  IoU=0.5668  SegF1=0.6777

**Notes:** fps aug + dropout + dice=1.5

---

## E104-dice15-1h [SUCCESS] — 2026-03-19 09:00

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: False
- frame_dropout: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 102
- Best val loss: None
- Elapsed: 36.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E104-dice15-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5154  IoU=0.6150  SegF1=0.9705
- Phrase: F1=0.5494  IoU=0.8082  SegF1=0.8291

**Dev metrics @ 25fps:**
- Sign: F1=0.4854  IoU=0.4567  SegF1=0.7551
- Phrase: F1=0.5290  IoU=0.6027  SegF1=0.6772

**Notes:** d=6 nh=8 dice=1.5, no robustness aug

---

## E105-face-1h [FAILED] — 2026-03-19 10:05

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: False
- frame_dropout: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: False
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,808,637
- Best epoch: -1
- Best val loss: None
- Elapsed: 65.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E105-face-1h_train.log

**Dev metrics @ 50fps:**
- N/A (training failed or no checkpoint)

**Dev metrics @ 25fps:**
- not run

**Notes:** face enabled d=6 nh=8, no robustness aug

---

## E106-fps-aug-fixed-1h [SUCCESS] — 2026-03-19 11:24

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 131
- Best val loss: None
- Elapsed: 42.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E106-fps-aug-fixed-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5062  IoU=0.6066  SegF1=0.9719
- Phrase: F1=0.5789  IoU=0.7865  SegF1=0.8598

**Dev metrics @ 25fps:**
- Sign: F1=0.4715  IoU=0.5710  SegF1=0.9741
- Phrase: F1=0.5404  IoU=0.8101  SegF1=0.8528

**Notes:** fps_aug fixed: 50fps-equiv RoPE + vel in disp/50fps-frame

---

## E107-fps-aug-dropout-fixed-1h [SUCCESS] — 2026-03-19 12:00

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 110
- Best val loss: None
- Elapsed: 36.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E107-fps-aug-dropout-fixed-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5090  IoU=0.6055  SegF1=0.9716
- Phrase: F1=0.5448  IoU=0.7992  SegF1=0.8469

**Dev metrics @ 25fps:**
- Sign: F1=0.4684  IoU=0.4891  SegF1=0.8560
- Phrase: F1=0.5136  IoU=0.7731  SegF1=0.8227

**Notes:** fps_aug + dropout=0.15, fixed scales

---

## E108-fps-aug-dice15-fixed-1h [SUCCESS] — 2026-03-19 12:27

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 76
- Best val loss: None
- Elapsed: 25.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E108-fps-aug-dice15-fixed-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4859  IoU=0.5939  SegF1=0.9803
- Phrase: F1=0.5461  IoU=0.6605  SegF1=0.7526

**Dev metrics @ 25fps:**
- Sign: F1=0.4550  IoU=0.5009  SegF1=0.8618
- Phrase: F1=0.5312  IoU=0.7041  SegF1=0.7730

**Notes:** fps_aug + dice=1.5, fixed scales

---

## E109-fps-aug-dropout-dice15-1h [SUCCESS] — 2026-03-19 13:06

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 119
- Best val loss: None
- Elapsed: 38.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E109-fps-aug-dropout-dice15-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5186  IoU=0.6137  SegF1=0.9730
- Phrase: F1=0.5543  IoU=0.6922  SegF1=0.6876

**Dev metrics @ 25fps:**
- Sign: F1=0.4908  IoU=0.5830  SegF1=0.9723
- Phrase: F1=0.5409  IoU=0.7567  SegF1=0.8286

**Notes:** fps_aug + dropout=0.15 + dice=1.5, fixed scales

---

## E110-fps-aug-face-1h [FAILED] — 2026-03-19 14:11

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: False
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,808,637
- Best epoch: -1
- Best val loss: None
- Elapsed: 65.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E110-fps-aug-face-1h_train.log

**Dev metrics @ 50fps:**
- N/A (training failed or no checkpoint)

**Dev metrics @ 25fps:**
- not run

**Notes:** fps_aug + face enabled, fixed scales

---

## E111-full-aug-1h [SUCCESS] — 2026-03-19 18:38

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 93
- Best val loss: None
- Elapsed: 30.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E111-full-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5066  IoU=0.5794  SegF1=0.9695
- Phrase: F1=0.5434  IoU=0.4998  SegF1=0.4774

**Dev metrics @ 25fps:**
- Sign: F1=0.4762  IoU=0.3069  SegF1=0.5446
- Phrase: F1=0.5441  IoU=0.7585  SegF1=0.8316

**Notes:** E109+tempo: fps_aug+dropout=0.15+dice=1.5+tempo_stretch

---

## E112-h384-full-aug-1h [SUCCESS] — 2026-03-19 19:08

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 64
- Best val loss: None
- Elapsed: 30.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E112-h384-full-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4798  IoU=0.5663  SegF1=0.9758
- Phrase: F1=0.5482  IoU=0.9052  SegF1=0.8694

**Dev metrics @ 25fps:**
- Sign: F1=0.4611  IoU=0.5506  SegF1=0.9743
- Phrase: F1=0.5252  IoU=0.8923  SegF1=0.8925

**Notes:** h=384 fps_aug+dropout+dice=1.5+tempo

---

## E113-h384-fps-dice15-1h [SUCCESS] — 2026-03-19 20:01

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 115
- Best val loss: None
- Elapsed: 51.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E113-h384-fps-dice15-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5138  IoU=0.6051  SegF1=0.9802
- Phrase: F1=0.5478  IoU=0.7564  SegF1=0.8069

**Dev metrics @ 25fps:**
- Sign: F1=0.4907  IoU=0.3847  SegF1=0.6547
- Phrase: F1=0.5240  IoU=0.6271  SegF1=0.6924

**Notes:** h=384 fps_aug+dice=1.5, no dropout, tempo

---

## E114-dropout05-1h [SUCCESS] — 2026-03-19 20:30

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.05
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 85
- Best val loss: None
- Elapsed: 28.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E114-dropout05-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5017  IoU=0.5974  SegF1=0.9721
- Phrase: F1=0.5435  IoU=0.8614  SegF1=0.8351

**Dev metrics @ 25fps:**
- Sign: F1=0.4673  IoU=0.5655  SegF1=0.9752
- Phrase: F1=0.5112  IoU=0.7776  SegF1=0.8504

**Notes:** fps_aug+dropout=0.05+dice=1.5+tempo

---

## E115-dropout30-1h [SUCCESS] — 2026-03-19 21:12

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.3
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 131
- Best val loss: None
- Elapsed: 42.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E115-dropout30-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5201  IoU=0.6132  SegF1=0.9761
- Phrase: F1=0.5686  IoU=0.8003  SegF1=0.8195

**Dev metrics @ 25fps:**
- Sign: F1=0.4890  IoU=0.5096  SegF1=0.8598
- Phrase: F1=0.5390  IoU=0.7577  SegF1=0.8274

**Notes:** fps_aug+dropout=0.30+dice=1.5+tempo

---

## E116-lr5e4-full-aug-1h [SUCCESS] — 2026-03-19 21:34

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 62
- Best val loss: None
- Elapsed: 20.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E116-lr5e4-full-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4621  IoU=0.5659  SegF1=0.9671
- Phrase: F1=0.5441  IoU=0.8552  SegF1=0.8796

**Dev metrics @ 25fps:**
- Sign: F1=0.4422  IoU=0.5533  SegF1=0.9603
- Phrase: F1=0.5292  IoU=0.7751  SegF1=0.7946

**Notes:** fps_aug+dropout+dice=1.5+lr=5e-4+tempo

---

## E117-h384-lr5e4-full-aug-1h [SUCCESS] — 2026-03-19 22:33

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 129
- Best val loss: None
- Elapsed: 58.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E117-h384-lr5e4-full-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5166  IoU=0.6019  SegF1=0.9622
- Phrase: F1=0.5642  IoU=0.6856  SegF1=0.7734

**Dev metrics @ 25fps:**
- Sign: F1=0.4898  IoU=0.4404  SegF1=0.7509
- Phrase: F1=0.5478  IoU=0.7940  SegF1=0.8500

**Notes:** h=384 fps_aug+dropout+dice=1.5+lr=5e-4+tempo

---

## E118-steps200-full-aug-1h [SUCCESS] — 2026-03-19 23:04

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 200
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 93
- Best val loss: None
- Elapsed: 30.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E118-steps200-full-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5103  IoU=0.5886  SegF1=0.9755
- Phrase: F1=0.5566  IoU=0.6186  SegF1=0.6603

**Dev metrics @ 25fps:**
- Sign: F1=0.4772  IoU=0.4277  SegF1=0.7488
- Phrase: F1=0.5450  IoU=0.7631  SegF1=0.8273

**Notes:** fps_aug+dropout+dice=1.5+steps=200+tempo

---

## E119-h384-steps200-full-aug-1h [SUCCESS] — 2026-03-19 23:54

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 200
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 109
- Best val loss: None
- Elapsed: 49.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E119-h384-steps200-full-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4708  IoU=0.5762  SegF1=0.9707
- Phrase: F1=0.5699  IoU=0.8122  SegF1=0.8488

**Dev metrics @ 25fps:**
- Sign: F1=0.4401  IoU=0.5440  SegF1=0.9709
- Phrase: F1=0.5305  IoU=0.7136  SegF1=0.7874

**Notes:** h=384 fps_aug+dropout+dice=1.5+steps=200+tempo

---

## E120-face-full-aug-1h [FAILED] — 2026-03-20 00:59

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: False
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,808,637
- Best epoch: -1
- Best val loss: None
- Elapsed: 65.0 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E120-face-full-aug-1h_train.log

**Dev metrics @ 50fps:**
- N/A (training failed or no checkpoint)

**Dev metrics @ 25fps:**
- not run

**Notes:** face+fps_aug+dropout+dice=1.5+tempo

---

## E121-full-aug-3h [SUCCESS] — 2026-03-20 04:55

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 87
- Best val loss: None
- Elapsed: 32.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E121-full-aug-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4984  IoU=0.5887  SegF1=0.9756
- Phrase: F1=0.5548  IoU=0.5372  SegF1=0.5852

**Dev metrics @ 25fps:**
- Sign: F1=0.4621  IoU=0.5646  SegF1=0.9735
- Phrase: F1=0.5393  IoU=0.6788  SegF1=0.7642

**Notes:** E109 config 3h: fps_aug+dropout=0.15+dice=1.5+tempo(fixed)

---

## E122-h384-full-aug-3h [SUCCESS] — 2026-03-20 05:28

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 68
- Best val loss: None
- Elapsed: 32.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E122-h384-full-aug-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4851  IoU=0.5664  SegF1=0.9755
- Phrase: F1=0.5479  IoU=0.7999  SegF1=0.8001

**Dev metrics @ 25fps:**
- Sign: F1=0.4612  IoU=0.5510  SegF1=0.9764
- Phrase: F1=0.5275  IoU=0.8843  SegF1=0.8649

**Notes:** h=384 fps_aug+dropout+dice=1.5+tempo(fixed) 3h

---

## E123-h384-lr5e4-full-aug-3h [SUCCESS] — 2026-03-20 06:16

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 90
- Best val loss: None
- Elapsed: 47.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E123-h384-lr5e4-full-aug-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5042  IoU=0.6018  SegF1=0.9698
- Phrase: F1=0.5438  IoU=0.9012  SegF1=0.8853

**Dev metrics @ 25fps:**
- Sign: F1=0.4734  IoU=0.5601  SegF1=0.9726
- Phrase: F1=0.5014  IoU=0.7754  SegF1=0.7546

**Notes:** h=384 lr=5e-4 fps_aug+dropout+dice=1.5+tempo(fixed) 3h

---

## E124-full-aug-tempo-fixed-1h [SUCCESS] — 2026-03-20 06:47

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 256
- label_smoothing: 0.0
- learning_rate: 0.001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 3,775,869
- Best epoch: 91
- Best val loss: None
- Elapsed: 30.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E124-full-aug-tempo-fixed-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5113  IoU=0.5606  SegF1=0.9788
- Phrase: F1=0.5570  IoU=0.4489  SegF1=0.5160

**Dev metrics @ 25fps:**
- Sign: F1=0.4906  IoU=0.4180  SegF1=0.7548
- Phrase: F1=0.5352  IoU=0.6784  SegF1=0.7449

**Notes:** E111 re-run: tempo bug fixed, validate 25fps recovery

---

## E125-fresh-30m [SUCCESS] — 2026-03-20 11:25

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 137
- Best val loss: None
- Elapsed: 61.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E125-fresh-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5169  IoU=0.6135  SegF1=0.9763
- Phrase: F1=0.5498  IoU=0.8016  SegF1=0.8533

**Dev metrics @ 25fps:**
- Sign: F1=0.4945  IoU=0.3833  SegF1=0.6532
- Phrase: F1=0.5245  IoU=0.6944  SegF1=0.7914

**Notes:** fresh E123 config 30min baseline

---

## E126-ft-lr1e4-cosine [FAILED] — 2026-03-20 11:26

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: ?
- Best epoch: -1
- Best val loss: None
- Elapsed: 1.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E126-ft-lr1e4-cosine_train.log

**Dev metrics @ 50fps:**
- N/A (training failed or no checkpoint)

**Dev metrics @ 25fps:**
- not run

**Notes:** ft E123, lr=1e-4, cosine

---

## E127-ft-lr5e5-cosine [FAILED] — 2026-03-20 11:27

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: ?
- Best epoch: -1
- Best val loss: None
- Elapsed: 0.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E127-ft-lr5e5-cosine_train.log

**Dev metrics @ 50fps:**
- N/A (training failed or no checkpoint)

**Dev metrics @ 25fps:**
- not run

**Notes:** ft E123, lr=5e-5, cosine

---

## E128-ft-lr1e5-cosine [FAILED] — 2026-03-20 11:28

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 1e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: ?
- Best epoch: -1
- Best val loss: None
- Elapsed: 0.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E128-ft-lr1e5-cosine_train.log

**Dev metrics @ 50fps:**
- N/A (training failed or no checkpoint)

**Dev metrics @ 25fps:**
- not run

**Notes:** ft E123, lr=1e-5, cosine

---

## E129-ft-lr5e5-constant [FAILED] — 2026-03-20 11:28

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: constant
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: ?
- Best epoch: -1
- Best val loss: None
- Elapsed: 0.8 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E129-ft-lr5e5-constant_train.log

**Dev metrics @ 50fps:**
- N/A (training failed or no checkpoint)

**Dev metrics @ 25fps:**
- not run

**Notes:** ft E123, lr=5e-5, constant LR

---

## E126-ft-lr1e4-cosine [SUCCESS] — 2026-03-20 11:54

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 54
- Best val loss: None
- Elapsed: 24.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E126-ft-lr1e4-cosine_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5114  IoU=0.5978  SegF1=0.9716
- Phrase: F1=0.5456  IoU=0.7751  SegF1=0.8538

**Dev metrics @ 25fps:**
- Sign: F1=0.4821  IoU=0.4362  SegF1=0.7487
- Phrase: F1=0.5379  IoU=0.7430  SegF1=0.7906

**Notes:** ft E123, lr=1e-4, cosine

---

## E127-ft-lr5e5-cosine [SUCCESS] — 2026-03-20 12:18

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 50
- Best val loss: None
- Elapsed: 23.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E127-ft-lr5e5-cosine_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5115  IoU=0.5916  SegF1=0.9693
- Phrase: F1=0.5468  IoU=0.8485  SegF1=0.8976

**Dev metrics @ 25fps:**
- Sign: F1=0.4840  IoU=0.3660  SegF1=0.6437
- Phrase: F1=0.5298  IoU=0.6795  SegF1=0.7137

**Notes:** ft E123, lr=5e-5, cosine

---

## E128-ft-lr1e5-cosine [SUCCESS] — 2026-03-20 12:42

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 1e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 50
- Best val loss: None
- Elapsed: 23.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E128-ft-lr1e5-cosine_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5111  IoU=0.6036  SegF1=0.9721
- Phrase: F1=0.5488  IoU=0.9060  SegF1=0.9159

**Dev metrics @ 25fps:**
- Sign: F1=0.4798  IoU=0.5567  SegF1=0.9720
- Phrase: F1=0.5208  IoU=0.7860  SegF1=0.8140

**Notes:** ft E123, lr=1e-5, cosine

---

## E129-ft-lr5e5-constant [SUCCESS] — 2026-03-20 13:06

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: constant
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 50
- Best val loss: None
- Elapsed: 23.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E129-ft-lr5e5-constant_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5115  IoU=0.5911  SegF1=0.9692
- Phrase: F1=0.5467  IoU=0.8483  SegF1=0.8996

**Dev metrics @ 25fps:**
- Sign: F1=0.4839  IoU=0.3657  SegF1=0.6436
- Phrase: F1=0.5298  IoU=0.7893  SegF1=0.8247

**Notes:** ft E123, lr=5e-5, constant LR

---

## E130-ft-noaug-cosine [SUCCESS] — 2026-03-20 13:36

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: False
- frame_dropout: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 63
- Best val loss: None
- Elapsed: 28.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E130-ft-noaug-cosine_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5240  IoU=0.5451  SegF1=0.8667
- Phrase: F1=0.5475  IoU=0.6913  SegF1=0.7664

**Dev metrics @ 25fps:**
- Sign: F1=0.4993  IoU=0.4631  SegF1=0.7608
- Phrase: F1=0.5376  IoU=0.5920  SegF1=0.6782

**Notes:** ft E123, lr=5e-5, cosine, no fps_aug/dropout

---

## E131-ft-nodropout-cosine [SUCCESS] — 2026-03-20 14:08

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 69
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E131-ft-nodropout-cosine_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5209  IoU=0.5361  SegF1=0.8645
- Phrase: F1=0.5467  IoU=0.7480  SegF1=0.8343

**Dev metrics @ 25fps:**
- Sign: F1=0.4964  IoU=0.3769  SegF1=0.6534
- Phrase: F1=0.5377  IoU=0.6902  SegF1=0.7850

**Notes:** ft E123, lr=5e-5, cosine, no frame dropout

---

## E132-ft-steps200-cosine [SUCCESS] — 2026-03-20 14:32

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 200
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 50
- Best val loss: None
- Elapsed: 23.3 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E132-ft-steps200-cosine_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5115  IoU=0.5916  SegF1=0.9695
- Phrase: F1=0.5469  IoU=0.8486  SegF1=0.8995

**Dev metrics @ 25fps:**
- Sign: F1=0.4840  IoU=0.3658  SegF1=0.6428
- Phrase: F1=0.5297  IoU=0.6795  SegF1=0.7137

**Notes:** ft E123, lr=5e-5, cosine, steps=200

---

## E133-ft-patience100-cosine [SUCCESS] — 2026-03-20 15:04

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 100
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 70
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E133-ft-patience100-cosine_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5115  IoU=0.5918  SegF1=0.9691
- Phrase: F1=0.5468  IoU=0.8486  SegF1=0.8979

**Dev metrics @ 25fps:**
- Sign: F1=0.4840  IoU=0.3658  SegF1=0.6425
- Phrase: F1=0.5301  IoU=0.6795  SegF1=0.7125

**Notes:** ft E123, lr=5e-5, cosine, patience=100

---

## E134-ft-dice10-cosine [SUCCESS] — 2026-03-20 15:31

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- dice_loss_weight: 1.0
- encoder_depth: 6
- epochs: 200
- finetune_from: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/models/E123-h384-lr5e4-full-aug-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 5e-05
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- optimizer: cosine
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 57
- Best val loss: None
- Elapsed: 26.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E134-ft-dice10-cosine_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5180  IoU=0.5950  SegF1=0.9739
- Phrase: F1=0.5460  IoU=0.7066  SegF1=0.7912

**Dev metrics @ 25fps:**
- Sign: F1=0.4957  IoU=0.3666  SegF1=0.6405
- Phrase: F1=0.5437  IoU=0.6927  SegF1=0.7679

**Notes:** ft E123, lr=5e-5, cosine, dice=1.0

---

## E135-body-dropout-1h [SUCCESS] — 2026-03-20 19:57

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- body_part_dropout: 0.3
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 139
- Best val loss: None
- Elapsed: 61.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E135-body-dropout-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5117  IoU=0.6060  SegF1=0.9678
- Phrase: F1=0.5476  IoU=0.7477  SegF1=0.7997

**Dev metrics @ 25fps:**
- Sign: F1=0.4827  IoU=0.4424  SegF1=0.7537
- Phrase: F1=0.5455  IoU=0.5942  SegF1=0.6766

**Notes:** body_part_dropout=0.3 (each hand independently 30%)

---

## E136-speed-aug-1h [SUCCESS] — 2026-03-20 20:23

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.0
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: True
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 54
- Best val loss: None
- Elapsed: 25.1 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E136-speed-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.3675  IoU=0.4831  SegF1=0.8618
- Phrase: F1=0.5258  IoU=0.8501  SegF1=0.7538

**Dev metrics @ 25fps:**
- Sign: F1=0.3481  IoU=0.4646  SegF1=0.8696
- Phrase: F1=0.4778  IoU=0.8211  SegF1=0.7461

**Notes:** speed_aug=0.75-1.25x signing speed (frame_times_ms scaling)

---

## E137-body-speed-aug-1h [SUCCESS] — 2026-03-20 20:52

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.3
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:01:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: True
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 62
- Best val loss: None
- Elapsed: 28.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E137-body-speed-aug-1h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.3657  IoU=0.4743  SegF1=0.8944
- Phrase: F1=0.4938  IoU=0.8178  SegF1=0.8988

**Dev metrics @ 25fps:**
- Sign: F1=0.3395  IoU=0.4511  SegF1=0.8283
- Phrase: F1=0.4515  IoU=0.7758  SegF1=0.8850

**Notes:** body_part_dropout=0.3 + speed_aug combined

---

## E138-frames2048-3h [SUCCESS] — 2026-03-20 22:19

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 4
- batch_size_schedule: False
- body_part_dropout: 0.0
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 114
- Best val loss: None
- Elapsed: 85.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E138-frames2048-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5253  IoU=0.6223  SegF1=0.9726
- Phrase: F1=0.5499  IoU=0.5657  SegF1=0.6424

**Dev metrics @ 25fps:**
- Sign: F1=0.5001  IoU=0.4539  SegF1=0.7597
- Phrase: F1=0.5365  IoU=0.6947  SegF1=0.7758

**Notes:** num_frames=2048 (full global attn), batch=4

---

## E139-curriculum-3h [SUCCESS] — 2026-03-20 23:15

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 4
- batch_size_schedule: False
- body_part_dropout: 0.0
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: 2048
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 125
- Best val loss: None
- Elapsed: 55.1 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E139-curriculum-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5129  IoU=0.6109  SegF1=0.9751
- Phrase: F1=0.5566  IoU=0.6051  SegF1=0.6877

**Dev metrics @ 25fps:**
- Sign: F1=0.4735  IoU=0.5061  SegF1=0.8588
- Phrase: F1=0.5362  IoU=0.6731  SegF1=0.7607

**Notes:** curriculum: num_frames 1024→2048 linear, batch=4

---

## E140-frames2048-body-speed-3h [SUCCESS] — 2026-03-21 00:10

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 4
- batch_size_schedule: False
- body_part_dropout: 0.3
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: True
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 72
- Best val loss: None
- Elapsed: 55.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E140-frames2048-body-speed-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.3649  IoU=0.4795  SegF1=0.8212
- Phrase: F1=0.5223  IoU=0.6974  SegF1=0.7309

**Dev metrics @ 25fps:**
- Sign: F1=0.3347  IoU=0.4022  SegF1=0.6122
- Phrase: F1=0.4609  IoU=0.5959  SegF1=0.6678

**Notes:** num_frames=2048 + body_part_dropout=0.3 + speed_aug, batch=4

---

## E141-curriculum-body-speed-3h [SUCCESS] — 2026-03-21 00:35

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 4
- batch_size_schedule: False
- body_part_dropout: 0.3
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: 2048
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: True
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 52
- Best val loss: None
- Elapsed: 23.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E141-curriculum-body-speed-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.3853  IoU=0.4563  SegF1=0.8784
- Phrase: F1=0.4959  IoU=0.8284  SegF1=0.8373

**Dev metrics @ 25fps:**
- Sign: F1=0.3622  IoU=0.4502  SegF1=0.8874
- Phrase: F1=0.4615  IoU=0.7899  SegF1=0.7951

**Notes:** curriculum 1024→2048 + body_part_dropout=0.3 + speed_aug, batch=4

---

## E143-frames2048-batch8-3h [SUCCESS] — 2026-03-21 03:43

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.0
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 108
- Best val loss: None
- Elapsed: 86.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E143-frames2048-batch8-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5285  IoU=0.6244  SegF1=0.9796
- Phrase: F1=0.5513  IoU=0.8607  SegF1=0.8451

**Dev metrics @ 25fps:**
- Sign: F1=0.5051  IoU=0.1325  SegF1=0.2654
- Phrase: F1=0.5270  IoU=0.8582  SegF1=0.8816

**Notes:** 2048 frames batch=8 (69 steps/ep, 2x context vs E123)

---

## E144-curriculum-batch8-3h [SUCCESS] — 2026-03-21 04:24

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.0
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: 2048
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 90
- Best val loss: None
- Elapsed: 40.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E144-curriculum-batch8-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5028  IoU=0.5924  SegF1=0.9696
- Phrase: F1=0.5453  IoU=0.8681  SegF1=0.9096

**Dev metrics @ 25fps:**
- Sign: F1=0.4720  IoU=0.4890  SegF1=0.8593
- Phrase: F1=0.5211  IoU=0.7594  SegF1=0.7637

**Notes:** curriculum 1024→2048 batch=8 throughout (no accum)

---

## E145-body-dropout01-3h [SUCCESS] — 2026-03-21 05:03

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 84
- Best val loss: None
- Elapsed: 38.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E145-body-dropout01-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4963  IoU=0.5947  SegF1=0.9736
- Phrase: F1=0.5462  IoU=0.9072  SegF1=0.9128

**Dev metrics @ 25fps:**
- Sign: F1=0.4613  IoU=0.5693  SegF1=0.9773
- Phrase: F1=0.5219  IoU=0.8801  SegF1=0.8620

**Notes:** body_part_dropout=0.1 (10% per hand), 1024 frames batch=8

---

## E146-e123-6h [SUCCESS] — 2026-03-21 06:20

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.0
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:06:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 175
- Best val loss: None
- Elapsed: 76.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E146-e123-6h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5291  IoU=0.6232  SegF1=0.9758
- Phrase: F1=0.5633  IoU=0.6981  SegF1=0.7484

**Dev metrics @ 25fps:**
- Sign: F1=0.5035  IoU=0.4528  SegF1=0.7610
- Phrase: F1=0.5325  IoU=0.6924  SegF1=0.7683

**Notes:** E123 config extended to 6h — does more time beat E138/E143?

---

## E147-frames2048-batch8-dropout01-3h [SUCCESS] — 2026-03-21 08:00

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 123
- Best val loss: None
- Elapsed: 98.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E147-frames2048-batch8-dropout01-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5278  IoU=0.6165  SegF1=0.9756
- Phrase: F1=0.5548  IoU=0.7393  SegF1=0.7536

**Dev metrics @ 25fps:**
- Sign: F1=0.4972  IoU=0.5727  SegF1=0.9786
- Phrase: F1=0.5281  IoU=0.7344  SegF1=0.7872

**Notes:** 2048 frames batch=8 + body_part_dropout=0.1

---

## E148-frames2048-batch8-dropout01-6h [SUCCESS] — 2026-03-21 09:21

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:06:00:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 98
- Best val loss: None
- Elapsed: 79.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E148-frames2048-batch8-dropout01-6h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5194  IoU=0.6058  SegF1=0.9795
- Phrase: F1=0.5423  IoU=0.7465  SegF1=0.7928

**Dev metrics @ 25fps:**
- Sign: F1=0.4938  IoU=0.3736  SegF1=0.6492
- Phrase: F1=0.5419  IoU=0.7663  SegF1=0.8515

**Notes:** E147 config extended to 6h: 2048fr+bs8+drop=0.1 (Case A: S50=0.617>=0.615 AND S25=0.573>=0.540)

---

## E149-finetune-e145-lr1e4-3h [SUCCESS] — 2026-03-21 10:10

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: models/E145-body-dropout01-3h/best.ckpt
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0001
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 85
- Best val loss: None
- Elapsed: 38.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E149-finetune-e145-lr1e4-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5167  IoU=0.5952  SegF1=0.9713
- Phrase: F1=0.5431  IoU=0.6930  SegF1=0.7818

**Dev metrics @ 25fps:**
- Sign: F1=0.4963  IoU=0.2463  SegF1=0.4398
- Phrase: F1=0.5348  IoU=0.6928  SegF1=0.7683

**Notes:** Finetune E145 best.ckpt (HM=0.705, best overall) at LR=1e-4; E148 overfit at 6h

---

## E150-best-config-12h [SUCCESS] — 2026-03-21 11:32

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:12:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 100
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 170
- Best val loss: None
- Elapsed: 74.9 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E150-best-config-12h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5100  IoU=0.5971  SegF1=0.9708
- Phrase: F1=0.5516  IoU=0.6363  SegF1=0.7033

**Dev metrics @ 25fps:**
- Sign: F1=0.4871  IoU=0.4989  SegF1=0.8584
- Phrase: F1=0.5340  IoU=0.6912  SegF1=0.7827

**Notes:** E145 config (best HM=0.705): 1024fr+bs8+drop=0.1, max time 12h, patience=100

---

## E151-attnmask-2048-batch8-30m [SUCCESS] — 2026-03-21 18:08

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.0
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 34
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E151-attnmask-2048-batch8-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4880  IoU=0.5886  SegF1=0.9684
- Phrase: F1=0.5407  IoU=0.6803  SegF1=0.7401

**Dev metrics @ 25fps:**
- Sign: F1=0.4496  IoU=0.4295  SegF1=0.7458
- Phrase: F1=0.5096  IoU=0.7690  SegF1=0.8360

**Notes:** attn masking fix + 2048 frames batch=8 (E143 conditions)

---

## E152-attnmask-1024-dropout01-30m [SUCCESS] — 2026-03-21 18:38

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 57
- Best val loss: None
- Elapsed: 27.7 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E152-attnmask-1024-dropout01-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4990  IoU=0.5818  SegF1=0.9737
- Phrase: F1=0.5596  IoU=0.5930  SegF1=0.6467

**Dev metrics @ 25fps:**
- Sign: F1=0.4768  IoU=0.4945  SegF1=0.8652
- Phrase: F1=0.5405  IoU=0.6775  SegF1=0.7541

**Notes:** attn masking + 1024 frames + dropout=0.1 (E145 config) — phrase diagnostic

---

## E153-attnmask-2048-dropout01-30m [SUCCESS] — 2026-03-21 19:12

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 34
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E153-attnmask-2048-dropout01-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5015  IoU=0.5954  SegF1=0.9741
- Phrase: F1=0.5459  IoU=0.5564  SegF1=0.6490

**Dev metrics @ 25fps:**
- Sign: F1=0.4776  IoU=0.4417  SegF1=0.7618
- Phrase: F1=0.5293  IoU=0.6969  SegF1=0.7920

**Notes:** attn masking + 2048 frames + dropout=0.1 (E147 conditions + mask)

---

## E154-attnmask-1536-dropout01-30m [SUCCESS] — 2026-03-21 19:47

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1536
- num_frames_end: None
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 45
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E154-attnmask-1536-dropout01-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4950  IoU=0.5823  SegF1=0.9669
- Phrase: F1=0.5750  IoU=0.7644  SegF1=0.8381

**Dev metrics @ 25fps:**
- Sign: F1=0.4677  IoU=0.4929  SegF1=0.8615
- Phrase: F1=0.5326  IoU=0.7570  SegF1=0.8442

**Notes:** attn masking + 1536 frames + dropout=0.1 — sweet spot test

---

## E155-1536-nofpsaug-dropout01-30m [SUCCESS] — 2026-03-21 20:16

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: False
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1536
- num_frames_end: None
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 33
- Best val loss: None
- Elapsed: 23.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E155-1536-nofpsaug-dropout01-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.3660  IoU=0.4867  SegF1=0.8249
- Phrase: F1=0.5344  IoU=0.7402  SegF1=0.7677

**Dev metrics @ 25fps:**
- Sign: F1=0.3564  IoU=0.5204  SegF1=0.8206
- Phrase: F1=0.4620  IoU=0.7973  SegF1=0.8535

**Notes:** 1536 frames + NO fps_aug + dropout=0.1 + mask — padding vs fps_aug test

---

## E156-2048-nodropframe-drop01-30m [SUCCESS] — 2026-03-21 20:50

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 34
- Best val loss: None
- Elapsed: 31.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E156-2048-nodropframe-drop01-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5035  IoU=0.5874  SegF1=0.9714
- Phrase: F1=0.5452  IoU=0.8422  SegF1=0.8796

**Dev metrics @ 25fps:**
- Sign: F1=0.4791  IoU=0.4958  SegF1=0.8665
- Phrase: F1=0.5245  IoU=0.8812  SegF1=0.8421

**Notes:** 2048+drop=0.1+mask, NO frame_dropout (vs E153 frame_dropout=0.15)

---

## E157-1536-nodropframe-drop01-30m [SUCCESS] — 2026-03-21 21:24

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1536
- num_frames_end: None
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 45
- Best val loss: None
- Elapsed: 31.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E157-1536-nodropframe-drop01-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4892  IoU=0.5955  SegF1=0.9737
- Phrase: F1=0.5467  IoU=0.5511  SegF1=0.6335

**Dev metrics @ 25fps:**
- Sign: F1=0.4614  IoU=0.5034  SegF1=0.8633
- Phrase: F1=0.5269  IoU=0.6311  SegF1=0.7097

**Notes:** 1536+drop=0.1+mask, NO frame_dropout — vs E156 (2048) and E154 (1536+frame_drop=0.15)

---

## E158-1024-nodropframe-drop01-30m [SUCCESS] — 2026-03-21 21:55

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:00:30:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 20
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 57
- Best val loss: None
- Elapsed: 28.1 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E158-1024-nodropframe-drop01-30m_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5060  IoU=0.5819  SegF1=0.9698
- Phrase: F1=0.5469  IoU=0.4010  SegF1=0.4923

**Dev metrics @ 25fps:**
- Sign: F1=0.4859  IoU=0.4903  SegF1=0.8630
- Phrase: F1=0.5606  IoU=0.4614  SegF1=0.5467

**Notes:** 1024+drop=0.1+mask, NO frame_dropout — vs E152 (fd=0.15) and E157 (1536+fd=0)

---

## E159-1536-nodropframe-drop01-3h [SUCCESS] — 2026-03-21 23:10

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1536
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 101
- Best val loss: None
- Elapsed: 68.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E159-1536-nodropframe-drop01-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5075  IoU=0.6074  SegF1=0.9721
- Phrase: F1=0.5586  IoU=0.4071  SegF1=0.4722

**Dev metrics @ 25fps:**
- Sign: F1=0.4777  IoU=0.4455  SegF1=0.7569
- Phrase: F1=0.5427  IoU=0.6919  SegF1=0.7552

**Notes:** DEFINITIVE: 1536+drop=0.1+mask+fd=0 — best config from sweep5 diagnostics

---

## E160-2048-batch8-drop01-fixchunk-3h [SUCCESS] — 2026-03-22 01:03

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 121
- Best val loss: None
- Elapsed: 99.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E160-2048-batch8-drop01-fixchunk-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5179  IoU=0.6068  SegF1=0.9638
- Phrase: F1=0.5461  IoU=0.8669  SegF1=0.9039

**Dev metrics @ 25fps:**
- Sign: F1=0.4902  IoU=0.5629  SegF1=0.9760
- Phrase: F1=0.5255  IoU=0.7111  SegF1=0.7983

**Notes:** E147 repro with fixed 2048-frame inference chunks + no attn mask

---

## E161-2048-batch8-drop01-fd0-fixchunk-3h [SUCCESS] — 2026-03-22 02:52

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.0
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 2048
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 135
- Best val loss: None
- Elapsed: 106.6 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E161-2048-batch8-drop01-fd0-fixchunk-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5247  IoU=0.6030  SegF1=0.9622
- Phrase: F1=0.5571  IoU=0.5865  SegF1=0.6336

**Dev metrics @ 25fps:**
- Sign: F1=0.4988  IoU=0.4345  SegF1=0.7556
- Phrase: F1=0.5432  IoU=0.7614  SegF1=0.8471

**Notes:** E160+fd=0: no frame_dropout, 2048 chunks match training, no mask

---

## E162-1536-batch8-drop01-fixchunk-3h [SUCCESS] — 2026-03-22 04:27

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1536
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 139
- Best val loss: None
- Elapsed: 89.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E162-1536-batch8-drop01-fixchunk-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5249  IoU=0.6156  SegF1=0.9712
- Phrase: F1=0.5491  IoU=0.6661  SegF1=0.7567

**Dev metrics @ 25fps:**
- Sign: F1=0.5219  IoU=0.5381  SegF1=0.8682
- Phrase: F1=0.5457  IoU=0.6532  SegF1=0.7385

**Notes:** 1536 sweet spot: E145 phrase quality + E160 sign gain, fd=0.15 regularizes

---

## E163-1024-batch8-drop01-fixchunk-3h [SUCCESS] — 2026-03-22 05:21

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 7,801,597
- Best epoch: 84
- Best val loss: None
- Elapsed: 39.5 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E163-1024-batch8-drop01-fixchunk-3h_train.log

**Dev metrics @ 50fps:**
- N/A (training failed or no checkpoint)

**Dev metrics @ 25fps:**
- not run

**Notes:** E145 repro with new code (no mask, chunk_size=1024) — confirm or beat baseline

---

## E165-1536-batch8-drop01-fixchunk-hmval-3h [SUCCESS] — 2026-03-22 07:09

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 6
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1536
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 8,097,277
- Best epoch: 70
- Best val loss: None
- Elapsed: 44.2 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E165-1536-batch8-drop01-fixchunk-hmval-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.4909  IoU=0.5598  SegF1=0.9654
- Phrase: F1=0.5448  IoU=0.8892  SegF1=0.8540

**Dev metrics @ 25fps:**
- Sign: F1=0.4763  IoU=0.5364  SegF1=0.9669
- Phrase: F1=0.5400  IoU=0.8765  SegF1=0.9095

**Notes:** E162 clean re-run: 1536fr + chunk_size fix + HM validation metric

---

**Dev metrics @ 50fps (auto-eval 2026-03-22 08:37):**
- Sign IoU: 0.5598
- Phrase IoU: 0.8892
- HM: 0.6871

**Dev metrics @ 25fps:**
- Sign IoU: 0.5362
- Phrase IoU: 0.8765

**Dev metrics @ 50fps (auto-eval 2026-03-22 12:04):**
- Sign IoU: 0.5601
- Phrase IoU: 0.8892
- HM: 0.6873

**Dev metrics @ 25fps:**
- Sign IoU: 0.5372
- Phrase IoU: 0.8765

## E166-1024-depth4-drop01-fixchunk-hmval-3h [SUCCESS] — 2026-03-22 08:35

**Config:**
- acceleration: False
- arch: cnn-medium-attn
- attn_dropout: 0.1
- attn_ff_mult: 2
- attn_nhead: 8
- b_dice_loss_weight: 0.0
- batch_size: 8
- batch_size_schedule: False
- body_part_dropout: 0.1
- dice_loss_weight: 1.5
- encoder_depth: 4
- epochs: 200
- finetune_from: None
- focal_gamma: 0.0
- fps_aug: True
- frame_dropout: 0.15
- hidden_dim: 384
- label_smoothing: 0.0
- learning_rate: 0.0005
- loss_b_weight: 5.0
- loss_i_weight: 3.0
- max_time: 00:03:00:00
- no_face: True
- no_normalize: False
- num_frames: 1024
- num_frames_end: None
- optimizer: None
- patience: 50
- phrase_b_weight: 2.0
- phrase_i_weight: 1.5
- phrase_weighted_loss: False
- pos_encoding: rope
- pose_dims: 3
- sign_b_weight: 3.0
- sign_i_weight: 2.0
- sign_weighted_loss: False
- speed_aug: False
- steps_per_epoch: 100
- target_fps: None
- velocity: True
- weighted_loss: False

**Training:**
- Params: 5,734,141
- Best epoch: 199
- Best val loss: None
- Elapsed: 85.4 min
- Log: /mnt/rylo-tnas/users/amit/dev/sign-language-processing/segmentation/logs/E166-1024-depth4-drop01-fixchunk-hmval-3h_train.log

**Dev metrics @ 50fps:**
- Sign: F1=0.5445  IoU=0.6411  SegF1=0.9733
- Phrase: F1=0.5464  IoU=0.8996  SegF1=0.9006

**Dev metrics @ 25fps:**
- Sign: F1=0.5604  IoU=0.6343  SegF1=0.9757
- Phrase: F1=0.5480  IoU=0.9037  SegF1=0.9152

**Notes:** depth=4 vs E163 depth=6: test whether marginal sign gain justifies +2.4M params

---

**Dev metrics @ 50fps (auto-eval 2026-03-22 08:37):**
- Sign IoU: 0.6411
- Phrase IoU: 0.8996
- HM: 0.7487

**Dev metrics @ 25fps:**
- Sign IoU: 0.6343
- Phrase IoU: 0.9037

---
**Dev metrics @ 50fps (auto-eval 2026-03-22 12:04):**
- Sign IoU: 0.6407
- Phrase IoU: 0.8996
- HM: 0.7484

**Dev metrics @ 25fps:**
- Sign IoU: 0.6342
- Phrase IoU: 0.9037

## E167-1536-depth4-drop01-hmval-3h

**Dev metrics @ 50fps (auto-eval 2026-03-22 10:07):**
- Sign IoU: 0.6452
- Phrase IoU: 0.9081
- HM: 0.7544

**Dev metrics @ 25fps:**
- Sign IoU: 0.6352
- Phrase IoU: 0.9065

---
**Dev metrics @ 50fps (auto-eval 2026-03-22 12:04):**
- Sign IoU: 0.6452
- Phrase IoU: 0.9081
- HM: 0.7544

**Dev metrics @ 25fps:**
- Sign IoU: 0.6352
- Phrase IoU: 0.9065

## E168-1536-depth6-drop01-hmval-6h

**Dev metrics @ 50fps (auto-eval 2026-03-22 11:53):**
- Sign IoU: 0.6447
- Phrase IoU: 0.9042
- HM: 0.7527

**Dev metrics @ 25fps:**
- Sign IoU: 0.6320
- Phrase IoU: 0.9018

**Dev metrics @ 50fps (auto-eval 2026-03-22 12:05):**
- Sign IoU: 0.6447
- Phrase IoU: 0.9042
- HM: 0.7527

**Dev metrics @ 25fps:**
- Sign IoU: 0.6320
- Phrase IoU: 0.9018

---
## E169-1024-depth4-drop01-hmval-6h

**Dev metrics @ 50fps (auto-eval 2026-03-22 14:30):**
- Sign IoU: 0.6569
- Phrase IoU: 0.9104
- HM: 0.7631

**Dev metrics @ 25fps:**
- Sign IoU: 0.6526
- Phrase IoU: 0.9080

---
## E170-2048-depth4-drop01-hmval-3h

**Dev metrics @ 50fps (auto-eval 2026-03-22 16:20):**
- Sign IoU: 0.6401
- Phrase IoU: 0.9043
- HM: 0.7496

**Dev metrics @ 25fps:**
- Sign IoU: 0.6251
- Phrase IoU: 0.9030
