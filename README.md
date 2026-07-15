# KDTwin

## Task-Aware Knowledge Distillation for Lightweight Multi-Task Driving Scene Segmentation

## Setup

```bash
git clone https://github.com/chequanghuy/KDTwin.git
cd KDTwin
pip install -r requirements.txt
```

## Dataset and Pretrained Models

Please prepare the BDD100K dataset following the instructions from [TwinLiteNet+](https://github.com/chequanghuy/TwinLiteNetPlus).

Pretrained teacher and student models for TwinLiteNet+ and TwinMixing can be downloaded here:

- [Pretrained models](https://drive.google.com/file/d/1jQi8owm2aSz0Rr7sOXrTN96UbHtqyHmM/view?usp=sharing)

The expected directory structure is:

```text
KDTwin/
├── data/
│   └── bdd100k/
├── pretrained/
├── train_kd.py
├── val.py
├── ....
```

## Training

To distill TwinLiteNet+:

```bash
python train_kd.py \
  --ema \
  --data_dir data \
  --model twinplus \
  --verbose
```

To distill TwinMixing:

```bash
python train_kd.py \
  --ema \
  --data_dir data \
  --model twinmixing \
  --verbose
```

## Evaluation

Distilled checkpoints can be downloaded here:

- [KDTwin TwinLiteNet+ Nano](https://drive.google.com/file/d/1umIJZuXL1K2FvRWQC1a0Xr0myRtJRr7R/view?usp=sharing)
- [KDTwin TwinMixing Nano](https://drive.google.com/file/d/1eez5ipFQCA2UXPQragovZP9Eq8NNMpmO/view?usp=sharing)

Evaluate TwinMixing Nano:

```bash
python val.py \
  --weight kd_twinmixing_nano.pth \
  --model twinmixing \
  --data_dir data \
  --verbose
```

Evaluate TwinLiteNet+ Nano:

```bash
python val.py \
  --weight kd_twinplus_nano.pth \
  --model twinplus \
  --data_dir data \
  --verbose
```


## Citation

```
@INPROCEEDINGS{9924897,
  author={Huy Che and Minh-Khoi Do and Dinh-Duy Phan and Duc-Khai Lam},
  booktitle={2026 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)}, 
  title={KDTwin: Task-Aware Knowledge Distillation for Lightweight Multi-Task Driving Scene Segmentation}, 
  year={2026}
 }
```


## Acknowledgements



* [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet)
* [TwinLiteNet+](https://github.com/chequanghuy/TwinLiteNetPlus)
* [TwinMixing](https://github.com/Jun0se7en/TwinMixing)
