# BIO-DRAM Tiger Project

This project implements a BIO-DRAM workflow for rare thyroid ultrasound classification with Tiger-based synthetic image generation, Bayesian optimization, and CNN classifier training.

## Main workflow

1. Prepare real image CSV files.
2. Generate synthetic images using the Tiger diffusion interface.
3. Train a CNN classifier using real and generated images.
4. Evaluate validation and test performance.
5. Use Bayesian optimization to tune generation parameters and classifier loss weights jointly.

## Recommended data layout

```text
data/
  raw/
    real/
      Benign/
      FA/
      PTC/
      FTC/
      MTC/
      ATC/
    tiger_conditions/
      init_image/
      mask_nd/
      mask_bg/
      condition_bg/
    synthetic/
      Benign/
      FA/
      PTC/
      FTC/
      MTC/
      ATC/
  splits/
    train.csv
    val.csv
    test.csv
  metadata/
    class_names.txt
    cost_matrix.csv
```

CSV format:

```text
path,label,source
/path/to/image.png,0,real
/path/to/synthetic.png,5,synthetic
```

## Quick start

```bash
pip install -r requirements.txt
python 01_prepare_csv.py --config configs/config.yaml
python 02_tiger_generate.py --config configs/config.yaml --class_name ATC --num_images 50
python 03_train_classifier.py --config configs/config.yaml
python 04_evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt --split test
python 05_bayes_tiger_cnn.py --config configs/config.yaml --n_trials 20
```
