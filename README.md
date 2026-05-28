# BIO-DRAM + Tiger Joint Training Project

This README explains the complete workflow of the BIO-DRAM + Tiger project, including data organization, Tiger generative model invocation, CNN classifier training and testing, Bayesian optimization, generator-classifier co-training logic, implemented modules, and modules that still need to be connected or updated.

---

## 1. Project Purpose

This project is designed for rare thyroid ultrasound subtype classification.

The framework contains two interacting systems:

1. **Tiger generative model**
   - Generates synthetic thyroid ultrasound images for rare classes.
   - Uses text prompts, initial images, foreground conditions, background conditions, masks, Stable Diffusion, ControlNet, and inpainting.
   - Can generate subtype-specific synthetic images such as FTC, MTC, and ATC.

2. **BIO-DRAM classification model**
   - Trains a CNN classifier using real and synthetic thyroid ultrasound images.
   - Uses class-imbalance-aware loss, clinical misdiagnosis cost loss, and recall-aware regularization.
   - Evaluates subtype-level diagnostic performance and high-risk misclassification reduction.

The most important design is that Tiger and the CNN classifier are jointly coordinated. Tiger generates synthetic images. The CNN classifier is trained on the mixture of real and generated images. Validation results are then used by Bayesian optimization to adjust generation parameters and classification loss weights.

---

## 2. Overall Workflow

```text
Real thyroid ultrasound images
        ↓
Tiger condition images and text prompts
        ↓
Tiger synthetic image generation
        ↓
Real + synthetic image mixing
        ↓
BIO-DRAM CNN classifier training
        ↓
Validation and safety-oriented evaluation
        ↓
Bayesian optimization
        ↓
Update Tiger generation parameters and classifier loss weights
        ↓
Repeat generation-training-evaluation loop
```

---

## 3. Recommended Project Structure

```text
BIO_DRAM_Tiger_project/
├── 01_prepare_csv.py
├── 02_tiger_generate.py
├── 03_train_classifier.py
├── 04_evaluate.py
├── 05_bayes_tiger_cnn.py
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   ├── real/
│   │   │   ├── Benign/
│   │   │   ├── FA/
│   │   │   ├── PTC/
│   │   │   ├── FTC/
│   │   │   ├── MTC/
│   │   │   └── ATC/
│   │   ├── synthetic/
│   │   │   ├── Benign/
│   │   │   ├── FA/
│   │   │   ├── PTC/
│   │   │   ├── FTC/
│   │   │   ├── MTC/
│   │   │   └── ATC/
│   │   └── tiger_conditions/
│   │       ├── init_image/
│   │       ├── mask_nd/
│   │       ├── mask_bg/
│   │       ├── condition_FG/
│   │       ├── condition_BG/
│   │       └── metadata.jsonl
│   ├── splits/
│   │   ├── all.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── metadata/
│       ├── class_names.txt
│       ├── cost_matrix.csv
│       └── bayes_trials.csv
├── outputs/
│   ├── checkpoints/
│   ├── predictions/
│   ├── synthetic_preview/
│   ├── logs/
│   └── bayes/
├── src/
│   └── biodram/
│       ├── datasets.py
│       ├── losses.py
│       ├── metrics.py
│       ├── models.py
│       ├── tiger_generator.py
│       ├── transforms.py
│       └── utils.py
├── requirements.txt
└── README.md
```

---

## 4. Environment Setup

### 4.1 Create Conda Environment

```bash
conda create -n biodram_tiger python=3.10 -y
conda activate biodram_tiger
```

### 4.2 Install Dependencies

```bash
pip install -r requirements.txt
```

Recommended packages:

```text
torch
torchvision
numpy
pandas
Pillow
opencv-python
scikit-learn
PyYAML
tqdm
diffusers
transformers
accelerate
safetensors
optuna
matplotlib
```

If CUDA is available, install a CUDA-compatible PyTorch version.

---

## 5. Data Organization

### 5.1 Real Images

Real thyroid ultrasound images should be placed in:

```text
data/raw/real/
├── Benign/
├── FA/
├── PTC/
├── FTC/
├── MTC/
└── ATC/
```

Example:

```text
data/raw/real/PTC/ptc_0001.png
data/raw/real/PTC/ptc_0002.png
data/raw/real/ATC/atc_0001.png
```

Class meaning:

| Folder | Meaning |
|---|---|
| `Benign` | Benign thyroid nodules |
| `FA` | Follicular adenoma |
| `PTC` | Papillary thyroid carcinoma |
| `FTC` | Follicular thyroid carcinoma |
| `MTC` | Medullary thyroid carcinoma |
| `ATC` | Anaplastic thyroid carcinoma |

---

### 5.2 Synthetic Images

Tiger-generated images should be saved to:

```text
data/raw/synthetic/
├── Benign/
├── FA/
├── PTC/
├── FTC/
├── MTC/
└── ATC/
```

Example:

```text
data/raw/synthetic/MTC/tiger_MTC_0001.png
data/raw/synthetic/ATC/tiger_ATC_0001.png
```

Usually, synthetic generation is mainly used for minority or rare subtypes:

```text
FTC
MTC
ATC
```

---

### 5.3 Tiger Condition Files

Tiger generation uses initial image, foreground mask or condition, background mask or condition, and text prompt.

Recommended folder structure:

```text
data/raw/tiger_conditions/
├── init_image/
├── mask_nd/
├── mask_bg/
├── condition_FG/
├── condition_BG/
└── metadata.jsonl
```

Folder meaning:

| Folder or file | Meaning |
|---|---|
| `init_image/` | Initial ultrasound images for inpainting or reference generation |
| `mask_nd/` | Nodule or foreground inpainting masks |
| `mask_bg/` | Background inpainting masks |
| `condition_FG/` | Foreground condition images |
| `condition_BG/` | Background condition images |
| `metadata.jsonl` | Generation metadata and prompts |

Example:

```text
data/raw/tiger_conditions/init_image/case001.png
data/raw/tiger_conditions/mask_nd/case001.png
data/raw/tiger_conditions/mask_bg/case001.png
data/raw/tiger_conditions/condition_FG/case001.png
data/raw/tiger_conditions/condition_BG/case001.png
```

Example `metadata.jsonl`:

```json
{"file_name":"case001.png","class_name":"MTC","init_image":"init_image/case001.png","mask_nd":"mask_nd/case001.png","mask_bg":"mask_bg/case001.png","condition_FG":"condition_FG/case001.png","condition_BG":"condition_BG/case001.png","text_nd":"medullary thyroid carcinoma, solid, hypoechoic, irregular margin, microcalcification","text_bg":"black and white ultrasound background, thyroid parenchyma, clear texture"}
{"file_name":"case002.png","class_name":"ATC","init_image":"init_image/case002.png","mask_nd":"mask_nd/case002.png","mask_bg":"mask_bg/case002.png","condition_FG":"condition_FG/case002.png","condition_BG":"condition_BG/case002.png","text_nd":"anaplastic thyroid carcinoma, large irregular nodule, heterogeneous echo, unclear margin","text_bg":"black and white ultrasound background, thyroid tissue"}
```

---

## 6. CSV Splits

The classifier reads image paths from CSV files.

```text
data/splits/
├── all.csv
├── train.csv
├── val.csv
└── test.csv
```

CSV format:

```csv
image_path,label,class_name,is_synthetic
data/raw/real/PTC/ptc_0001.png,2,PTC,0
data/raw/real/ATC/atc_0001.png,5,ATC,0
data/raw/synthetic/ATC/tiger_ATC_0001.png,5,ATC,1
```

Field meaning:

| Column | Meaning |
|---|---|
| `image_path` | Image file path |
| `label` | Integer class label |
| `class_name` | Class name |
| `is_synthetic` | 0 for real image, 1 for synthetic image |

---

## 7. Metadata Files

### 7.1 `class_names.txt`

The class order must be consistent across classifier output, cost matrix, and evaluation scripts.

Recommended order:

```text
Benign
FA
PTC
FTC
MTC
ATC
```

---

### 7.2 `cost_matrix.csv`

The cost matrix is used for clinical misdiagnosis cost-sensitive learning.

Rows are true classes. Columns are predicted classes.

Example:

```csv
true_pred,Benign,FA,PTC,FTC,MTC,ATC
Benign,0,1,2,3,3,4
FA,1,0,2,3,3,4
PTC,5,4,0,4,5,8
FTC,6,5,4,0,5,8
MTC,7,6,5,6,0,9
ATC,10,9,8,8,9,0
```

Higher values indicate more severe clinical consequences.

---

### 7.3 `bayes_trials.csv`

This file stores Bayesian optimization results.

Example:

```csv
trial,synthetic_ratio,guidance_scale,controlnet_scale,num_inference_steps,lambda_cost,lambda_recall,val_auc,val_macro_f1,val_recall_ATC,safety_score
1,0.5,7.5,0.8,30,1.0,1.0,0.873,0.801,0.820,0.891
2,1.0,8.0,0.6,40,1.5,1.2,0.881,0.815,0.860,0.905
```

---

## 8. Configuration File

Main configuration file:

```text
configs/config.yaml
```

Recommended content:

```yaml
project:
  seed: 42
  device: cuda
  output_dir: outputs

data:
  real_root: data/raw/real
  synthetic_root: data/raw/synthetic
  tiger_condition_root: data/raw/tiger_conditions
  split_dir: data/splits
  metadata_dir: data/metadata
  class_names:
    - Benign
    - FA
    - PTC
    - FTC
    - MTC
    - ATC
  image_size: 224
  val_ratio: 0.1
  test_ratio: 0.1
  include_synthetic_in_training: true

tiger:
  pretrained_model_path: ../model/pretrainmodel
  controlnet_fg_path: ../modelsaved/finetrainmodel/checkpoint-3000/controlnet
  controlnet_bg_path: ../modelsaved/finetrainmodel/checkpoint-5000/controlnet
  output_root: data/raw/synthetic
  num_inference_steps: 30
  guidance_scale: 7.5
  controlnet_conditioning_scale: 0.8
  eta: 1.0
  image_size: 512
  negative_prompt: ""
  scheduler: DDIM
  dtype: float16
  generation_mode: reference

classifier:
  model_name: resnet34
  pretrained: true
  num_classes: 6
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.0001
  num_workers: 4

loss:
  lambda_ce: 1.0
  lambda_margin: 0.5
  lambda_cost: 1.0
  lambda_recall: 1.0
  margin_alpha: 0.35

bayes:
  n_trials: 20
  target_classes:
    - FTC
    - MTC
    - ATC
  optimize_generation: true
  optimize_loss_weights: true
  objective_metric: safety_score
```

---

## 9. Step 1: Prepare Training CSV Files

### 9.1 Command

```bash
python 01_prepare_csv.py --config configs/config.yaml
```

### 9.2 What This Script Does

This script:

1. Scans `data/raw/real/`.
2. Scans `data/raw/synthetic/`.
3. Assigns labels according to `class_names`.
4. Creates `all.csv`.
5. Splits real data into training, validation, and test subsets.
6. Adds synthetic images to the training subset only if enabled.

### 9.3 Output

```text
data/splits/all.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

---

## 10. Step 2: Call Tiger Generative Model

### 10.1 Command

Generate 50 ATC images:

```bash
python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name ATC \
  --num_images 50
```

Generate 80 MTC images:

```bash
python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name MTC \
  --num_images 80
```

Generate 80 FTC images:

```bash
python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name FTC \
  --num_images 80
```

### 10.2 Tiger Generation Logic

The Tiger generation script performs:

1. Load metadata from `metadata.jsonl`.
2. Select cases matching the target class.
3. Load initial image.
4. Load foreground mask.
5. Load background mask.
6. Load foreground condition image.
7. Load background condition image.
8. Load text prompt.
9. Build inpainting condition.
10. Load Stable Diffusion + ControlNet pipeline.
11. Generate foreground or background image.
12. Save synthetic image into `data/raw/synthetic/{class_name}/`.

### 10.3 Main Tiger Function

The generation script should call `TigerGenerator` in:

```text
src/biodram/tiger_generator.py
```

A simplified usage pattern:

```python
from src.biodram.tiger_generator import TigerGenerator

generator = TigerGenerator(config)
generator.generate_class(
    class_name="ATC",
    num_images=50,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.8,
    num_inference_steps=30
)
```

### 10.4 Tiger Parameters

| Parameter | Meaning |
|---|---|
| `class_name` | Target class to generate |
| `num_images` | Number of synthetic images |
| `guidance_scale` | Text prompt guidance strength |
| `controlnet_conditioning_scale` | ControlNet condition strength |
| `num_inference_steps` | Number of denoising steps |
| `eta` | Sampling randomness |
| `seed` | Reproducibility control |
| `negative_prompt` | Features to avoid |
| `image_size` | Generation image size |

### 10.5 Foreground and Background Generation

Tiger may generate lesion foreground and background separately.

Foreground generation focuses on subtype-specific nodule features.

Example foreground prompt:

```text
medullary thyroid carcinoma, solid, hypoechoic, irregular margin, microcalcification
```

Background generation focuses on thyroid parenchyma or general ultrasound texture.

Example background prompt:

```text
black and white ultrasound background, thyroid parenchyma, clear texture
```

### 10.6 Inpainting Condition Function

The inpainting condition replaces masked pixels with `-1.0`.

```python
def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    image[image_mask > 0.5] = -1.0
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(image)
```

### 10.7 Output

Generated images are saved into:

```text
data/raw/synthetic/{class_name}/
```

Example:

```text
data/raw/synthetic/ATC/tiger_ATC_000001.png
data/raw/synthetic/ATC/tiger_ATC_000002.png
```

---

## 11. Step 3: Train the BIO-DRAM Classifier

### 11.1 Command

```bash
python 03_train_classifier.py --config configs/config.yaml
```

### 11.2 Classifier Inputs

The classifier uses:

```text
data/splits/train.csv
data/splits/val.csv
```

The training set may include both real and synthetic images.

Validation and test sets should contain real images only unless a separate synthetic evaluation is desired.

### 11.3 Model Backbone

Default backbone:

```text
ResNet34
```

Optional backbones:

```text
ResNet50
EfficientNet
ConvNeXt
Swin Transformer
```

### 11.4 BIO-DRAM Loss Components

The total loss can include:

```text
Cross-entropy loss
Dynamic margin loss
Clinical cost-sensitive loss
Recall-aware regularization
```

The total objective:

```text
L_total = lambda_ce * L_ce
        + lambda_margin * L_margin
        + lambda_cost * L_cost
        + lambda_recall * L_recall
```

### 11.5 Clinical Cost-Sensitive Loss

The cost matrix penalizes severe clinical mistakes more strongly.

Example high-risk mistakes:

```text
ATC predicted as Benign
MTC predicted as Benign
FTC predicted as FA
```

### 11.6 Recall-Aware Regularization

Recall-aware regularization increases penalty for false negatives in clinically important classes.

Target classes can be:

```text
FTC
MTC
ATC
```

### 11.7 Output

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
outputs/logs/train_log.csv
```

---

## 12. Step 4: Evaluate the Classifier

### 12.1 Command

```bash
python 04_evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

### 12.2 Metrics

Evaluation includes:

| Metric | Purpose |
|---|---|
| Accuracy | Overall correctness |
| Macro F1 | Balanced multi-class performance |
| Weighted F1 | Distribution-aware performance |
| Per-class recall | Rare subtype sensitivity |
| Per-class precision | False-positive control |
| Confusion matrix | Error pattern analysis |
| Cost-weighted error | Clinical safety evaluation |
| High-risk false negative count | Safety-oriented evaluation |

### 12.3 Output

```text
outputs/predictions/test_predictions.csv
outputs/predictions/test_metrics.json
outputs/predictions/confusion_matrix.png
```

---

## 13. Step 5: Bayesian Optimization for Tiger-CNN Cooperation

### 13.1 Purpose

`05_bayes_tiger_cnn.py` coordinates Tiger and the CNN classifier.

It searches for generation parameters and classifier loss weights that maximize validation safety and classification performance.

### 13.2 Command

```bash
python 05_bayes_tiger_cnn.py \
  --config configs/config.yaml \
  --n_trials 20
```

### 13.3 Search Space

Bayesian optimization can search:

| Parameter | Meaning |
|---|---|
| `synthetic_ratio` | Number of generated images used for each rare class |
| `guidance_scale` | Text prompt guidance strength in Tiger |
| `controlnet_conditioning_scale` | ControlNet condition strength |
| `num_inference_steps` | Tiger denoising steps |
| `lambda_cost` | Clinical cost loss weight |
| `lambda_recall` | Recall-aware loss weight |
| `lambda_margin` | Dynamic margin loss weight |
| `margin_alpha` | Minority-class margin strength |

Example search space:

```yaml
bayes_search_space:
  synthetic_ratio:
    type: float
    low: 0.25
    high: 2.0
  guidance_scale:
    type: float
    low: 4.0
    high: 10.0
  controlnet_conditioning_scale:
    type: float
    low: 0.2
    high: 1.2
  num_inference_steps:
    type: int
    low: 20
    high: 50
  lambda_cost:
    type: float
    low: 0.1
    high: 3.0
  lambda_recall:
    type: float
    low: 0.1
    high: 3.0
  lambda_margin:
    type: float
    low: 0.0
    high: 2.0
```

### 13.4 Objective Function

The Bayesian objective should not depend only on accuracy.

A recommended objective:

```text
safety_score = 0.40 * macro_auc
             + 0.25 * macro_f1
             + 0.25 * mean_recall_rare
             - 0.10 * normalized_cost_error
```

Where:

```text
mean_recall_rare = mean(recall_FTC, recall_MTC, recall_ATC)
```

This design favors models that maintain strong overall performance while reducing high-risk false negatives.

### 13.5 One Bayesian Trial

Each trial performs:

```text
Sample generation parameters and loss weights
        ↓
Call Tiger to generate synthetic images for rare classes
        ↓
Update train.csv with real + selected synthetic images
        ↓
Train CNN classifier
        ↓
Evaluate validation set
        ↓
Compute safety_score
        ↓
Return safety_score to Bayesian optimizer
```

### 13.6 Output

```text
outputs/bayes/trial_001/
outputs/bayes/trial_002/
outputs/bayes/best_params.json
data/metadata/bayes_trials.csv
```

---

## 14. Joint Training and Cooperation Between Tiger and CNN

### 14.1 Why Joint Coordination Is Needed

Tiger generation quality should be judged by whether generated images improve downstream classification.

A synthetic image is useful only if it helps the classifier:

1. Improve rare subtype recall.
2. Reduce high-risk false negatives.
3. Preserve overall AUC and F1.
4. Avoid generating unrealistic or misleading samples.

### 14.2 Cooperation Mechanism

The project uses a generator-classifier feedback loop:

```text
Tiger generator produces synthetic samples
        ↓
CNN classifier learns from real + synthetic images
        ↓
Validation metrics reveal whether synthetic samples are useful
        ↓
Bayesian optimizer updates Tiger generation parameters
        ↓
Tiger generates improved synthetic samples
```

### 14.3 Dynamic Association

The dynamic association between Tiger and the CNN classifier is implemented through Bayesian optimization.

Tiger does not update its neural weights during each trial. Instead, its generation behavior is adjusted by tunable inference and sampling parameters.

Parameters include:

```text
guidance_scale
controlnet_conditioning_scale
num_inference_steps
synthetic_ratio
prompt template
foreground-background generation mode
```

The CNN model is retrained or fine-tuned after each synthetic dataset update.

---

## 15. Recommended Running Order

### 15.1 Single Training Run Without Bayesian Optimization

```bash
python 01_prepare_csv.py --config configs/config.yaml

python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name FTC \
  --num_images 50

python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name MTC \
  --num_images 50

python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name ATC \
  --num_images 50

python 01_prepare_csv.py --config configs/config.yaml

python 03_train_classifier.py --config configs/config.yaml

python 04_evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

### 15.2 Full Generator-Classifier Bayesian Optimization Run

```bash
python 01_prepare_csv.py --config configs/config.yaml

python 05_bayes_tiger_cnn.py \
  --config configs/config.yaml \
  --n_trials 20

python 04_evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

---

## 16. Implemented Modules

The current project includes the following implemented or partially implemented components.

### 16.1 Implemented

| File | Status | Function |
|---|---|---|
| `01_prepare_csv.py` | Implemented | Generate CSV splits from real and synthetic folders |
| `02_tiger_generate.py` | Implemented interface | Call Tiger generation module |
| `03_train_classifier.py` | Implemented | Train CNN classifier with BIO-DRAM losses |
| `04_evaluate.py` | Implemented | Evaluate checkpoint and save metrics |
| `05_bayes_tiger_cnn.py` | Implemented framework | Run Bayesian optimization over generation and classifier parameters |
| `src/biodram/datasets.py` | Implemented | Dataset reader |
| `src/biodram/losses.py` | Implemented | CE, margin, cost-sensitive, recall-aware losses |
| `src/biodram/metrics.py` | Implemented | Classification metrics and safety metrics |
| `src/biodram/models.py` | Implemented | CNN classifier backbone |
| `src/biodram/tiger_generator.py` | Implemented interface | Tiger wrapper for Stable Diffusion + ControlNet generation |
| `src/biodram/transforms.py` | Implemented | Image preprocessing and augmentation |
| `src/biodram/utils.py` | Implemented | Config, seed, IO, logging utilities |

---

## 17. Not Fully Updated Yet

The following parts are included as framework or placeholder logic and may need to be updated according to the actual local environment, Tiger checkpoint location, and data format.

### 17.1 Tiger Checkpoint Paths

Need to update in `configs/config.yaml`:

```yaml
tiger:
  pretrained_model_path: ../model/pretrainmodel
  controlnet_fg_path: ../modelsaved/finetrainmodel/checkpoint-3000/controlnet
  controlnet_bg_path: ../modelsaved/finetrainmodel/checkpoint-5000/controlnet
```

Update these paths to match the actual local Tiger model folders.

---

### 17.2 Tiger Foreground and Background Generation

Current framework supports Tiger generation, but the following may need local adjustment:

| Component | Need to update |
|---|---|
| Foreground ControlNet path | Match actual checkpoint |
| Background ControlNet path | Match actual checkpoint |
| `mask_nd` folder name | Match actual foreground mask folder |
| `mask_bg` folder name | Match actual background mask folder |
| `condition_FG` folder name | Match actual foreground condition folder |
| `condition_BG` folder name | Match actual background condition folder |
| Prompt templates | Match subtype descriptions |
| Image output size | Match Tiger checkpoint training size |

---

### 17.3 Prompt Engineering

Default prompts are examples only.

Need to update subtype prompts according to clinical knowledge.

Example prompts:

```text
papillary thyroid carcinoma, solid, hypoechoic, microcalcification, irregular margin
follicular thyroid carcinoma, solid nodule, heterogeneous echo, peripheral halo
medullary thyroid carcinoma, solid hypoechoic nodule, calcification, irregular margin
anaplastic thyroid carcinoma, large irregular nodule, heterogeneous echo, unclear margin
```

---

### 17.4 Synthetic Image Quality Control

The current framework generates images and uses validation metrics for indirect filtering.

Recommended updates:

1. Add visual quality filtering.
2. Add duplicate filtering.
3. Add CLIP-like image-text consistency score if available.
4. Add MoSo score if implemented.
5. Add radiologist review flag if manual quality control is used.

Suggested CSV fields:

```csv
image_path,class_name,prompt,seed,guidance_scale,controlnet_scale,quality_score,keep
```

---

### 17.5 Bayesian Optimization Runtime

The full Bayesian loop is computationally expensive because each trial may include:

1. Image generation.
2. CSV update.
3. CNN training.
4. Validation.
5. Metric computation.

For quick debugging, reduce:

```yaml
classifier:
  num_epochs: 5

bayes:
  n_trials: 3
```

For final experiments, increase:

```yaml
classifier:
  num_epochs: 100

bayes:
  n_trials: 20
```

---

### 17.6 Classifier Backbone

Current default is ResNet34.

Can be updated to:

```text
ResNet50
EfficientNet-B0
ConvNeXt-Tiny
Swin-Tiny
DenseNet121
```

Need to ensure output dimension equals number of classes.

---

### 17.7 Clinical Cost Matrix

The example cost matrix must be replaced by expert-defined clinical misdiagnosis scores.

Need to confirm:

1. Class order.
2. True label rows.
3. Predicted label columns.
4. Higher values for severe false negatives.
5. Zero diagonal.

---

### 17.8 External Validation

Current project structure supports train, validation, and test.

If external validation is available, add:

```text
data/splits/external.csv
```

Then run:

```bash
python 04_evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split external
```

---

### 17.9 Rare-Class Safety Metrics

Current framework can compute per-class metrics.

Need to ensure rare classes are explicitly defined in config:

```yaml
bayes:
  target_classes:
    - FTC
    - MTC
    - ATC
```

Recommended safety metrics:

```text
ATC recall
MTC recall
FTC recall
High-risk false negative count
Cost-weighted error burden
Macro F1
Macro AUC
```

---

## 18. Recommended Debugging Workflow

### 18.1 Check Real Data

```bash
python 01_prepare_csv.py --config configs/config.yaml
```

Confirm:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

### 18.2 Run a Tiny Classifier Training

Set:

```yaml
classifier:
  num_epochs: 2
```

Run:

```bash
python 03_train_classifier.py --config configs/config.yaml
```

### 18.3 Test Tiger on One Image

```bash
python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name ATC \
  --num_images 1
```

Confirm output:

```text
data/raw/synthetic/ATC/
```

### 18.4 Run Short Bayesian Optimization

Set:

```yaml
bayes:
  n_trials: 2

classifier:
  num_epochs: 2
```

Run:

```bash
python 05_bayes_tiger_cnn.py \
  --config configs/config.yaml \
  --n_trials 2
```

---

## 19. Expected Outputs

After the full pipeline, expected outputs include:

```text
outputs/checkpoints/best.pt
outputs/checkpoints/last.pt
outputs/predictions/test_predictions.csv
outputs/predictions/test_metrics.json
outputs/predictions/confusion_matrix.png
outputs/bayes/best_params.json
outputs/bayes/trial_*/trial_metrics.json
data/metadata/bayes_trials.csv
data/raw/synthetic/*/*.png
```

---

## 20. Common Problems and Fixes

### Problem 1: Tiger model path not found

Check:

```yaml
tiger:
  pretrained_model_path:
  controlnet_fg_path:
  controlnet_bg_path:
```

Make sure these folders exist.

---

### Problem 2: CUDA out of memory

Try:

```yaml
classifier:
  batch_size: 8
```

For Tiger:

```yaml
tiger:
  dtype: float16
```

Also enable CPU offload in the Tiger generator if needed.

---

### Problem 3: No synthetic images included in training

Check:

```yaml
data:
  include_synthetic_in_training: true
```

Then rerun:

```bash
python 01_prepare_csv.py --config configs/config.yaml
```

---

### Problem 4: Class labels do not match cost matrix

Check:

```text
data/metadata/class_names.txt
data/metadata/cost_matrix.csv
configs/config.yaml
```

All class order definitions must match.

---

### Problem 5: Bayesian optimization is too slow

Use short debug mode:

```yaml
classifier:
  num_epochs: 2

bayes:
  n_trials: 2
```

After debugging, restore full settings.

---

## 21. Suggested Final Experimental Design

Recommended comparisons:

| Experiment | Training data | Loss | Purpose |
|---|---|---|---|
| Baseline | Real only | CE | Standard classifier |
| Class-balanced | Real only | CE + class weights | Distribution imbalance |
| Tiger augmentation | Real + synthetic | CE | Effect of generation |
| BIO-DRAM | Real + synthetic | CE + margin + cost + recall | Clinical safety |
| BIO-DRAM + Bayes | Real + optimized synthetic | Optimized loss weights | Joint generator-classifier calibration |

Recommended final reported metrics:

```text
Accuracy
Macro AUC
Macro F1
Per-class recall
ATC recall
MTC recall
FTC recall
High-risk false negatives
Cost-weighted error burden
Confusion matrix
```

---

## 22. Citation and Reproducibility Notes

For reproducibility, always record:

```text
Random seed
Tiger checkpoint paths
Generated image prompts
Generated image seeds
Generation parameters
Train/val/test split files
Cost matrix
Model checkpoint
Config file
Bayesian trial log
```

Recommended save files:

```text
outputs/logs/run_config.yaml
outputs/logs/train_log.csv
outputs/predictions/test_predictions.csv
data/metadata/bayes_trials.csv
```

---

## 23. Minimal Full Run

```bash
conda activate biodram_tiger

python 01_prepare_csv.py --config configs/config.yaml

python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name ATC \
  --num_images 50

python 01_prepare_csv.py --config configs/config.yaml

python 03_train_classifier.py --config configs/config.yaml

python 04_evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

---

## 24. Full Joint Optimization Run

```bash
conda activate biodram_tiger

python 01_prepare_csv.py --config configs/config.yaml

python 05_bayes_tiger_cnn.py \
  --config configs/config.yaml \
  --n_trials 20

python 04_evaluate.py \
  --config configs/config.yaml \
  --checkpoint outputs/checkpoints/best.pt \
  --split test
```

---

## 25. Summary

This project implements a rare thyroid subtype classification framework where Tiger generation and BIO-DRAM classification are connected through Bayesian optimization.

The generator creates synthetic rare-subtype thyroid ultrasound images. The classifier learns from both real and generated images. Bayesian optimization evaluates whether generated images improve validation performance and safety-oriented rare-subtype recall, then updates generation parameters and classifier loss weights.

The final goal is not only to improve overall classification accuracy, but also to reduce clinically dangerous high-risk misclassifications.

---

## 26. Relation to the Previous Tiger Codebase

This project does not rewrite the Tiger generation model from scratch. The Tiger-related code in this repository is designed as a wrapper and workflow controller around the previous Tiger implementation.

The Tiger generation module is based on the earlier Tiger code structure, including:

```text
StableDiffusionControlNetInpaintPipeline
ControlNetModel
DDIMScheduler or DPMSolverMultistepScheduler
foreground inpainting condition
background inpainting condition
init image
nodule mask
background mask
foreground condition image
background condition image
text prompt
```

The previous Tiger project uses the following generation logic:

```text
init_image
        ↓
mask_nd and condition_FG
        ↓
foreground or nodule-region generation
        ↓
mask_bg and condition_BG
        ↓
background-region generation
        ↓
synthetic thyroid ultrasound image
```

Therefore, in the present BIO-DRAM + Tiger project, `src/biodram/tiger_generator.py` should be regarded as an interface layer that calls the earlier Tiger generation pipeline.

---

## 27. How the Previous Tiger Code Is Mapped into This Project

The earlier Tiger code used paths similar to:

```text
../Figure/paper/image/
../Figure/paper/mask_nd/
../Figure/paper/mask_bg/
../dataset/Allclass/condition_bg/
../modelsaved/finetrainmodel/checkpoint-3000/controlnet
../modelsaved/finetrainmodel/checkpoint-5000/controlnet
../model/pretrainmodel
```

In this project, these paths are reorganized as:

```text
data/raw/tiger_conditions/init_image/
data/raw/tiger_conditions/mask_nd/
data/raw/tiger_conditions/mask_bg/
data/raw/tiger_conditions/condition_FG/
data/raw/tiger_conditions/condition_BG/
```

The model checkpoint paths are controlled by `configs/config.yaml`:

```yaml
tiger:
  pretrained_model_path: ../model/pretrainmodel
  controlnet_fg_path: ../modelsaved/finetrainmodel/checkpoint-3000/controlnet
  controlnet_bg_path: ../modelsaved/finetrainmodel/checkpoint-5000/controlnet
```

If your local Tiger folders are different, update only the paths in `config.yaml`. The Python scripts should not require path edits.

---

## 28. Previous Tiger Function Preserved in the Wrapper

The earlier Tiger inpainting condition function should be preserved in `src/biodram/tiger_generator.py`.

```python
def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    image[image_mask > 0.5] = -1.0
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    return torch.from_numpy(image)
```

This function converts an image and its mask into a ControlNet-compatible inpainting condition. Masked pixels are set to `-1.0`, following the previous Tiger implementation.

---

## 29. Tiger Foreground and Background Generation Modes

The previous Tiger code contained two major generation branches:

### 29.1 Foreground or Nodule Generation

This branch uses the nodule mask and foreground condition.

Typical inputs:

```text
init_image
mask_nd
condition_FG
text_nd
foreground ControlNet checkpoint
```

Example prompt:

```text
malignant medullary, solid, cystic, uneven echo, white points, enormous nodes
```

This branch is used to generate subtype-specific thyroid nodule features.

---

### 29.2 Background Generation

This branch uses the background mask and background condition.

Typical inputs:

```text
init_image
mask_bg
condition_BG
text_bg
background ControlNet checkpoint
```

Example prompt:

```text
black and white definition detail
```

This branch is used to generate or refine ultrasound background texture.

---

## 30. How to Call the Previous Tiger Logic in the Current Project

The command remains:

```bash
python 02_tiger_generate.py \
  --config configs/config.yaml \
  --class_name ATC \
  --num_images 50
```

Internally, this command should call:

```python
from src.biodram.tiger_generator import TigerGenerator

generator = TigerGenerator(config)
generator.generate_class(
    class_name="ATC",
    num_images=50,
    guidance_scale=7.5,
    controlnet_conditioning_scale=0.8,
    num_inference_steps=30
)
```

The function `generate_class()` should:

1. Read `metadata.jsonl`.
2. Select metadata records matching `class_name`.
3. Load `init_image`.
4. Load `mask_nd` or `mask_bg`.
5. Load `condition_FG` or `condition_BG`.
6. Load prompt text.
7. Build inpainting condition using `make_inpaint_condition`.
8. Load Stable Diffusion + ControlNet pipeline.
9. Generate the synthetic image.
10. Save the output to `data/raw/synthetic/{class_name}/`.

---

## 31. How Bayesian Optimization Controls the Previous Tiger Generator

The previous Tiger code used fixed generation parameters, such as:

```text
num_inference_steps
guidance_scale
eta
controlnet_conditioning_scale
seed
prompt
```

In this project, these parameters become tunable variables in Bayesian optimization.

The Bayesian optimization script `05_bayes_tiger_cnn.py` adjusts:

```text
guidance_scale
controlnet_conditioning_scale
num_inference_steps
synthetic_ratio
lambda_cost
lambda_recall
lambda_margin
```

This means the generation model and the classifier are not independent. Tiger generates candidate synthetic images under a sampled parameter setting. The CNN classifier is then trained with these images. The validation performance is returned to the optimizer, which decides whether the current generation setting improves rare-class classification and clinical safety.

---

## 32. Important Difference from the Previous Tiger Code

The earlier Tiger script was mainly an image generation script.

The current BIO-DRAM + Tiger project adds:

```text
automatic dataset preparation
synthetic image mixing
CNN classifier training
clinical cost-sensitive loss
recall-aware regularization
evaluation metrics
Bayesian optimization feedback loop
```

Therefore, the previous Tiger code is used as the generative engine, while this project adds the classifier, safety-aware loss functions, and generator-classifier coordination.

---

## 33. Parts Directly Referencing the Previous Tiger Code

The following files are expected to reference or wrap the earlier Tiger implementation:

| Current file | Relation to previous Tiger code |
|---|---|
| `src/biodram/tiger_generator.py` | Main wrapper for Stable Diffusion + ControlNet generation |
| `02_tiger_generate.py` | Command-line entry for image generation |
| `05_bayes_tiger_cnn.py` | Calls Tiger generation with Bayesian-updated parameters |
| `configs/config.yaml` | Stores Tiger model paths and generation parameters |
| `data/raw/tiger_conditions/metadata.jsonl` | Replaces earlier scattered image, mask, condition, and prompt paths |

---

## 34. Parts That Still Need Local Confirmation

Because the previous Tiger code depends on local checkpoint and dataset paths, the following parts must be checked before running:

```text
pretrained Stable Diffusion model path
foreground ControlNet checkpoint path
background ControlNet checkpoint path
init image folder name
nodule mask folder name
background mask folder name
foreground condition folder name
background condition folder name
metadata.jsonl field names
output image size
scheduler type
GPU memory configuration
```

If the previous Tiger code used different folder names, update `metadata.jsonl` and `configs/config.yaml`.

---

## 35. Suggested Local Migration from the Old Tiger Folder

If you already have the old Tiger folder, migrate files as follows:

```text
Old Tiger path                                   New project path
--------------------------------------------------------------------------------
../Figure/paper/image/                           data/raw/tiger_conditions/init_image/
../Figure/paper/mask_nd/                         data/raw/tiger_conditions/mask_nd/
../Figure/paper/mask_bg/                         data/raw/tiger_conditions/mask_bg/
../dataset/Allclass/condition_bg/                data/raw/tiger_conditions/condition_BG/
foreground condition folder                      data/raw/tiger_conditions/condition_FG/
generated rare subtype images                    data/raw/synthetic/{class_name}/
```

Then create or update:

```text
data/raw/tiger_conditions/metadata.jsonl
```

Each line should map one image to its masks, conditions, class name, and prompts.

---

## 36. Recommended Note for the Manuscript or Internal Documentation

The code implementation of the generation component follows the previous Tiger generation pipeline based on Stable Diffusion, ControlNet, and inpainting. In the present BIO-DRAM framework, Tiger is wrapped as a controllable synthetic image generator. Its inference parameters are optimized through Bayesian optimization according to downstream CNN validation performance, allowing the generation process to be dynamically associated with classifier training and rare-subtype recall improvement.

