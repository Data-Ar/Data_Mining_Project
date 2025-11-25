# Diabetic Retinopathy Detection using Color Fundus Photos with Advanced Transfer Learning

This reposiory is for an end-end reproducible training pipeline for detecting diabetic retinopathy (DR) from color fundus photographs in the Ocular Disease Intelligent Recognition (ODIR) dataset. The project utilizes advanced data augmentation techniques and fine-tuning of several backboone for DR detection task. The ODIR-5K dtatset used for this project has been customized to allow for a binary classification task.

## Highlights

- Multiple torchvision/timm backbones (ResNet, DenseNet, EfficientNet, ViT) behind a single training interface
- Advanced augmentation, focal loss, class weighting, mixed precision, ReduceLROnPlateau, and early stopping baked in
- Cohort-aware metrics (overall / female / male) plus CSV logging and optional AUC comparison plots
- CLI scripts + YAML configs for hands-free experiments, while the original notebook remains available for reference

## Repository Layout

```
dr_odir_classifier/
├── configs/                # YAML configs for reproducibility
├── notebooks/              # Original exploratory notebook
├── scripts/                # CLI entry points (train/eval)
├── src/dr_odir_classifier/
│   ├── data/               # Dataset + transforms helpers
│   ├── models/             # Backbone factory
│   ├── training/           # Train loops + config dataclass
│   ├── evaluation.py       # Eval + TTA
│   ├── losses.py           # Focal loss impl
│   ├── metrics.py          # Metric helpers + pretty printer
│   └── plots.py            # AUC comparison visualiser
├── requirements.txt        # Minimal dependency pinning
├── pyproject.toml          # Packaging metadata
└── README.md               # You are here
```


## Dataset Layout

```
ODIR_Data/
  train/*.npz
  val/*.npz
  test/*.npz
```

Each `.npz` file must include:

- `slo_fundus` (H×W×C) RGB or single-channel fundus image
- `dr_class` (0/1) – binary DR label used for training
- `male` (0/1) – used for subgroup fairness metrics

No additional preprocessing is required; the loader normalises/augments on the fly. If your copy of ODIR uses different resolutions, update `img_size` in the config.

---

## Exploratory Figures



## Quickstart

1. **Environment**
   ```bash
   cd /medailab/medailab/sulaiman/dr_odir_classifier
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .      # or: pip install -r requirements.txt
   ```

2. **Sanity run**
   ```bash
   PYTHONPATH=src python scripts/train_backbones.py --epochs 1 --batch_size 8
   ```
   Confirms data loading, augmentation, and logging work before long experiments.

## Configuration

Default settings live in `configs/default.yaml`. Key fields:

| Field | Description |
| --- | --- |
| `arch` | Primary backbone to train (e.g., `resnet50`) |
| `extra_arches` | Optional list of additional backbones to train sequentially |
| `use_focal_loss` / `use_class_weights` | Toggle imbalance strategies |
| `img_size`, `batch_size`, `epochs`, `lr`, `weight_decay` | Core hyperparameters |
| `out_dir`, `csv_path` | Output folders for checkpoints / metric logs |
| `plot_results` | Whether to emit `backbone_auc_comparison.png` |

Override any field via CLI flags or by editing/duplicating the YAML.

---

## Training Workflows

### Config-driven run
```bash
PYTHONPATH=src python scripts/train_backbones.py \
  --config configs/default.yaml \
  --root_dir /path/to/ODIR_Data \
  --out_dir ./artifacts
```
- Produces best checkpoints in `out_dir`
- Appends per-group metrics to the configured CSV
- Uses any extra backbones specified in the YAML

### Ad-hoc run with CLI overrides
```bash
PYTHONPATH=src python scripts/train_backbones.py \
  --arch resnet50 \
  --extra_arches densenet121 vit_b_16 \
  --epochs 40 \
  --batch_size 16 \
  --img_size 224 \
  --plot_results
## Evaluation & Test-Time Augmentation

```bash
PYTHONPATH=src python scripts/evaluate_tta.py \
  --checkpoint artifacts/best_resnet50.pt \
  --root_dir /path/to/ODIR_Data \
  --tta --num_augments 5
```
- Without `--tta`, the script runs a single forward pass.
- With `--tta`, it averages over flips/rotations for improved stability.
- Outputs formatted metrics for overall/female/male cohorts.

---

## Artifacts & Logging

- **Checkpoints** – Saved as `best_<arch>.pt` with best validation AUC.
- **Metrics CSV** – Each training run appends per-group metrics (AUC, precision, recall, specificity, F1).
- **Plots** – When `--plot_results` is set, `backbone_auc_comparison.png` summarises AUCs across backbones.
- **Confusion Matrices** – Use `evaluation.plot_confusion_matrix` for custom visual diagnostics.
