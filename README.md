# Diabetic Retinopathy Detection using Color Fundus Photos with Transfer Learning

End-to-end, reproducible training pipeline for detecting diabetic retinopathy (DR) from color fundus photographs in the Ocular Disease Intelligent Recognition (ODIR) dataset. The project operationalises the original `data_project.ipynb` exploration into a GitHub-friendly codebase that demonstrates how transfer learning across CNN/ViT backbones can accelerate DR screening research.

---

## Highlights

- Multiple torchvision/timm backbones (ResNet, DenseNet, EfficientNet, ViT) behind a single training interface
- Advanced augmentation, focal loss, class weighting, mixed precision, ReduceLROnPlateau, and early stopping baked in
- Cohort-aware metrics (overall / female / male) plus CSV logging and optional AUC comparison plots
- CLI scripts + YAML configs for hands-free experiments, while the original notebook remains available for reference

---

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

---

## Dataset Statistics

The numbers below mirror the dataset snapshot used in the original notebook (image resolution: 200×200, RGB). Adjust as needed if you regenerate new splits.

| Split | # Samples | Approx. DR+ (%) | Image Size |
| --- | ---: | ---: | --- |
| Train | 4,476 | ~30% | 200 × 200 × 3 |
| Val | 641 | ~30% | 200 × 200 × 3 |
| Test | 1,914 | ~30% | 200 × 200 × 3 |

> **Notes**
> - Counts reflect the snapshot used in `data_project.ipynb`. Recompute easily by running a simple stats script if your local copy differs.
> - Each `.npz` entry includes `slo_fundus`, `dr_class`, and `male`. Additional metadata (race, language, etc.) can be incorporated later for fairness analyses.
> - Sex distribution is roughly balanced, enabling subgroup metrics (overall/female/male).

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

![Sample confusion matrix](assets/confusion_matrix.png)
*Confusion matrix generated from the original notebook inference pass.*

![ROC curves across thresholds](assets/roc_curves.png)
*ROC curves for representative backbones (ResNet, DenseNet, EfficientNet) highlighting the effect of transfer learning.*

Add your own figures—dataset histograms, sample grids, attribution maps—by dropping PNGs into `assets/` and linking them here. This keeps the README visually aligned with the exploratory analysis in `data_project.ipynb`.

---

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

---

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
  --img_size 256 \
  --plot_results
```

---

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

---

## Notebook Reference

`notebooks/data_project.ipynb` mirrors the original exploratory workflow (data inspection, visualization, etc.). It shares the same dataset assumptions but is intentionally decoupled from production code; keep reusable logic inside `src/` and reserve the notebook for experimentation/demos.

---

## Extending the Benchmark

- **New backbone** – Implement the architecture in `models/factory.py` and add its keyword to `TrainingConfig`.
- **Additional metadata splits** – Modify `metrics.py` and `_group_metrics` in `evaluation.py` to include new demographic slices.
- **Experiment tracking** – Wrap training/evaluation calls with your tool of choice (Weights & Biases, MLflow, Aim).
- **Unit tests** – Add pytest cases under a `tests/` directory (dataset shape, metrics correctness, etc.).
- **Containers** – Build a Dockerfile that installs `requirements.txt`, copies this repo, and exposes the scripts for reproducible deployments.

---

## Troubleshooting

- **“No .npz files found”** – Verify the `--root_dir` path and folder naming (`train/`, `val/`, `test/`).
- **CUDA OOM** – Reduce `batch_size`, disable mixed precision (`--no-use_mixed_precision` via config), or downscale `img_size`.
- **Metrics stuck at zero** – Ensure `dr_class` contains both classes in each split; imbalance may require longer training or different thresholds.
- **Slow dataloading** – Increase `num_workers` or place the dataset on faster storage.

---

## Roadmap Ideas

- Add automated tests + CI
- Integrate experiment tracking out of the box
- Provide Docker/Apptainer recipes
- Release pretrained weights + sample predictions

PRs and issues are welcome!

