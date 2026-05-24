# Acoustic Keystroke Inference
### A Comparative Study of Hybrid Neural Network Architectures for Side-Channel Attacks

**Author:** Vinit Rane — MS Cybersecurity, California State University, Dominguez Hills  
**Paper:** [IEEE Access submission — link to be added]  
**Dataset:** [Zenodo DOI — to be assigned]

---

## Overview

This repository contains all code, scripts, and results for the acoustic keystroke inference study comparing CoAtNet-0, MaxViT-Tiny, and Swin-Tiny for keyboard side-channel attack classification.

**Key results (5-seed statistical validation):**
| Model | Custom E1 Mean±Std | Custom E2 | E1+E2 Combined |
|-------|-------------------|-----------|----------------|
| CoAtNet-0 | 86.67% ± 2.29% | 89.63% | 90.93% |
| MaxViT-Tiny | 85.48% ± 0.98% | 93.33% | **93.70%** |
| Swin-Tiny | 78.15% | 90.00% | 85.37% |
| MFCC+SVM | 80.74% | 75.19% | — |

---

## Repository Structure

```
├── mka_dataloader.py       # MKA dataset loader (6 keyboard platforms)
├── custom_dataloader.py    # Custom E1/E2 dataset loader
├── train.py                # Train on MKA dataset
├── train_custom.py         # Train on custom dataset
├── train_multiseed.py      # Multi-seed statistical validation
├── segment.py              # Onset-based keystroke segmentation
├── baseline_mfcc_svm.py    # MFCC + SVM/RF classical baseline
├── cross_dataset.py        # Cross-dataset generalization experiments
├── plot_results.py         # Generate all publication figures
├── results/                # Training history JSON files
│   ├── clean_run/          # MKA results
│   ├── custom/             # Custom dataset results
│   ├── multiseed/          # Multi-seed validation results
│   ├── cross_dataset/      # Cross-dataset experiment results
│   └── baseline/           # Classical baseline results
└── figures/                # Publication figures (PNG)
```

---

## Setup

```bash
# Create conda environment
conda create -n acoustic python=3.10
conda activate acoustic

# Install dependencies
pip install torch torchvision torchaudio
pip install timm librosa scikit-learn matplotlib seaborn
```

---

## Datasets

### MKA Dataset
Download from: https://data.mendeley.com/datasets/p5kj7t9t6v  
Place at: `MKA datasets/` in the project root.

### Custom Dataset (this paper)
Download from Zenodo: [DOI to be assigned]  
Contains segmented WAV clips for:
- MacBook Pro E1 (clean) — 1,800 clips, 36 classes
- MacBook Pro E2 (window noise) — 1,800 clips, 36 classes  
- Dell Latitude 5430 E1 (clean) — 1,800 clips, 36 classes
- Dell Latitude 5430 E2 (window noise) — 1,800 clips, 36 classes

Place at: `~/Downloads/CustomDataset/`

---

## Reproduction

### Train on MKA
```bash
python train.py \
  --root "MKA datasets" \
  --platform all --model coatnet --epochs 100 --lr 1e-4
```

### Train on Custom Dataset
```bash
python train_custom.py --env E1 --model all --epochs 100 --lr 1e-4
python train_custom.py --env E2 --model all --epochs 100 --lr 1e-4
python train_custom.py --env both --model all --epochs 100 --lr 1e-4
```

### Statistical Validation (5 seeds)
```bash
# Custom dataset — stratified random split
python train_multiseed.py --dataset custom --env E1 --model coatnet --epochs 100

# MKA dataset — built-in stratified split
python train_multiseed.py --dataset mka --model coatnet --epochs 100
```

### Classical Baseline
```bash
python baseline_mfcc_svm.py --dataset custom --env E1
python baseline_mfcc_svm.py --dataset mka --root "MKA datasets"
```

### Cross-Dataset Experiments
```bash
python cross_dataset.py --experiment all --model coatnet --epochs 100
```

### Generate Figures
```bash
python plot_results.py \
  --results_dir results/clean_run \
  --custom_dir results/custom \
  --output_dir figures
```

---

## Recording Protocol (Custom Dataset)

- **Device:** iPhone Pro Max, 17cm to left of keyboard, on folded cloth
- **Keys:** 36 alphanumeric (0-9, a-z), 50 presses per key
- **Environments:**
  - E1: Quiet room, door closed
  - E2: Window open, outdoor ambient noise
- **Naming:** `{key}_{env}_{device}_{session}.m4a` e.g. `a_E1_01.m4a`
- **Conversion:** `afconvert -f WAVE -d LEI16 input.m4a output.wav`
- **Segmentation:** `python segment.py --input wav/ --output segmented/ --env E1`

---

## Citation

```bibtex
@article{rane2025acoustic,
  title={Acoustic Keystroke Inference: A Comparative Study of Hybrid Neural
         Network Architectures for Side-Channel Attacks},
  author={Rane, Vinit},
  journal={IEEE Access},
  year={2025},
  note={Under review}
}
```

---

## License

MIT License. See LICENSE for details.

## Ethical Statement

All keystroke recordings were self-collected by the author on personally owned devices. No third-party data was collected without consent. This research is disclosed for defensive awareness purposes only.
