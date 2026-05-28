# Acoustic Keystroke Inference
### A Comparative Study of Hybrid Neural Network Architectures for Side-Channel Attacks

**First Author:** Dipta Roy — Dept. of ECE, California State University, Northridge  
**Corresponding Author:** Shuchona Malek Orthi — MS IT Project Management, Westcliff University, Irvine, CA, USA  
**ORCID:** [0009-0007-5397-4561](https://orcid.org/0009-0007-5397-4561)  
**Email:** s.orthi.339@westcliff.edu  
**Paper:** IEEE Access — Under Review  
**Dataset DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20369332.svg)](https://doi.org/10.5281/zenodo.20369332)

---

## Authors

| Name | Affiliation | ORCID |
|------|-------------|-------|
| Dipta Roy *(First Author)* | Westcliff University, Irvine, CA, USA | [0009-0007-5397-4561](https://orcid.org/0009-0007-5397-4561) |
| **Shuchona Malek Orthi** *(Corresponding Author)* | Westcliff University, Irvine, CA, USA | [0009-0007-5397-4561](https://orcid.org/0009-0007-5397-4561) |
| Dipta Roy | Dept. of ECE, California State University, Northridge, CA, USA | [0009-0006-7065-6146](https://orcid.org/0009-0006-7065-6146) |
| Kazi Mushfiq Rafid | Dept. of ECE, California State University, Northridge, CA, USA | [0009-0003-5271-2659](https://orcid.org/0009-0003-5271-2659) |
| Muhammad Safwat Rahman | Southeast Missouri State University, Cape Girardeau, MO, USA | [0009-0008-4200-0187](https://orcid.org/0009-0008-4200-0187) |
| Mohammad Arafath Uz Zaman Khan | Dept. of CS, California State University, Dominguez Hills, CA, USA | [0009-0004-1046-763X](https://orcid.org/0009-0004-1046-763X) |
| Tarik Hossain | Dept. of IT, Sikkim Manipal University, Sikkim, India | — |

---

## Overview

This repository contains all code, scripts, and results for the acoustic keystroke inference study comparing **CoAtNet-0**, **MaxViT-Tiny**, and **Swin-Tiny** for keyboard side-channel attack classification across multiple environments and keyboard types.

**Key results (5-seed statistical validation):**

| Model | MKA Test Mean±Std | Custom E1 Mean±Std | Custom E1+E2 |
|-------|-------------------|-------------------|--------------|
| CoAtNet-0 | 50.77% ± 3.51% | 86.67% ± 2.29% | 90.93% |
| MaxViT-Tiny | 56.21% ± 3.75% | 85.48% ± 0.98% | **93.70%** |
| Swin-Tiny | 4.10% (MKA) / 90.00% (E2) | 78.15% | 85.37% |
| MFCC+SVM | 36.41% | 80.74% ± 2.11% | — |

**Cross-dataset generalization (all near-random — confirms environment specificity):**

| Transfer Direction | Test Acc | Top-3 |
|-------------------|----------|-------|
| MKA → Custom E1 | 2.94% | 7.94% |
| Custom E1 → MKA | 2.31% | 8.80% |
| E1 → E2 (same keyboard) | 2.56% | 8.22% |
| MacBook → Dell | 1.67% | 7.72% |
| Dell → MacBook | 2.72% | 7.78% |

---

## Repository Structure

```
├── mka_dataloader.py         # MKA dataset loader (6 keyboard platforms)
├── custom_dataloader.py      # Custom E1/E2 dataset loader
├── train.py                  # Train on MKA dataset
├── train_custom.py           # Train on custom dataset (supports --no_pretrain)
├── train_multiseed.py        # Multi-seed statistical validation (v2, stratified)
├── segment.py                # Onset-based keystroke segmentation
├── baseline_mfcc_svm.py      # MFCC + SVM/RF classical baseline
├── verify_mfcc.py            # MFCC cross-keyboard verification (3 seeds)
├── cross_dataset.py          # Cross-dataset generalization experiments
├── plot_results.py           # Generate publication figures
├── results/                  # Training history JSON files
│   ├── clean_run/            # MKA results
│   ├── custom/               # Custom dataset results + predictions
│   ├── multiseed/            # Multi-seed validation results
│   ├── cross_dataset/        # Cross-dataset experiment results
│   ├── scratch/              # From-scratch ablation results
│   └── baseline/             # MFCC baseline + verification results
└── figures/                  # Publication figures (PNG, 200 DPI)
```

---

## Setup

```bash
# Create conda environment
conda create -n acoustic python=3.10
conda activate acoustic

# Install dependencies
pip install torch torchvision torchaudio
pip install timm librosa scikit-learn matplotlib seaborn scipy
```

---

## Datasets

### MKA Dataset
Download from: https://data.mendeley.com/datasets/p5kj7t9t6v  
Place at: `MKA datasets/` in the project root.

### Custom Dataset (this paper)
Download from Zenodo: **https://doi.org/10.5281/zenodo.20369332**  
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
# With ImageNet pretrained weights
python train_custom.py --env E1 --model all --epochs 100 --lr 1e-4

# From scratch (ablation — no pretrained weights)
python train_custom.py --env E1 --model all --epochs 100 --lr 1e-4 --no_pretrain
```

### Statistical Validation (5 seeds)
```bash
# Custom dataset
python train_multiseed.py --dataset custom --env E1 --model coatnet --epochs 100

# MKA dataset (stratified split)
python train_multiseed.py --dataset mka --model coatnet --epochs 100
```

### Classical Baseline + MFCC Verification
```bash
python baseline_mfcc_svm.py --dataset custom --env E1
python verify_mfcc.py   # 3-seed cross-keyboard verification
```

### Cross-Dataset Generalization
```bash
# All directions
python cross_dataset.py --experiment all --model coatnet --epochs 100

# Single direction
python cross_dataset.py --experiment dell_to_macbook --model coatnet --epochs 100
```

---

## Recording Protocol

- **Device:** iPhone Pro Max, 17cm to left of keyboard on folded cloth
- **Keys:** 36 alphanumeric (0–9, a–z), 50 presses per key
- **Environments:** E1 = quiet room, E2 = window open (outdoor ambient noise)
- **Naming:** `{key}_{env}_{session}.m4a` e.g. `a_E1_01.m4a`
- **Conversion:** `afconvert -f WAVE -d LEI16 input.m4a output.wav`
- **Segmentation:** `python segment.py --input wav/ --output segmented/ --env E1`

---

## Citation

```bibtex
@article{orthi2025acoustic,
  title   = {Acoustic Keystroke Inference: A Comparative Study of Hybrid
             Neural Network Architectures for Side-Channel Attacks},
  author  = {Roy, Dipta and Rafid, Kazi Mushfiq and Rahman, Muhammad Safwat
             and Khan, Mohammad Arafath Uz Zaman and Hossain, Tarik
             and Orthi, Shuchona Malek
},
  journal = {IEEE Access},
  year    = {2025},
  note    = {Under review},
  doi     = {10.5281/zenodo.20369332}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Ethical Statement

All keystroke recordings were self-collected by the authors on personally owned devices. No third-party data was collected without consent. This research is disclosed for defensive awareness purposes only.
