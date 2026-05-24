# Experimental Results

All results reported in the paper. Single-run results use the best checkpoint by validation accuracy.
Statistical validation (mean ± std) computed across 5 random seeds [42, 123, 456, 789, 1337].

---

## MKA All-Platform Benchmark (36 classes, 36 samples/class)

| Model | Params | Best Val Acc | Test Acc | vs. MFCC+RF |
|-------|--------|-------------|----------|-------------|
| CoAtNet-0 | 26.7M | 50.52% | 50.26% | +7.70pp |
| MaxViT-Tiny | 30.4M | 67.53% | 54.87% | +12.31pp |
| Swin-Tiny | 27.5M | 5.15% | 4.10% | -38.46pp |
| MFCC+SVM | — | 28.87% | 36.41% | — |
| MFCC+RF | — | 30.41% | 42.56% | — |

---

## Custom MacBook Dataset (50 clips/class, 100 epochs)

| Model | Env | Best Val | Test Acc | Mean±Std (5 seeds) | Range |
|-------|-----|----------|----------|-------------------|-------|
| CoAtNet-0 | E1 Clean | 92.22% | 88.89% | **86.67% ± 2.29%** | [83.70%, 89.63%] |
| CoAtNet-0 | E2 Window | 95.19% | 89.63% | — | — |
| CoAtNet-0 | E1+E2 | 92.04% | 90.93% | — | — |
| MaxViT-Tiny | E1 Clean | 92.96% | 89.63% | **85.48% ± 0.98%** | [84.44%, 87.04%] |
| MaxViT-Tiny | E2 Window | 96.30% | 93.33% | — | — |
| MaxViT-Tiny | E1+E2 | 92.22% | **93.70%** | — | — |
| Swin-Tiny | E1 Clean | 81.48% | 78.15% | — | — |
| Swin-Tiny | E2 Window | 95.19% | 90.00% | — | — |
| Swin-Tiny | E1+E2 | 87.96% | 85.37% | — | — |
| MFCC+SVM | E1 | 81.85% | 80.74% | — | — |
| MFCC+RF | E2 | 85.93% | 82.59% | — | — |

---

## Cross-Dataset Generalization (CoAtNet-0)

| Experiment | Train | Test | Test Acc | Top-3 Acc |
|-----------|-------|------|----------|-----------|
| MKA → Custom E1 | MKA (1,296) | Custom E1 (1,800) | 2.94% | 7.94% |
| Custom E1 → MKA | Custom E1 (1,800) | MKA (1,296) | 2.31% | 8.80% |
| E1 → E2 (same keyboard) | MacBook E1 | MacBook E2 | 2.56% | 8.22% |
| MacBook → Dell | MacBook E1 | Dell E1 | TBD | TBD |

---

## Classical Baselines — Cross-Device Analysis

| Method | MacBook E1 | Dell E1 | MacBook E2 | Dell E2 |
|--------|-----------|---------|-----------|---------|
| MFCC+SVM | 80.74% | 80.74% | 75.19% | 75.19% |
| MFCC+RF | 75.19% | 75.19% | 82.59% | 82.59% |

**Finding:** MFCC features are keyboard-invariant — identical results across MacBook and Dell.

---

## Multi-Seed Per-Run Results

### CoAtNet-0 on Custom E1
| Seed | Val Acc | Test Acc |
|------|---------|----------|
| 42 | 88.15% | 86.30% |
| 123 | 88.15% | 84.81% |
| 456 | 81.85% | 83.70% |
| 789 | 88.89% | 88.89% |
| 1337 | 88.15% | 89.63% |
| **Mean ± Std** | **87.04% ± 2.61%** | **86.67% ± 2.29%** |

### MaxViT-Tiny on Custom E1
| Seed | Val Acc | Test Acc |
|------|---------|----------|
| 42 | 89.26% | 85.93% |
| 123 | 85.93% | 84.44% |
| 456 | 83.70% | 84.44% |
| 789 | 85.56% | 85.56% |
| 1337 | 84.44% | 87.04% |
| **Mean ± Std** | **85.78% ± 1.91%** | **85.48% ± 0.98%** |
