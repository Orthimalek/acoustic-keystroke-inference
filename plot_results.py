"""
Thesis Figures & Analysis Script
==================================
Thesis: Acoustic Keystroke Inference
Author: Vinit Rane | CSUDH CYB Program

Generates all publication-quality figures for the paper:
  Fig 1 — System pipeline diagram
  Fig 2 — Mel-spectrogram examples per key and environment
  Fig 3 — Training curves (loss + accuracy) for all models on MKA
  Fig 4 — Training curves on Custom E1
  Fig 5 — Confusion matrix (best model on Custom E1)
  Fig 6 — Bar chart comparing all model results
  Fig 7 — Top-k accuracy comparison

Usage:
    python plot_results.py --results_dir results/clean_run \
                           --custom_dir results/custom \
                           --output_dir figures
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "coatnet": "#2196F3",
    "maxvit":  "#4CAF50",
    "swin":    "#F44336",
    "mka":     "#9C27B0",
    "e1":      "#FF9800",
    "e2":      "#00BCD4",
    "both":    "#795548",
}

MODEL_LABELS = {
    "coatnet": "CoAtNet-0",
    "maxvit":  "MaxViT-Tiny",
    "swin":    "Swin-Tiny",
}


# ── Helper ────────────────────────────────────────────────────────────────────

def load_history(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_fig(fig, output_dir: Path, name: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Saved] {path}")
    return path


# ── Figure 1: System Pipeline ─────────────────────────────────────────────────

def plot_pipeline(output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    steps = [
        ("Keyboard\nTyping",      "#E3F2FD", "#1565C0", 0.7),
        ("Smartphone\nMicrophone","#E8F5E9", "#2E7D32", 2.3),
        ("Onset\nDetection",      "#FFF3E0", "#E65100", 3.9),
        ("Mel-\nSpectrogram",     "#FCE4EC", "#880E4F", 5.5),
        ("Deep Learning\nModel",  "#EDE7F6", "#4527A0", 7.3),
        ("Key\nPrediction",       "#E0F7FA", "#006064", 9.1),
        ("Password\nRecovery",    "#F3E5F5", "#6A1B9A", 10.9),
    ]

    for i, (label, facecolor, edgecolor, x) in enumerate(steps):
        box = mpatches.FancyBboxPatch(
            (x, 1.0), 1.4, 2.0,
            boxstyle="round,pad=0.1",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + 0.7, 2.0, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color=edgecolor)

        if i < len(steps) - 1:
            ax.annotate("", xy=(steps[i+1][3], 2.0), xytext=(x + 1.4, 2.0),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

    ax.text(7.0, 0.3,
            "Attack Pipeline: Record → Segment → Feature Extract → Classify → Reconstruct",
            ha="center", va="center", fontsize=9, style="italic", color="#555555")

    ax.set_title("Fig. 1: Acoustic Keystroke Inference System Pipeline", fontweight="bold", pad=10)
    return save_fig(fig, output_dir, "fig1_pipeline")


# ── Figure 2: Mel-Spectrogram Examples ───────────────────────────────────────

def plot_spectrograms(output_dir: Path):
    try:
        import librosa
        import librosa.display
    except ImportError:
        print("[Skip] librosa not available for spectrogram plot")
        return None

    e1_dir = Path.home() / "Downloads/CustomDataset/E1_clean/segmented"
    e2_dir = Path.home() / "Downloads/CustomDataset/E2_window/segmented"

    sample_keys = ["a", "s", "d", "f"]
    fig, axes = plt.subplots(2, len(sample_keys), figsize=(12, 5))

    for col, key in enumerate(sample_keys):
        for row, (env_dir, env_label) in enumerate([(e1_dir, "E1 Clean"), (e2_dir, "E2 Window")]):
            key_dir = env_dir / key
            if not key_dir.exists():
                axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center")
                continue
            wavs = sorted(key_dir.glob("*.wav"))
            if not wavs:
                axes[row, col].text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            y, sr = librosa.load(str(wavs[0]), sr=44100, duration=0.5)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=225)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            img = librosa.display.specshow(
                mel_db, sr=sr, hop_length=225, x_axis="time", y_axis="mel",
                ax=axes[row, col], cmap="magma"
            )
            axes[row, col].set_title(f"Key '{key}' — {env_label}", fontsize=9)
            axes[row, col].set_xlabel("")
            if col > 0:
                axes[row, col].set_ylabel("")

    fig.suptitle("Fig. 2: Mel-Spectrogram Examples — Keys 'a','s','d','f' in E1 vs E2",
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    return save_fig(fig, output_dir, "fig2_spectrograms")


# ── Figure 3: MKA Training Curves ────────────────────────────────────────────

def plot_training_curves_mka(results_dir: Path, output_dir: Path):
    models = ["coatnet", "maxvit", "swin"]
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    for idx, model in enumerate(models):
        hist_path = results_dir / f"{model}_all_history.json"
        if not hist_path.exists():
            print(f"[Skip] {hist_path} not found")
            continue

        hist = load_history(hist_path)
        ax   = fig.add_subplot(gs[idx])
        epochs = range(1, len(hist["train_acc"]) + 1)

        ax.plot(epochs, hist["train_acc"], color=COLORS[model], lw=1.5, label="Train", alpha=0.8)
        ax.plot(epochs, hist["val_acc"],   color=COLORS[model], lw=2.0, label="Val", linestyle="--")

        best_ep = hist["best_epoch"]
        best_val = hist["best_val_acc"]
        ax.axvline(best_ep, color="gray", linestyle=":", alpha=0.6, lw=1)
        ax.scatter([best_ep], [best_val], color=COLORS[model], zorder=5, s=50)
        ax.annotate(f"{best_val*100:.1f}%", xy=(best_ep, best_val),
                    xytext=(best_ep + len(epochs)*0.05, best_val - 0.05),
                    fontsize=8, color=COLORS[model])

        ax.set_title(f"{MODEL_LABELS[model]}\n(MKA All-Platform)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Fig. 3: Training and Validation Accuracy — MKA Dataset (lr=1e-4, pretrained)",
                 fontweight="bold", y=1.02)
    return save_fig(fig, output_dir, "fig3_mka_training_curves")


# ── Figure 4: Custom E1 Training Curves ──────────────────────────────────────

def plot_training_curves_custom(custom_dir: Path, output_dir: Path):
    models = ["coatnet", "maxvit", "swin"]
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.35)

    for idx, model in enumerate(models):
        hist_path = custom_dir / f"{model}_E1_history.json"
        if not hist_path.exists():
            print(f"[Skip] {hist_path} not found")
            ax = fig.add_subplot(gs[idx])
            ax.text(0.5, 0.5, "Training\nin progress",
                    ha="center", va="center", fontsize=11, color="gray",
                    transform=ax.transAxes)
            ax.set_title(f"{MODEL_LABELS.get(model, model)}\n(Custom E1)", fontweight="bold")
            continue

        hist = load_history(hist_path)
        ax   = fig.add_subplot(gs[idx])
        epochs = range(1, len(hist["train_acc"]) + 1)

        ax.plot(epochs, hist["train_acc"], color=COLORS[model], lw=1.5, label="Train", alpha=0.8)
        ax.plot(epochs, hist["val_acc"],   color=COLORS[model], lw=2.0, label="Val",   linestyle="--")

        best_ep  = hist.get("best_epoch", 0)
        best_val = hist.get("best_val_acc", 0)
        if best_ep > 0:
            ax.axvline(best_ep, color="gray", linestyle=":", alpha=0.6, lw=1)
            ax.scatter([best_ep], [best_val], color=COLORS[model], zorder=5, s=50)
            ax.annotate(f"{best_val*100:.1f}%", xy=(best_ep, best_val),
                        xytext=(best_ep + 2, best_val - 0.06), fontsize=8, color=COLORS[model])

        test_acc = hist.get("test_acc", None)
        if test_acc:
            ax.axhline(test_acc, color="black", linestyle="-.", alpha=0.5, lw=1)
            ax.text(len(epochs)*0.02, test_acc + 0.01, f"Test: {test_acc*100:.1f}%",
                    fontsize=8, color="black")

        ax.set_title(f"{MODEL_LABELS.get(model, model)}\n(Custom E1)", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Fig. 4: Training and Validation Accuracy — Custom E1 Dataset (50 clips/class)",
                 fontweight="bold", y=1.02)
    return save_fig(fig, output_dir, "fig4_custom_training_curves")


# ── Figure 5: Results Bar Chart ───────────────────────────────────────────────

def plot_results_bar(output_dir: Path):
    # Known results — update when all training complete
    data = {
        "CoAtNet-0": {
            "MKA":        50.26,
            "Custom E1":  88.89,
            "Custom E2":  None,
            "E1+E2":      None,
        },
        "MaxViT-Tiny": {
            "MKA":        54.87,
            "Custom E1":  None,
            "Custom E2":  None,
            "E1+E2":      None,
        },
        "Swin-Tiny": {
            "MKA":        4.10,
            "Custom E1":  None,
            "Custom E2":  None,
            "E1+E2":      None,
        },
    }

    datasets  = ["MKA", "Custom E1", "Custom E2", "E1+E2"]
    models    = list(data.keys())
    x         = np.arange(len(datasets))
    width     = 0.25
    fig, ax   = plt.subplots(figsize=(12, 6))

    model_colors = [COLORS["coatnet"], COLORS["maxvit"], COLORS["swin"]]

    for i, (model, color) in enumerate(zip(models, model_colors)):
        values  = [data[model][d] for d in datasets]
        x_pos   = x + (i - 1) * width

        bars = ax.bar(
            x_pos,
            [v if v is not None else 0 for v in values],
            width, label=model, color=color, alpha=0.85, edgecolor="white", linewidth=0.5
        )

        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color=color)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 3,
                        "TBD", ha="center", va="bottom", fontsize=7,
                        color="gray", style="italic")

    ax.axhline(y=95.0, color="#B71C1C", linestyle="--", alpha=0.6, lw=1.5,
               label="Harrison et al. [1] — 95%")
    ax.axhline(y=98.3, color="#E65100", linestyle=":", alpha=0.6, lw=1.5,
               label="Spata et al. [6] — 98.3%")

    ax.set_xlabel("Dataset / Environment")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Fig. 5: Model Test Accuracy Across Datasets\n(Gray bars = results pending training completion)",
                 fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 108)
    ax.legend(loc="upper right", fontsize=9)

    ax.text(3.4, 96.5, "Harrison\n95%",  fontsize=7, color="#B71C1C", ha="center")
    ax.text(3.4, 99.8, "Spata\n98.3%",   fontsize=7, color="#E65100", ha="center")

    return save_fig(fig, output_dir, "fig5_results_comparison")


# ── Figure 6: Confusion Matrix ────────────────────────────────────────────────

def plot_confusion_matrix_from_model(custom_dir: Path, output_dir: Path):
    """
    Loads saved predictions from training and plots confusion matrix.
    Falls back to placeholder if predictions not saved.
    """
    pred_path = custom_dir / "coatnet_E1_predictions.json"
    if not pred_path.exists():
        print(f"[Info] Confusion matrix predictions not found at {pred_path}")
        print("[Info] Run training with --save_predictions flag to generate.")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5,
                "Confusion matrix will be generated\nafter training completes.\n\nRun: python train_custom.py --env E1 --model coatnet",
                ha="center", va="center", fontsize=12, transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))
        ax.set_title("Fig. 6: Confusion Matrix — CoAtNet-0 on Custom E1 (Pending)", fontweight="bold")
        ax.axis("off")
        return save_fig(fig, output_dir, "fig6_confusion_matrix")

    with open(pred_path) as f:
        data = json.load(f)

    preds  = data["predictions"]
    labels = data["labels"]

    CLASSES = [str(i) for i in range(10)] + [chr(c) for c in range(ord('a'), ord('z')+1)]
    cm = np.zeros((36, 36), dtype=int)
    for p, l in zip(preds, labels):
        cm[l][p] += 1

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_norm, annot=False, fmt=".0%", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES,
                linewidths=0.3, ax=ax, cbar_kws={"label": "Normalized Count"})

    ax.set_xlabel("Predicted Key", fontweight="bold")
    ax.set_ylabel("True Key", fontweight="bold")
    ax.set_title("Fig. 6: Confusion Matrix — CoAtNet-0 on Custom E1 Dataset", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, output_dir, "fig6_confusion_matrix")


# ── Figure 7: Top-k Accuracy ─────────────────────────────────────────────────

def plot_topk(output_dir: Path):
    """
    Plots top-k accuracy for available models.
    Update topk_data with actual values once experiments run.
    """
    topk_data = {
        "CoAtNet-0 (MKA)":     {1: 50.26, 3: None, 5: None},
        "MaxViT-Tiny (MKA)":   {1: 54.87, 3: None, 5: None},
        "CoAtNet-0 (E1)":      {1: 88.89, 3: None, 5: None},
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    k_vals = [1, 3, 5]
    colors = [COLORS["coatnet"], COLORS["maxvit"], "#FF9800"]

    for (label, kdata), color in zip(topk_data.items(), colors):
        vals = [kdata.get(k) for k in k_vals]
        known = [(k, v) for k, v in zip(k_vals, vals) if v is not None]
        if known:
            ks, vs = zip(*known)
            ax.plot(ks, vs, "o-", color=color, lw=2, markersize=8, label=label)
            for k, v in zip(ks, vs):
                ax.annotate(f"{v:.1f}%", xy=(k, v), xytext=(k, v+1.5),
                            ha="center", fontsize=9, color=color)

    ax.axhline(97, color="gray", linestyle="--", alpha=0.5, lw=1)
    ax.text(1.1, 97.5, "Harrison et al. Top-3: 97%", fontsize=8, color="gray")

    ax.set_xlabel("k (Top-k Accuracy)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks([1, 3, 5])
    ax.set_xticklabels(["Top-1", "Top-3", "Top-5"])
    ax.set_ylim(0, 105)
    ax.set_title("Fig. 7: Top-k Accuracy Comparison\n(Top-3/5 values pending cross-dataset evaluation)",
                 fontweight="bold")
    ax.legend(fontsize=9)
    return save_fig(fig, output_dir, "fig7_topk_accuracy")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results/clean_run")
    parser.add_argument("--custom_dir",  type=str, default="./results/custom")
    parser.add_argument("--output_dir",  type=str, default="./figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    custom_dir  = Path(args.custom_dir)
    output_dir  = Path(args.output_dir)

    print(f"\n{'='*50}")
    print(f"  Generating thesis figures")
    print(f"  Output: {output_dir}")
    print(f"{'='*50}\n")

    plot_pipeline(output_dir)
    plot_spectrograms(output_dir)
    plot_training_curves_mka(results_dir, output_dir)
    plot_training_curves_custom(custom_dir, output_dir)
    plot_results_bar(output_dir)
    plot_confusion_matrix_from_model(custom_dir, output_dir)
    plot_topk(output_dir)

    print(f"\n{'='*50}")
    print(f"  All figures saved to: {output_dir}")
    print(f"  Copy to paper folder and reference in LaTeX/Word")
    print(f"{'='*50}\n")
