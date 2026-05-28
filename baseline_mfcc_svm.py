"""
MFCC + SVM Baseline Script
===========================
Thesis: Acoustic Keystroke Inference
Author: Shuchona Malek Orthi | Westcliff University

Implements classical MFCC + SVM baseline for comparison against
deep learning models. Standard in ASCA literature.

Usage:
    python baseline_mfcc_svm.py --dataset mka \
        --root "/Users/dominik/Downloads/Multi-Keyboard Acoustic (MKA) Datasets/MKA datasets"

    python baseline_mfcc_svm.py --dataset custom --env E1
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SR          = 44_100
N_MFCC      = 40
DURATION    = 1.0

ALPHANUMERIC = (
    [str(i) for i in range(10)] +
    [chr(c) for c in range(ord('a'), ord('z') + 1)]
)

ENV_PATHS = {
    "E1": Path.home() / "Downloads/CustomDataset/E1_clean/segmented",
    "E2": Path.home() / "Downloads/CustomDataset/E2_window/segmented",
}

# ── Feature extraction ────────────────────────────────────────────────────────

def extract_mfcc(path: Path, n_mfcc: int = N_MFCC) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SR, duration=DURATION, mono=True)
    if len(y) < int(SR * 0.1):
        return None
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y = y / max_amp
    mfcc      = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=n_mfcc)
    mfcc_d    = librosa.feature.delta(mfcc)
    mfcc_d2   = librosa.feature.delta(mfcc, order=2)
    features  = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        mfcc_d.mean(axis=1), mfcc_d2.mean(axis=1),
    ])
    return features.astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mka(root: str, max_per_class: int = 36) -> tuple:
    root = Path(root)
    wav_dir = root / "All Dataset" / "Sound Segment(wav)"
    X, y = [], []
    class_to_idx = {c: i for i, c in enumerate(ALPHANUMERIC)}

    for key_dir in sorted(wav_dir.iterdir()):
        if not key_dir.is_dir():
            continue
        cls = key_dir.name.lower()
        if cls not in class_to_idx:
            continue
        label = class_to_idx[cls]
        for wav in sorted(key_dir.glob("*.wav"))[:max_per_class]:
            feat = extract_mfcc(wav)
            if feat is not None:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)


def load_custom(env: str, max_per_class: int = 50) -> tuple:
    seg_dir = ENV_PATHS[env]
    X, y = [], []
    class_to_idx = {c: i for i, c in enumerate(ALPHANUMERIC)}

    for key_dir in sorted(seg_dir.iterdir()):
        if not key_dir.is_dir():
            continue
        cls = key_dir.name.lower()
        if cls not in class_to_idx:
            continue
        label = class_to_idx[cls]
        for wav in sorted(key_dir.glob("*.wav"))[:max_per_class]:
            feat = extract_mfcc(wav)
            if feat is not None:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y)


# ── Train and evaluate ────────────────────────────────────────────────────────

def run_baseline(X, y, dataset_name: str) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.176, random_state=42, stratify=y_train
    )

    print(f"\n{'='*55}")
    print(f"  MFCC+SVM Baseline — {dataset_name}")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Feature dim: {X.shape[1]}")
    print(f"{'='*55}")

    results = {}

    # SVM with RBF kernel
    print("\n[1/2] Training SVM (RBF kernel)...")
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", random_state=42))
    ])
    svm_pipe.fit(X_train, y_train)
    svm_val  = accuracy_score(y_val,  svm_pipe.predict(X_val))
    svm_test = accuracy_score(y_test, svm_pipe.predict(X_test))
    print(f"  SVM Val Acc  : {svm_val*100:.2f}%")
    print(f"  SVM Test Acc : {svm_test*100:.2f}%")
    results["svm"] = {"val_acc": round(svm_val, 4), "test_acc": round(svm_test, 4)}

    # Random Forest
    print("\n[2/2] Training Random Forest...")
    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_val  = accuracy_score(y_val,  rf_pipe.predict(X_val))
    rf_test = accuracy_score(y_test, rf_pipe.predict(X_test))
    print(f"  RF Val Acc   : {rf_val*100:.2f}%")
    print(f"  RF Test Acc  : {rf_test*100:.2f}%")
    results["random_forest"] = {"val_acc": round(rf_val, 4), "test_acc": round(rf_test, 4)}

    print(f"\n{'='*55}")
    print(f"  BASELINE SUMMARY — {dataset_name}")
    print(f"  MFCC+SVM  : {svm_test*100:.2f}% test accuracy")
    print(f"  MFCC+RF   : {rf_test*100:.2f}% test accuracy")
    print(f"  (Compare: CoAtNet DL = 88.89% on Custom E1)")
    print(f"{'='*55}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mka", "custom"], default="custom")
    parser.add_argument("--env", choices=["E1", "E2"], default="E1")
    parser.add_argument("--root", type=str,
                        default="/Users/dominik/Downloads/Multi-Keyboard Acoustic (MKA) Datasets/MKA datasets")
    parser.add_argument("--output_dir", type=str, default="./results/baseline")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.dataset == "mka":
        print("Loading MKA dataset...")
        X, y = load_mka(args.root)
        name = "MKA All-Platform"
    else:
        print(f"Loading Custom {args.env} dataset...")
        X, y = load_custom(args.env)
        name = f"Custom {args.env}"

    print(f"Loaded {len(X)} samples, {X.shape[1]} features, {len(set(y))} classes")
    results = run_baseline(X, y, name)

    out_path = Path(args.output_dir) / f"baseline_{args.dataset}_{args.env}.json"
    with open(out_path, "w") as f:
        json.dump({"dataset": name, "results": results}, f, indent=2)
    print(f"[Saved] {out_path}")
