"""
Cross-Dataset Generalization Experiment
=========================================
Thesis: Acoustic Keystroke Inference
Author: Vinit Rane | CSUDH CYB Program

Evaluates model generalization across datasets:
  1. Train on MKA → Test on Custom E1
  2. Train on Custom E1 → Test on MKA
  3. Train on MacBook (Custom) → Test on Dell (when available)
  4. Train on E1 → Test on E2 (environment transfer)

Usage:
    python cross_dataset.py --experiment all \
        --root "/Users/dominik/Downloads/Multi-Keyboard Acoustic (MKA) Datasets/MKA datasets" \
        --model coatnet
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

try:
    import timm
except ImportError:
    raise ImportError("Run: python -m pip install timm")

import sys
sys.path.append(str(Path(__file__).parent))
from mka_dataloader import MKADataset, PLATFORMS
from custom_dataloader import CustomKeystrokeDataset, ENV_PATHS, ALPHANUMERIC_CLASSES

# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("[Device] Apple M1 MPS")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(name: str, num_classes: int = 36) -> nn.Module:
    name = name.lower()
    if name == "coatnet":
        m = timm.create_model("coatnet_0_rw_224", pretrained=True, num_classes=num_classes)
    elif name == "maxvit":
        m = timm.create_model("maxvit_tiny_tf_224.in1k", pretrained=True, num_classes=num_classes)
    elif name == "swin":
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[Model] {name} | Params: {params:,}")
    return m


# ── Train / Eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    if scheduler:
        scheduler.step()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


@torch.no_grad()
def topk_accuracy(model, loader, device, k_values=[1, 3, 5]):
    model.eval()
    results = {k: 0 for k in k_values}
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        for k in k_values:
            topk = outputs.topk(k, dim=1).indices
            correct = topk.eq(labels.unsqueeze(1).expand_as(topk)).any(dim=1).sum().item()
            results[k] += correct
        total += images.size(0)
    return {k: round(v / total, 4) for k, v in results.items()}


# ── Cross-dataset experiment ──────────────────────────────────────────────────

def run_cross_experiment(
    experiment_name: str,
    train_ds,
    test_ds,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
) -> Dict:
    device = get_device()

    from torch.utils.data import random_split
    n = len(train_ds)
    n_train = int(n * 0.85)
    n_val   = n - n_train
    generator = torch.Generator().manual_seed(42)
    train_split, val_split = random_split(train_ds, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_split, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_split,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,     batch_size=batch_size, shuffle=False, num_workers=0)

    model     = build_model(model_name, num_classes=36).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_ckpt = output_dir / f"{experiment_name}_{model_name}_best.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Experiment : {experiment_name}")
    print(f"  Model      : {model_name.upper()}")
    print(f"  Train size : {len(train_split)} | Val: {len(val_split)} | Test: {len(test_ds)}")
    print(f"{'='*60}")

    start = time.time()
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        v_loss, v_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), best_ckpt)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"  Ep {epoch:4d}/{epochs} | Train: {t_acc:.4f} | Val: {v_acc:.4f} | Best Val: {best_val_acc:.4f} | {elapsed:.0f}s")

    # Final test evaluation
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    topk = topk_accuracy(model, test_loader, device)

    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET TEST ACCURACY : {test_acc*100:.2f}%")
    print(f"  Top-1: {topk[1]*100:.2f}%  Top-3: {topk[3]*100:.2f}%  Top-5: {topk[5]*100:.2f}%")
    print(f"{'='*60}\n")

    result = {
        "experiment": experiment_name,
        "model": model_name,
        "best_val_acc": round(best_val_acc, 4),
        "test_acc": round(test_acc, 4),
        "test_loss": round(test_loss, 4),
        "topk": topk,
        "predictions": preds,
        "labels": labels,
    }

    out_path = output_dir / f"{experiment_name}_{model_name}_result.json"
    save_result = {k: v for k, v in result.items() if k not in ["predictions", "labels"]}
    with open(out_path, "w") as f:
        json.dump(save_result, f, indent=2)
    print(f"[Saved] {out_path}")
    return result


# ── Experiment definitions ────────────────────────────────────────────────────

def get_mka_dataset(root: str):
    return MKADataset(root=root, platform="all", task="alphanumeric", augment=False)

def get_custom_dataset(env: str, max_clips: int = 50):
    return CustomKeystrokeDataset(ENV_PATHS[env], max_clips=max_clips, augment=False)

def get_dell_dataset(env: str, max_clips: int = 50):
    dell_path = Path.home() / f"Downloads/CustomDataset/Dell_{env}_clean/segmented" if env == "E1" else \
                Path.home() / f"Downloads/CustomDataset/Dell_{env}_window/segmented"
    if not dell_path.exists():
        print(f"[Skip] Dell {env} dataset not found at {dell_path}")
        return None
    return CustomKeystrokeDataset(dell_path, max_clips=max_clips, augment=False)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=["mka_to_custom", "custom_to_mka",
                                 "e1_to_e2", "macbook_to_dell", "all"])
    parser.add_argument("--model",      default="coatnet",
                        choices=["coatnet", "maxvit", "swin", "all"])
    parser.add_argument("--root",       type=str,
                        default="/Users/dominik/Downloads/Multi-Keyboard Acoustic (MKA) Datasets/MKA datasets")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str,   default="./results/cross_dataset")
    args = parser.parse_args()

    output_dir  = Path(args.output_dir)
    models      = ["coatnet", "maxvit"] if args.model == "all" else [args.model]
    experiments = []

    if args.experiment in ["mka_to_custom", "all"]:
        mka_ds     = get_mka_dataset(args.root)
        custom_e1  = get_custom_dataset("E1")
        experiments.append(("MKA_to_CustomE1", mka_ds, custom_e1))

    if args.experiment in ["custom_to_mka", "all"]:
        custom_e1  = get_custom_dataset("E1")
        mka_ds     = get_mka_dataset(args.root)
        experiments.append(("CustomE1_to_MKA", custom_e1, mka_ds))

    if args.experiment in ["e1_to_e2", "all"]:
        e1_ds = get_custom_dataset("E1")
        e2_ds = get_custom_dataset("E2")
        experiments.append(("E1_to_E2", e1_ds, e2_ds))

    if args.experiment in ["macbook_to_dell", "all"]:
        macbook = get_custom_dataset("E1")
        dell    = get_dell_dataset("E1")
        if dell is not None:
            experiments.append(("MacBook_to_Dell", macbook, dell))

    all_results = []
    for exp_name, train_ds, test_ds in experiments:
        for model_name in models:
            result = run_cross_experiment(
                exp_name, train_ds, test_ds,
                model_name, args.epochs, args.batch_size, args.lr, output_dir
            )
            all_results.append(result)

    # Summary table
    print("\n" + "="*70)
    print(f"  {'Experiment':<25} {'Model':<10} {'Val Acc':>9} {'Test Acc':>9} {'Top-3':>8}")
    print("-"*70)
    for r in all_results:
        print(f"  {r['experiment']:<25} {r['model']:<10} "
              f"{r['best_val_acc']*100:>8.2f}%  "
              f"{r['test_acc']*100:>8.2f}%  "
              f"{r['topk'][3]*100:>7.2f}%")
    print("="*70)

    summary_path = output_dir / "cross_dataset_summary.json"
    save_all = [{k: v for k, v in r.items() if k not in ["predictions", "labels"]}
                for r in all_results]
    with open(summary_path, "w") as f:
        json.dump(save_all, f, indent=2)
    print(f"\n[Saved] {summary_path}")
