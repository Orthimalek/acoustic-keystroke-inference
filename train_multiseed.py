"""
Multi-Seed Training Script — Statistical Validation
=====================================================
Thesis: Acoustic Keystroke Inference
Author: Shuchona Malek Orthi | Westcliff University

Runs each model × dataset combination across 5 random seeds.
Reports mean ± std for all accuracy metrics.
Required for IEEE Access / TIFS submission.

Key fix (v2): MKA dataset uses stratified splitting via its built-in
build_dataloaders() to ensure each class is represented in all splits.
Custom datasets use random_split (balanced by design at 50 clips/class).

Usage:
    python train_multiseed.py --dataset custom --env E1 --model coatnet --epochs 100
    python train_multiseed.py --dataset mka --model coatnet --epochs 100
    python train_multiseed.py --dataset mka --model all --epochs 100
    python train_multiseed.py --dataset custom --env E1 --model all --epochs 100
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

try:
    import timm
except ImportError:
    raise ImportError("Run: python -m pip install timm")

import sys
sys.path.append(str(Path(__file__).parent))
from mka_dataloader import MKADataset, build_dataloaders, PLATFORMS
from custom_dataloader import CustomKeystrokeDataset, ENV_PATHS, ALPHANUMERIC_CLASSES

# ── Device ──────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ── Model ────────────────────────────────────────────────────────────────────

def build_model(name: str, num_classes: int = 36) -> nn.Module:
    name = name.lower()
    if name == "coatnet":
        return timm.create_model("coatnet_0_rw_224",         pretrained=True, num_classes=num_classes)
    elif name == "maxvit":
        return timm.create_model("maxvit_tiny_tf_224.in1k",  pretrained=True, num_classes=num_classes)
    elif name == "swin":
        return timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")

# ── Train / Eval helpers ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        criterion(model(images), labels).backward()
        optimizer.step()
    scheduler.step()

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total   += images.size(0)
    return correct / total

# ── MKA stratified split per seed ───────────────────────────────────────────

def get_mka_loaders(root: str, seed: int, batch_size: int):
    """
    Uses MKA's built-in stratified build_dataloaders with the seed
    controlling torch/numpy state before the call.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_dl, val_dl, test_dl, dataset = build_dataloaders(
        root=root,
        platform="all",
        task="alphanumeric",
        batch_size=batch_size,
        num_workers=0,
        seed=seed,
    )
    return train_dl, val_dl, test_dl

# ── Custom random split per seed ─────────────────────────────────────────────

def get_custom_loaders(seg_path, seed: int, batch_size: int, max_clips: int = 50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    full_ds = CustomKeystrokeDataset(seg_path, max_clips=max_clips, augment=False)
    n       = len(full_ds)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    n_test  = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test], generator=g)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dl, val_dl, test_dl

# ── Single seed run ──────────────────────────────────────────────────────────

def run_one_seed(
    seed: int,
    model_name: str,
    loader_fn,       # callable(seed) -> (train_dl, val_dl, test_dl)
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
    run_label: str,
) -> Dict:
    device    = get_device()
    train_dl, val_dl, test_dl = loader_fn(seed)

    model     = build_model(model_name, num_classes=36).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    ckpt_path    = output_dir / f"{run_label}_seed{seed}_best.pt"

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, train_dl, optimizer, criterion, device, scheduler)
        val_acc = eval_acc(model, val_dl, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Seed {seed} | Ep {epoch:4d}/{epochs} | Val: {val_acc:.4f} | Best: {best_val_acc:.4f}")

    # Test with best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc = eval_acc(model, test_dl, device)

    return {
        "seed":         seed,
        "best_val_acc": round(best_val_acc, 4),
        "test_acc":     round(test_acc,     4),
    }

# ── Multi-seed runner ─────────────────────────────────────────────────────────

def run_multiseed(
    model_name: str,
    loader_fn,
    seeds: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
    run_label: str,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Multi-seed: {model_name.upper()} | {run_label}")
    print(f"  Seeds: {seeds} | Epochs: {epochs} | LR: {lr}")
    print(f"{'='*60}")

    results = []
    start   = time.time()

    for seed in seeds:
        print(f"\n  ── Seed {seed} ──────────────────────────────")
        r = run_one_seed(seed, model_name, loader_fn,
                         epochs, batch_size, lr, output_dir, run_label)
        results.append(r)
        print(f"  Seed {seed} done: Val={r['best_val_acc']:.4f}  Test={r['test_acc']:.4f}")

    val_accs  = [r["best_val_acc"] for r in results]
    test_accs = [r["test_acc"]     for r in results]

    summary = {
        "model":        model_name,
        "run_label":    run_label,
        "seeds":        seeds,
        "epochs":       epochs,
        "lr":           lr,
        "per_seed":     results,
        "val_mean":     round(float(np.mean(val_accs)),  4),
        "val_std":      round(float(np.std(val_accs)),   4),
        "test_mean":    round(float(np.mean(test_accs)), 4),
        "test_std":     round(float(np.std(test_accs)),  4),
        "test_min":     round(float(np.min(test_accs)),  4),
        "test_max":     round(float(np.max(test_accs)),  4),
        "total_time_s": round(time.time() - start, 0),
    }

    print(f"\n{'='*60}")
    print(f"  MULTI-SEED RESULTS: {model_name.upper()} | {run_label}")
    print(f"  Val  : {summary['val_mean']*100:.2f}% ± {summary['val_std']*100:.2f}%")
    print(f"  Test : {summary['test_mean']*100:.2f}% ± {summary['test_std']*100:.2f}%")
    print(f"  Range: [{summary['test_min']*100:.2f}%, {summary['test_max']*100:.2f}%]")
    print(f"  Time : {summary['total_time_s']/3600:.1f} hours")
    print(f"{'='*60}\n")

    out_path = output_dir / f"{run_label}_{model_name}_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Saved] {out_path}")
    return summary

# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary_table(summaries: List[Dict]):
    print("\n" + "="*78)
    print(f"  {'Model':<12} {'Dataset':<15} {'Val Mean±Std':>15} {'Test Mean±Std':>15} {'Range':>18}")
    print("-"*78)
    for s in summaries:
        print(
            f"  {s['model']:<12} {s['run_label']:<15} "
            f"{s['val_mean']*100:>6.2f}%±{s['val_std']*100:<6.2f}%  "
            f"{s['test_mean']*100:>6.2f}%±{s['test_std']*100:<6.2f}%  "
            f"[{s['test_min']*100:.2f}%, {s['test_max']*100:.2f}%]"
        )
    print("="*78 + "\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="custom", choices=["mka", "custom", "both"])
    parser.add_argument("--env",        default="E1",     choices=["E1", "E2"])
    parser.add_argument("--model",      default="coatnet",choices=["coatnet", "maxvit", "swin", "all"])
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--max_clips",  type=int,   default=50)
    parser.add_argument("--seeds",      type=str,   default="42,123,456,789,1337")
    parser.add_argument("--output_dir", type=str,   default="./results/multiseed")
    parser.add_argument("--root",       type=str,
                        default="/Users/dominik/Downloads/Multi-Keyboard Acoustic (MKA) Datasets/MKA datasets")
    args = parser.parse_args()

    seeds      = [int(s) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    models     = ["coatnet", "maxvit", "swin"] if args.model == "all" else [args.model]

    # Build loader_fn based on dataset type
    if args.dataset == "mka":
        run_label = "mka"
        loader_fn = lambda seed: get_mka_loaders(args.root, seed, args.batch_size)
    elif args.dataset == "both":
        from torch.utils.data import ConcatDataset
        run_label = "custom_both"
        def loader_fn(seed):
            e1 = CustomKeystrokeDataset(ENV_PATHS["E1"], max_clips=args.max_clips, augment=False)
            e2 = CustomKeystrokeDataset(ENV_PATHS["E2"], max_clips=args.max_clips, augment=False)
            combined = ConcatDataset([e1, e2])
            n = len(combined)
            n_train = int(n * 0.70); n_val = int(n * 0.15); n_test = n - n_train - n_val
            g = torch.Generator().manual_seed(seed)
            train_ds, val_ds, test_ds = random_split(combined, [n_train, n_val, n_test], generator=g)
            return (DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0),
                    DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0),
                    DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0))
    else:
        run_label = f"custom_{args.env}"
        seg_path  = ENV_PATHS[args.env]
        loader_fn = lambda seed: get_custom_loaders(seg_path, seed, args.batch_size, args.max_clips)

    device = get_device()
    print(f"[Device]  {device}")
    print(f"[Seeds]   {seeds}")
    print(f"[Dataset] {run_label}")
    print(f"[Split]   {'stratified (MKA built-in)' if args.dataset == 'mka' else 'random (balanced classes)'}")

    all_summaries = []
    for model_name in models:
        summary = run_multiseed(
            model_name=model_name,
            loader_fn=loader_fn,
            seeds=seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=output_dir,
            run_label=run_label,
        )
        all_summaries.append(summary)

    if len(all_summaries) > 1:
        print_summary_table(all_summaries)

    combined_path = output_dir / f"multiseed_summary_{run_label}.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[Saved] Combined → {combined_path}")
