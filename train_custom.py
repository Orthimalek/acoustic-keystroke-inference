"""
Custom Dataset Training Script
================================
Thesis: Acoustic Keystroke Inference: A Comparative Study of Hybrid
        Neural Network Architectures for Side Channel Attacks
Author: Vinit Rane | CSUDH CYB Program

Trains CoAtNet, Swin, MaxViT on custom recorded dataset.

Usage:
    # Train on E1 clean
    python train_custom.py --env E1 --model coatnet --epochs 100

    # Train on E2 window
    python train_custom.py --env E2 --model maxvit --epochs 100

    # Train on E1+E2 combined
    python train_custom.py --env both --model all --epochs 100
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import timm
except ImportError:
    raise ImportError("Run: python -m pip install timm")

import sys
sys.path.append(str(Path(__file__).parent))
from custom_dataloader import build_custom_dataloaders

# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("[Device] Apple M1 MPS (GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("[Device] CPU")
        return torch.device("cpu")


# ── Model factory ─────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(model_name: str, num_classes: int = 36) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "coatnet":
        model = timm.create_model(
            "coatnet_0_rw_224", pretrained=True, num_classes=num_classes
        )
        print(f"[Model] CoAtNet-0 (pretrained) | Params: {count_params(model):,}")

    elif model_name == "swin":
        model = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=True, num_classes=num_classes
        )
        print(f"[Model] Swin-Tiny (pretrained) | Params: {count_params(model):,}")

    elif model_name == "maxvit":
        model = timm.create_model(
            "maxvit_tiny_tf_224.in1k", pretrained=True, num_classes=num_classes
        )
        print(f"[Model] MaxViT-Tiny (pretrained) | Params: {count_params(model):,}")

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: coatnet, swin, maxvit")

    return model


# ── Training loop ─────────────────────────────────────────────────────────────

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
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


# ── Full pipeline ─────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    env: str,
    max_clips: int,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
) -> Dict:
    device = get_device()

    train_dl, val_dl, test_dl, ref_ds = build_custom_dataloaders(
        env=env,
        max_clips=max_clips,
        batch_size=batch_size,
        num_workers=0,
        seed=42,
    )
    num_classes = len(ref_ds.classes)
    model = build_model(model_name, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "model": model_name, "env": env,
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "best_val_acc": 0.0, "best_epoch": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = output_dir / f"{model_name}_{env}_best.pt"

    print(f"\n{'='*60}")
    print(f"  Training  : {model_name.upper()}  |  Env: {env}")
    print(f"  Epochs    : {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Max clips : {max_clips}/class  |  Device: {device}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(
            model, train_dl, optimizer, criterion, device, scheduler
        )
        v_loss, v_acc = evaluate(model, val_dl, criterion, device)

        history["train_loss"].append(round(t_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_acc"].append(round(v_acc, 4))

        if v_acc > history["best_val_acc"]:
            history["best_val_acc"] = round(v_acc, 4)
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), best_ckpt)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(
                f"  Epoch {epoch:4d}/{epochs} | "
                f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f} | "
                f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f} | "
                f"Best Val: {history['best_val_acc']:.4f} (ep {history['best_epoch']}) | "
                f"Elapsed: {elapsed:.0f}s"
            )

    print(f"\n[Test] Loading best checkpoint (epoch {history['best_epoch']})...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc = evaluate(model, test_dl, criterion, device)
    history["test_loss"] = round(test_loss, 4)
    history["test_acc"]  = round(test_acc, 4)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  FINAL TEST ACCURACY : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  FINAL TEST LOSS     : {test_loss:.4f}")
    print(f"  BEST VAL ACCURACY   : {history['best_val_acc']:.4f} (epoch {history['best_epoch']})")
    print(f"  Total training time : {total_time:.0f}s")
    print(f"{'='*60}\n")

    history_path = output_dir / f"{model_name}_{env}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Saved] History -> {history_path}")
    print(f"[Saved] Checkpoint -> {best_ckpt}")

    return history


# ── Results table ─────────────────────────────────────────────────────────────

def print_comparison_table(results: List[Dict]):
    print("\n" + "="*65)
    print(f"  {'Model':<12} {'Env':<8} {'Val Acc':>9} {'Test Acc':>9} {'Best Ep':>8}")
    print("-"*65)
    for r in results:
        print(
            f"  {r['model']:<12} {r['env']:<8} "
            f"{r['best_val_acc']*100:>8.2f}%  "
            f"{r['test_acc']*100:>8.2f}%  "
            f"{r['best_epoch']:>7}"
        )
    print("="*65 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom Dataset Trainer")
    parser.add_argument(
        "--env", type=str, default="E1",
        choices=["E1", "E2", "both"],
        help="E1=clean | E2=window | both=combined"
    )
    parser.add_argument(
        "--model", type=str, default="coatnet",
        choices=["coatnet", "swin", "maxvit", "all"],
    )
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--max_clips",  type=int,   default=50,
                        help="Max clips per class (default 50 for balanced dataset)")
    parser.add_argument("--output_dir", type=str,   default="./results/custom")
    args = parser.parse_args()

    models_to_run = (
        ["coatnet", "swin", "maxvit"] if args.model == "all" else [args.model]
    )
    output_dir  = Path(args.output_dir)
    all_results = []

    for model_name in models_to_run:
        result = train_model(
            model_name=model_name,
            env=args.env,
            max_clips=args.max_clips,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            output_dir=output_dir,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print_comparison_table(all_results)

    summary_path = output_dir / f"summary_{args.env}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Saved] Summary -> {summary_path}")
