"""
Custom Dataset Training Script
================================
Thesis: Acoustic Keystroke Inference
Author: Shuchona Malek Orthi | Westcliff University

Trains CoAtNet, Swin, MaxViT on custom recorded dataset.
Supports pretrained and scratch (--no_pretrain) modes.

Usage:
    python train_custom.py --env E1 --model coatnet --epochs 100
    python train_custom.py --env E1 --model maxvit  --epochs 100 --no_pretrain
    python train_custom.py --env E1 --model swin    --epochs 100 --no_pretrain
    python train_custom.py --env E1 --model all     --epochs 100 --no_pretrain
"""

import argparse, time, json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import timm
except ImportError:
    raise ImportError("Run: pip install timm")

import sys
sys.path.append(str(Path(__file__).parent))
from custom_dataloader import build_custom_dataloaders

# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        print("[Device] Apple M1 MPS (GPU)")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print(f"[Device] CUDA — {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("[Device] CPU")
    return torch.device("cpu")

# ── Model factory ─────────────────────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model(model_name: str, num_classes: int = 36, pretrained: bool = True) -> nn.Module:
    model_name = model_name.lower()
    tag = "pretrained" if pretrained else "scratch"

    if model_name == "coatnet":
        model = timm.create_model("coatnet_0_rw_224", pretrained=pretrained, num_classes=num_classes)
        print(f"[Model] CoAtNet-0 ({tag}) | Params: {count_params(model):,}")
    elif model_name == "swin":
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=num_classes)
        print(f"[Model] Swin-Tiny ({tag}) | Params: {count_params(model):,}")
    elif model_name == "maxvit":
        model = timm.create_model("maxvit_tiny_tf_224.in1k", pretrained=pretrained, num_classes=num_classes)
        print(f"[Model] MaxViT-Tiny ({tag}) | Params: {count_params(model):,}")
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
    if scheduler: scheduler.step()
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

def train_model(model_name, env, max_clips, epochs, batch_size, lr,
                output_dir, pretrained=True) -> Dict:
    device = get_device()
    train_dl, val_dl, test_dl, ref_ds = build_custom_dataloaders(
        env=env, max_clips=max_clips, batch_size=batch_size, num_workers=0, seed=42)
    num_classes = len(ref_ds.classes)
    model = build_model(model_name, num_classes=num_classes, pretrained=pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    tag = "pretrained" if pretrained else "scratch"
    history = {
        "model": model_name, "env": env, "pretrained": pretrained,
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "best_val_acc": 0.0, "best_epoch": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_suffix = f"{model_name}_{env}_{tag}"
    best_ckpt = output_dir / f"{ckpt_suffix}_best.pt"

    print(f"\n{'='*60}")
    print(f"  Training  : {model_name.upper()} [{tag}]  |  Env: {env}")
    print(f"  Epochs    : {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Max clips : {max_clips}/class  |  Device: {device}")
    print(f"{'='*60}")

    start = time.time()
    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_dl, optimizer, criterion, device, scheduler)
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
            elapsed = time.time() - start
            print(f"  Epoch {epoch:4d}/{epochs} | "
                  f"Train: {t_acc:.4f} | Val: {v_acc:.4f} | "
                  f"Best Val: {history['best_val_acc']:.4f} (ep {history['best_epoch']}) | "
                  f"{elapsed:.0f}s")

    print(f"\n[Test] Loading best checkpoint (epoch {history['best_epoch']})...")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    test_loss, test_acc = evaluate(model, test_dl, criterion, device)
    history["test_loss"] = round(test_loss, 4)
    history["test_acc"]  = round(test_acc, 4)

    total_time = time.time() - start
    print(f"\n{'='*60}")
    print(f"  FINAL TEST ACCURACY : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  BEST VAL ACCURACY   : {history['best_val_acc']:.4f} (epoch {history['best_epoch']})")
    print(f"  Mode: {'ImageNet pretrained' if pretrained else 'SCRATCH — no pretrained weights'}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"{'='*60}\n")

    history_path = output_dir / f"{ckpt_suffix}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Saved] {history_path}")
    return history

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        type=str, default="E1", choices=["E1","E2","both"])
    parser.add_argument("--model",      type=str, default="coatnet", choices=["coatnet","swin","maxvit","all"])
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--max_clips",  type=int,   default=50)
    parser.add_argument("--output_dir", type=str,   default="./results/custom")
    parser.add_argument("--no_pretrain", action="store_true",
                        help="Train from scratch without ImageNet pretrained weights")
    args = parser.parse_args()

    pretrained   = not args.no_pretrain
    models       = ["coatnet","swin","maxvit"] if args.model == "all" else [args.model]
    output_dir   = Path(args.output_dir)
    all_results  = []

    tag = "pretrained" if pretrained else "scratch"
    print(f"[Mode]  {'ImageNet pretrained' if pretrained else 'SCRATCH — no pretrained weights'}")

    for model_name in models:
        r = train_model(model_name, args.env, args.max_clips,
                        args.epochs, args.batch_size, args.lr,
                        output_dir, pretrained=pretrained)
        all_results.append(r)

    if len(all_results) > 1:
        print("\n" + "="*65)
        print(f"  {'Model':<12} {'Env':<6} {'Mode':<12} {'Val Acc':>9} {'Test Acc':>9}")
        print("-"*65)
        for r in all_results:
            mode = "pretrained" if r["pretrained"] else "scratch"
            print(f"  {r['model']:<12} {r['env']:<6} {mode:<12} "
                  f"{r['best_val_acc']*100:>8.2f}%  {r['test_acc']*100:>8.2f}%")
        print("="*65)

    summary_path = output_dir / f"summary_{args.env}_{tag}.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] {summary_path}")
