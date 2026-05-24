"""
MKA Dataset Loader — Acoustic Keystroke Inference Thesis
=========================================================
Loads the Multi-Keyboard Acoustic (MKA) Dataset (Rawf et al., 2024)
from its Mendeley structure and prepares mel-spectrograms for
CoAtNet / MOAT / Swin Transformer training.

Dataset DOI: https://doi.org/10.17632/bpt2hvf8n3.3
Paper DOI  : https://doi.org/10.1016/j.dib.2024.110949

MKA folder structure on disk (verified May 2026):
    MKA datasets/
        hp/
            Sound Segment(wav)/       ← no space before bracket
                a1.wav ... a6.wav
                b1.wav ... b6.wav
                ...
        Lenovo/
            Sound Segment(wav)/
        MSI/   Mac/   Messenger/   Zoom/
        All Dataset/                  ← no trailing 's'
            Sound Segment(wav)/
                ahp1.wav ... ahp6.wav

Usage:
    python mka_dataloader.py --root "/Users/dominik/Downloads/Multi-Keyboard Acoustic (MKA) Datasets/MKA datasets" --platform hp --task alphanumeric
"""

import os
import re
import argparse
import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import librosa
import librosa.display
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ── Optional: install if missing ──────────────────────────────────────────────
# pip install librosa torch torchvision torchaudio

# ── Constants ─────────────────────────────────────────────────────────────────

# Folder names exactly as they appear on disk (verified against MKA download)
PLATFORMS = ["hp", "Lenovo", "MSI", "Mac", "Messenger", "Zoom"]

# 36 alphanumeric keys — matches Harrison et al. (2023) benchmark
ALPHANUMERIC_CLASSES = (
    [str(i) for i in range(10)] +                    # 0–9
    [chr(c) for c in range(ord('a'), ord('z') + 1)]  # a–z
)

# All ~73 keys in MKA (lowercase class names as stored on disk)
ALL_CLASSES = ALPHANUMERIC_CLASSES + [
    "apostrophe", "dash", "comma", "semicolon",
    "bracketopen", "bracketclose", "backtick", "equal",
    "altl", "altr", "asterisk", "backslash", "backspace", "caps",
    "cmdl", "start", "down", "end", "enter", "esc", "fn", "home",
    "lctrl", "left", "lshift", "menu", "fullstop", "pgdn", "pgup",
    "rctrl", "right", "rshift", "slash", "space", "tab", "up", "delete",
]

# Mel-spectrogram defaults — matching Harrison et al. (2023)
SR            = 44_100   # MKA native sampling rate
DURATION      = 1.0      # each segment is exactly 1 second
N_MELS        = 64
N_FFT         = 1024
HOP_LENGTH    = 225
IMAGE_SIZE    = 224       # 224x224 required for Swin/MaxViT window attention


# ── Helper: map raw filename stem → class label ───────────────────────────────

def stem_to_class(stem: str, platform: str) -> Optional[str]:
    """
    MKA naming convention:
      Per-platform :  'a1' … 'a6',  'space1' … 'space6'
      All-datasets :  'ahp1' … 'ahp6',  'spacehp1' …

    Returns lowercase class name or None if not recognised.
    """
    stem = stem.lower()
    # strip trailing digit (sample index 1–6)
    stem = re.sub(r'\d+$', '', stem)
    # for all-datasets folder, strip platform suffix
    if platform.lower() == "all":
        for p in [p.lower() for p in PLATFORMS]:
            if stem.endswith(p):
                stem = stem[: -len(p)]
                break
    return stem if stem else None


# ── Dataset class ─────────────────────────────────────────────────────────────

class MKADataset(Dataset):
    """
    PyTorch Dataset for the MKA Keyboard Acoustic corpus.

    Parameters
    ----------
    root        : str   Path to the 'MKA Datasets' top-level folder.
    platform    : str   One of PLATFORMS or 'all' (uses 'All Datasets').
    task        : str   'alphanumeric' (36 classes) or 'full' (~73 classes).
    transform   : callable  Optional torchvision transform applied to the
                            mel-spectrogram tensor (C×H×W, float32).
    augment     : bool  Apply SpecAugment-style time-shift + masking.
    """

    @staticmethod
    def _normalise_class(name: str) -> str:
        """Map MKA folder names to canonical lowercase class names."""
        mapping = {
            "lctrl": "lctrl", "rctrl": "rctrl",
            "lshift": "lshift", "rshift": "rshift",
            "altl": "altl",     "altr": "altr",
            "altleft": "altl",  "altright": "altr",
            "fulstop": "fullstop", "fullstop": "fullstop",
            "bracketopen": "bracketopen", "bracketclose": "bracketclose",
            "apostrophe": "apostrophe", "asterisk": "asterisk",
            "backslash": "backslash",   "backtick": "backtick",
            "backspace": "backspace",   "caps": "caps",
            "cmdl": "cmdl", "start": "start",
            "dash": "dash", "comma": "comma", "semicolon": "semicolon",
            "equal": "equal", "slash": "slash", "space": "space",
            "enter": "enter", "esc": "esc",     "fn": "fn",
            "tab": "tab",     "up": "up",       "down": "down",
            "left": "left",   "right": "right", "home": "home",
            "end": "end",     "pgup": "pgup",   "pgdn": "pgdn",
            "delete": "delete", "menu": "menu",
        }
        return mapping.get(name, name)

    def __init__(
        self,
        root: str,
        platform: str = "all",
        task: str = "alphanumeric",
        transform=None,
        augment: bool = False,
    ):
        self.root      = Path(root)
        self.platform  = platform
        self.augment   = augment
        self.transform = transform

        # Select class set
        if task == "alphanumeric":
            self.classes = ALPHANUMERIC_CLASSES
        else:
            self.classes = ALL_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Locate wav folder — names verified against MKA disk structure
        if platform.lower() == "all":
            wav_dir = self.root / "All Dataset" / "Sound Segment(wav)"
        else:
            wav_dir = self.root / platform / "Sound Segment(wav)"

        if not wav_dir.exists():
            raise FileNotFoundError(
                f"WAV folder not found: {wav_dir}\n"
                f"Check that --root points to 'MKA Datasets' and that "
                f"the sub-folder name matches exactly (mind spaces)."
            )

        # Collect (path, label_idx) pairs
        # MKA structure: Sound Segment(wav)/{key_name}/*.wav
        # Each key has its own subdirectory; folder name = class label
        self.samples: List[Tuple[Path, int]] = []
        skipped_dirs = 0
        skipped_files = 0

        for key_dir in sorted(wav_dir.iterdir()):
            if not key_dir.is_dir():
                continue
            cls_name = key_dir.name.lower()
            # Map common MKA folder name variants to canonical class names
            cls_name = MKADataset._normalise_class(cls_name)
            if cls_name not in self.class_to_idx:
                skipped_dirs += 1
                continue
            label_idx = self.class_to_idx[cls_name]
            wav_files = sorted(key_dir.glob("*.wav"))
            if not wav_files:
                wav_files = sorted(key_dir.glob("*.WAV"))
            for wav_path in wav_files:
                self.samples.append((wav_path, label_idx))
            if not wav_files:
                skipped_files += 1

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No WAV files found under {wav_dir}. "
                "Check the folder contents — expected one subdirectory per key."
            )

        if skipped_dirs:
            warnings.warn(
                f"{skipped_dirs} key folders skipped (not in '{task}' class set). "
                "Normal for 'alphanumeric' task — non-alphanumeric keys are excluded."
            )

        print(
            f"[MKADataset] Platform={platform} | Task={task} | "
            f"Samples={len(self.samples)} | Classes={len(self.classes)}"
        )

    # ── core audio → spectrogram ───────────────────────────────────────────

    def _load_wav(self, path: Path) -> np.ndarray:
        y, _ = librosa.load(str(path), sr=SR, duration=DURATION, mono=True)
        # Pad or trim to exactly DURATION * SR samples
        target_len = int(SR * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        return y

    def _time_shift(self, y: np.ndarray, shift_max: float = 0.4) -> np.ndarray:
        """Randomly shift signal by up to shift_max * length — SpecAugment."""
        shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
        return np.roll(y, shift)

    def _mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Normalize to [0, 1]
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return mel_db.astype(np.float32)

    def _spec_augment(self, mel: np.ndarray, mask_frac: float = 0.1) -> np.ndarray:
        """Time and frequency masking (SpecAugment, Park et al. 2019)."""
        mel = mel.copy()
        n_freq, n_time = mel.shape
        mean_val = mel.mean()

        # Frequency mask
        f = int(np.random.uniform(0, mask_frac) * n_freq)
        f0 = np.random.randint(0, max(1, n_freq - f))
        mel[f0: f0 + f, :] = mean_val

        # Time mask
        t = int(np.random.uniform(0, mask_frac) * n_time)
        t0 = np.random.randint(0, max(1, n_time - t))
        mel[:, t0: t0 + t] = mean_val

        return mel

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        wav_path, label = self.samples[idx]

        y   = self._load_wav(wav_path)
        if self.augment:
            y = self._time_shift(y)

        mel = self._mel_spectrogram(y)
        if self.augment:
            mel = self._spec_augment(mel)

        # Resize to IMAGE_SIZE × IMAGE_SIZE and add channel dim → (1, H, W)
        # torchvision resize expects PIL or (C,H,W) tensor
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)   # (1, n_mels, time)
        mel_tensor = torch.nn.functional.interpolate(
            mel_tensor.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                        # (1, 64, 64)

        # Expand to 3 channels for ImageNet-pretrained models
        mel_tensor = mel_tensor.repeat(3, 1, 1)            # (3, 64, 64)

        if self.transform:
            mel_tensor = self.transform(mel_tensor)

        return mel_tensor, label

    # ── Utility ───────────────────────────────────────────────────────────

    def class_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {c: 0 for c in self.classes}
        for _, idx in self.samples:
            counts[self.classes[idx]] += 1
        return counts

    def idx_to_class(self, idx: int) -> str:
        return self.classes[idx]


# ── Factory: train / val / test splits ───────────────────────────────────────

def build_dataloaders(
    root: str,
    platform: str = "all",
    task: str = "alphanumeric",
    train_frac: float = 0.70,
    val_frac: float   = 0.15,
    batch_size: int   = 16,
    num_workers: int  = 2,
    seed: int         = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, MKADataset]:
    """
    Returns (train_loader, val_loader, test_loader, full_dataset).
    Splits are random with a fixed seed for reproducibility.
    Augmentation is applied only to the training split.
    """
    # Build full dataset without augment first (for splitting)
    full_ds = MKADataset(root=root, platform=platform, task=task, augment=False)

    n_total = len(full_ds)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)
    n_test  = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )

    # Attach augmentation only to training subset
    train_ds.dataset = MKADataset(
        root=root, platform=platform, task=task, augment=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(
        f"[Splits] Train={len(train_ds)} | Val={len(val_ds)} | Test={len(test_ds)}"
    )
    return train_loader, val_loader, test_loader, full_ds


# ── Quick sanity check ────────────────────────────────────────────────────────

def sanity_check(root: str, platform: str = "HP", task: str = "alphanumeric"):
    print("\n=== MKA Dataset Sanity Check ===")
    ds = MKADataset(root=root, platform=platform, task=task)

    print(f"Total samples : {len(ds)}")
    print(f"Num classes   : {len(ds.classes)}")
    print(f"Classes       : {ds.classes}")

    dist = ds.class_distribution()
    print("\nSamples per class:")
    for cls, cnt in dist.items():
        print(f"  {cls:15s}: {cnt}")

    # Load one sample
    tensor, label = ds[0]
    print(f"\nSample tensor shape : {tensor.shape}")   # expect (3, 64, 64)
    print(f"Label               : {label} → '{ds.idx_to_class(label)}'")
    print("=================================\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MKA Dataset Loader")
    parser.add_argument(
        "--root", type=str, required=True,
        help="Path to the 'MKA Datasets' top-level folder"
    )
    parser.add_argument(
        "--platform", type=str, default="hp",
        choices=PLATFORMS + ["all"],
        help="Platform to load (default: hp)"
    )
    parser.add_argument(
        "--task", type=str, default="alphanumeric",
        choices=["alphanumeric", "full"],
        help="Class set: 36 alphanumeric or full ~73 keys"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16
    )
    args = parser.parse_args()

    sanity_check(args.root, args.platform, args.task)

    train_dl, val_dl, test_dl, dataset = build_dataloaders(
        root=args.root,
        platform=args.platform,
        task=args.task,
        batch_size=args.batch_size,
    )

    # Peek at one batch
    images, labels = next(iter(train_dl))
    print(f"Batch images shape : {images.shape}")   # (B, 3, 64, 64)
    print(f"Batch labels shape : {labels.shape}")   # (B,)
    print(f"Label sample       : {[dataset.idx_to_class(l.item()) for l in labels[:6]]}")
