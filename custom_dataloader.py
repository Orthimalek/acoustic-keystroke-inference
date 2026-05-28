"""
Custom Keystroke Dataset Loader
================================
Thesis: Acoustic Keystroke Inference: A Comparative Study of Hybrid
        Neural Network Architectures for Side Channel Attacks
Author: Shuchona Malek Orthi | Westcliff University

Loads segmented custom keystroke recordings from:
    ~/Downloads/CustomDataset/E1_clean/segmented/{key}/*.wav
    ~/Downloads/CustomDataset/E2_window/segmented/{key}/*.wav

Usage:
    python custom_dataloader.py --env E1 --max_clips 50
    python custom_dataloader.py --env both --max_clips 50
"""

import warnings
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset

# ── Constants ─────────────────────────────────────────────────────────────────

SR          = 44_100
CLIP_SAMPLES = int(SR * 0.5)   # 0.5 second clips
IMAGE_SIZE  = 224               # required for Swin/MaxViT/CoAtNet
N_MELS      = 64
N_FFT       = 1024
HOP_LENGTH  = 225

ALPHANUMERIC_CLASSES = (
    [str(i) for i in range(10)] +
    [chr(c) for c in range(ord('a'), ord('z') + 1)]
)

ENV_PATHS = {
    "E1": Path.home() / "Downloads/CustomDataset/E1_clean/segmented",
    "E2": Path.home() / "Downloads/CustomDataset/E2_window/segmented",
}


# ── Dataset ───────────────────────────────────────────────────────────────────

class CustomKeystrokeDataset(Dataset):
    """
    Loads individual segmented keystroke WAV clips.

    Parameters
    ----------
    env_path   : Path to segmented folder e.g. E1_clean/segmented/
    max_clips  : Max clips per class (default 50, caps dataset for balance)
    augment    : Apply SpecAugment during training
    """

    def __init__(
        self,
        env_path: Path,
        max_clips: int = 50,
        augment: bool = False,
        classes: Optional[List[str]] = None,
    ):
        self.env_path  = Path(env_path)
        self.max_clips = max_clips
        self.augment   = augment
        self.classes   = classes or ALPHANUMERIC_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[Path, int]] = []
        skipped = 0

        for key_dir in sorted(self.env_path.iterdir()):
            if not key_dir.is_dir():
                continue
            cls_name = key_dir.name.lower()
            if cls_name not in self.class_to_idx:
                skipped += 1
                continue

            label = self.class_to_idx[cls_name]
            wav_files = sorted(key_dir.glob("*.wav"))[:max_clips]
            for wav_path in wav_files:
                self.samples.append((wav_path, label))

        if skipped:
            warnings.warn(f"{skipped} key folders skipped (not alphanumeric).")

        print(
            f"[CustomDataset] Path={self.env_path.parent.name} | "
            f"Samples={len(self.samples)} | "
            f"Classes={len(self.classes)} | "
            f"Max clips/class={max_clips}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_clip(self, path: Path) -> np.ndarray:
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        # Pad or trim to CLIP_SAMPLES
        if len(y) < CLIP_SAMPLES:
            y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
        else:
            y = y[:CLIP_SAMPLES]
        # Normalise amplitude
        max_amp = np.max(np.abs(y))
        if max_amp > 0:
            y = y / max_amp
        return y.astype(np.float32)

    def _time_shift(self, y: np.ndarray, shift_max: float = 0.3) -> np.ndarray:
        shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
        return np.roll(y, shift)

    def _mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        return mel_db.astype(np.float32)

    def _spec_augment(self, mel: np.ndarray, mask_frac: float = 0.1) -> np.ndarray:
        mel = mel.copy()
        n_freq, n_time = mel.shape
        mean_val = mel.mean()
        f = int(np.random.uniform(0, mask_frac) * n_freq)
        f0 = np.random.randint(0, max(1, n_freq - f))
        mel[f0: f0 + f, :] = mean_val
        t = int(np.random.uniform(0, mask_frac) * n_time)
        t0 = np.random.randint(0, max(1, n_time - t))
        mel[:, t0: t0 + t] = mean_val
        return mel

    def __getitem__(self, idx: int):
        wav_path, label = self.samples[idx]
        y = self._load_clip(wav_path)

        if self.augment:
            y = self._time_shift(y)

        mel = self._mel_spectrogram(y)

        if self.augment:
            mel = self._spec_augment(mel)

        # Resize to IMAGE_SIZE × IMAGE_SIZE
        mel_tensor = torch.from_numpy(mel).unsqueeze(0)
        mel_tensor = torch.nn.functional.interpolate(
            mel_tensor.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        mel_tensor = mel_tensor.repeat(3, 1, 1)  # (3, 224, 224)

        return mel_tensor, label

    def class_distribution(self) -> Dict[str, int]:
        counts = {c: 0 for c in self.classes}
        for _, idx in self.samples:
            counts[self.classes[idx]] += 1
        return counts

    def idx_to_class(self, idx: int) -> str:
        return self.classes[idx]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_custom_dataloaders(
    env: str = "E1",
    max_clips: int = 50,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    batch_size: int = 16,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, CustomKeystrokeDataset]:
    """
    Returns (train_loader, val_loader, test_loader, full_dataset).

    env options:
      'E1'   — E1 clean only
      'E2'   — E2 window only
      'both' — E1 + E2 combined
    """
    generator = torch.Generator().manual_seed(seed)

    if env == "both":
        ds_e1 = CustomKeystrokeDataset(
            ENV_PATHS["E1"], max_clips=max_clips, augment=False
        )
        ds_e2 = CustomKeystrokeDataset(
            ENV_PATHS["E2"], max_clips=max_clips, augment=False
        )
        full_ds = ConcatDataset([ds_e1, ds_e2])
        # Use ds_e1 for class info
        ref_ds = ds_e1
    else:
        full_ds = CustomKeystrokeDataset(
            ENV_PATHS[env], max_clips=max_clips, augment=False
        )
        ref_ds = full_ds

    n_total = len(full_ds)
    n_train = int(n_total * train_frac)
    n_val   = int(n_total * val_frac)
    n_test  = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )

    # Attach augmentation to training split
    if env == "both":
        aug_e1 = CustomKeystrokeDataset(
            ENV_PATHS["E1"], max_clips=max_clips, augment=True
        )
        aug_e2 = CustomKeystrokeDataset(
            ENV_PATHS["E2"], max_clips=max_clips, augment=True
        )
        train_ds.dataset = ConcatDataset([aug_e1, aug_e2])
    else:
        train_ds.dataset = CustomKeystrokeDataset(
            ENV_PATHS[env], max_clips=max_clips, augment=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    print(
        f"[Splits] Train={len(train_ds)} | "
        f"Val={len(val_ds)} | Test={len(test_ds)}"
    )

    return train_loader, val_loader, test_loader, ref_ds


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="E1", choices=["E1", "E2", "both"])
    parser.add_argument("--max_clips", type=int, default=50)
    args = parser.parse_args()

    print(f"\n=== Custom Dataset Sanity Check — {args.env} ===")
    train_dl, val_dl, test_dl, ref_ds = build_custom_dataloaders(
        env=args.env, max_clips=args.max_clips
    )

    images, labels = next(iter(train_dl))
    print(f"Batch shape : {images.shape}")
    print(f"Label sample: {[ref_ds.idx_to_class(l.item()) for l in labels[:6]]}")
    print("=== Sanity check passed ===\n")
