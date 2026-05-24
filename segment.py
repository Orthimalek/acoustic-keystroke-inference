"""
Acoustic Keystroke Segmentation Script
=======================================
Thesis: Acoustic Keystroke Inference: A Comparative Study of Hybrid
        Neural Network Architectures for Side Channel Attacks
Author: Vinit Rane | CSUDH CYB Program

Splits raw multi-press WAV recordings into individual keystroke clips
using librosa onset detection with amplitude normalisation.

Input  : ~/Downloads/CustomDataset/E1_clean/wav/*.wav
         Each file contains ~50 keypresses for one key
Output : ~/Downloads/CustomDataset/E1_clean/segmented/{key}/
         Each subfolder contains individual 0.5s WAV clips

Usage:
    python segment.py --input ~/Downloads/CustomDataset/E1_clean/wav \
                      --output ~/Downloads/CustomDataset/E1_clean/segmented \
                      --env E1
"""

import os
import re
import argparse
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

# ── Constants ─────────────────────────────────────────────────────────────────

SR           = 44_100
CLIP_DURATION = 0.5        # seconds per individual keystroke clip
CLIP_SAMPLES  = int(SR * CLIP_DURATION)
MIN_PRESSES   = 30         # warn if fewer detected
TARGET_PRESSES = 50

# ── Onset detection ───────────────────────────────────────────────────────────

def detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect keystroke onset times using librosa onset detection.
    Uses RMS energy envelope with peak picking.
    Returns array of onset times in seconds.
    """
    # Normalise audio to [-1, 1] to handle low amplitude recordings
    max_amp = np.max(np.abs(y))
    if max_amp > 0:
        y_norm = y / max_amp
    else:
        y_norm = y

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y_norm, sr=sr,
        hop_length=512,
        aggregate=np.median,
    )

    # Peak picking — wait=10 prevents double-detection within ~0.1s
    onset_frames = librosa.util.peak_pick(
        onset_env,
        pre_max=10, post_max=10,
        pre_avg=20, post_avg=20,
        delta=0.3,
        wait=20,
    )

    onset_times = librosa.frames_to_time(
        onset_frames, sr=sr, hop_length=512
    )

    return onset_times


def extract_clips(y: np.ndarray, sr: int, onset_times: np.ndarray):
    """
    Extract fixed-length clips centred around each onset.
    Returns list of numpy arrays, one per keystroke.
    """
    clips = []
    half = CLIP_SAMPLES // 2

    for t in onset_times:
        center = int(t * sr)
        start  = max(0, center - half // 4)   # slight pre-onset context
        end    = start + CLIP_SAMPLES

        if end > len(y):
            # Pad if clip extends beyond recording
            clip = y[start:]
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        else:
            clip = y[start:end]

        # Per-clip normalisation
        max_amp = np.max(np.abs(clip))
        if max_amp > 0:
            clip = clip / max_amp * 0.9

        clips.append(clip.astype(np.float32))

    return clips


# ── Main segmentation ─────────────────────────────────────────────────────────

def segment_file(
    wav_path: Path,
    output_dir: Path,
    env: str,
) -> int:
    """
    Segments one raw recording into individual keystroke clips.
    Returns number of clips saved.
    """
    # Parse key name from filename e.g. a_E1_01.wav → 'a'
    stem = wav_path.stem                        # e.g. 'a_E1_01'
    key  = stem.split('_')[0]                   # e.g. 'a'

    # Load audio
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)

    # Detect onsets
    onset_times = detect_onsets(y, sr)

    if len(onset_times) < MIN_PRESSES:
        warnings.warn(
            f"[{wav_path.name}] Only {len(onset_times)} onsets detected "
            f"(expected ~{TARGET_PRESSES}). Check recording quality."
        )

    # Extract clips
    clips = extract_clips(y, sr, onset_times)

    # Save clips
    key_dir = output_dir / key
    key_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i, clip in enumerate(clips, start=1):
        out_path = key_dir / f"{key}_{env}_{i:03d}.wav"
        sf.write(str(out_path), clip, SR)
        saved += 1

    return saved


def run_segmentation(input_dir: Path, output_dir: Path, env: str):
    wav_files = sorted(input_dir.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {input_dir}")

    print(f"\n{'='*55}")
    print(f"  Segmentation — Environment: {env}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Files  : {len(wav_files)}")
    print(f"{'='*55}\n")

    total_clips = 0
    results = []

    for wav_path in wav_files:
        stem = wav_path.stem
        key  = stem.split('_')[0]

        n_clips = segment_file(wav_path, output_dir, env)
        total_clips += n_clips
        results.append((key, n_clips))
        print(f"  {wav_path.name:<25} → {n_clips:>3} clips  [{key}/]")

    print(f"\n{'='*55}")
    print(f"  Total clips saved : {total_clips}")
    print(f"  Output directory  : {output_dir}")
    print(f"{'='*55}\n")

    # Warn about keys with low clip counts
    low = [(k, n) for k, n in results if n < MIN_PRESSES]
    if low:
        print("  ⚠️  Low clip count warnings:")
        for k, n in low:
            print(f"     Key '{k}': only {n} clips detected")
        print("  Consider re-recording these keys.\n")
    else:
        print("  ✅ All keys have sufficient clips.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment raw keystroke recordings into individual clips"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Folder containing raw WAV files (one per key)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output folder for segmented clips"
    )
    parser.add_argument(
        "--env", type=str, default="E1",
        choices=["E1", "E2", "E3", "E4"],
        help="Environment label (E1=clean, E2=window, E3=washing, E4=dishwasher)"
    )
    args = parser.parse_args()

    run_segmentation(
        input_dir  = Path(args.input),
        output_dir = Path(args.output),
        env        = args.env,
    )
