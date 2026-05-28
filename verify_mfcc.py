"""
MFCC Verification Script
=========================
Addresses reviewer concern (a) and (7):
Verifies that MacBook and Dell datasets are loaded independently
and that identical accuracy is a genuine finding, not a bug.

Run:
    python verify_mfcc.py
"""

import sys, json
import numpy as np
from pathlib import Path
sys.path.append('.')

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import librosa

SR = 44100; N_MFCC = 40; DURATION = 1.0
ALPHANUMERIC = [str(i) for i in range(10)] + \
               [chr(c) for c in range(ord('a'), ord('z')+1)]

DATASETS = {
    'MacBook_E1': Path.home() / 'Downloads/CustomDataset/E1_clean/segmented',
    'MacBook_E2': Path.home() / 'Downloads/CustomDataset/E2_window/segmented',
    'Dell_E1':    Path.home() / 'Downloads/CustomDataset/Dell_E1_clean/segmented',
    'Dell_E2':    Path.home() / 'Downloads/CustomDataset/Dell_E2_window/segmented',
}

SEEDS = [42, 123, 456]

def extract_mfcc(path):
    y, _ = librosa.load(str(path), sr=SR, duration=DURATION, mono=True)
    if len(y) < int(SR * 0.1): return None
    y = y / (np.max(np.abs(y)) + 1e-8)
    mfcc   = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    mfcc_d = librosa.feature.delta(mfcc)
    mfcc_d2= librosa.feature.delta(mfcc, order=2)
    return np.concatenate([mfcc.mean(1), mfcc.std(1),
                           mfcc_d.mean(1), mfcc_d2.mean(1)]).astype(np.float32)

def load_dataset(seg_path, max_clips=50):
    X, y = [], []
    c2i = {c: i for i, c in enumerate(ALPHANUMERIC)}
    for kd in sorted(seg_path.iterdir()):
        if not kd.is_dir(): continue
        cls = kd.name.lower()
        if cls not in c2i: continue
        for wav in sorted(kd.glob('*.wav'))[:max_clips]:
            feat = extract_mfcc(wav)
            if feat is not None:
                X.append(feat); y.append(c2i[cls])
    return np.array(X), np.array(y)

results = {}
for ds_name, seg_path in DATASETS.items():
    if not seg_path.exists():
        print(f'SKIP {ds_name}: path not found')
        continue
    print(f'\nLoading {ds_name} from {seg_path}...')
    X, y = load_dataset(seg_path)
    print(f'  {len(X)} samples, first file hash: {hash(X[0].tobytes()) % 99999}')

    seed_results = {}
    for seed in SEEDS:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.15, random_state=seed, stratify=y)

        svm_pipe = Pipeline([('sc', StandardScaler()),
                             ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                                        random_state=seed))])
        svm_pipe.fit(X_tr, y_tr)
        svm_preds = svm_pipe.predict(X_te)
        svm_acc   = accuracy_score(y_te, svm_preds)

        rf_pipe = Pipeline([('sc', StandardScaler()),
                            ('rf', RandomForestClassifier(n_estimators=200,
                                                          random_state=seed, n_jobs=-1))])
        rf_pipe.fit(X_tr, y_tr)
        rf_preds = rf_pipe.predict(X_te)
        rf_acc   = accuracy_score(y_te, rf_preds)

        seed_results[seed] = {
            'svm_acc': round(svm_acc*100, 2),
            'rf_acc':  round(rf_acc*100, 2),
            'svm_preds': svm_preds.tolist(),
            'rf_preds':  rf_preds.tolist(),
            'y_test':    y_te.tolist()
        }
        print(f'  Seed {seed}: SVM={svm_acc*100:.2f}%  RF={rf_acc*100:.2f}%')

    results[ds_name] = seed_results

# Cross-keyboard prediction comparison
print('\n=== CROSS-KEYBOARD PREDICTION COMPARISON (Seed 42) ===')
pairs = [('MacBook_E1', 'Dell_E1'), ('MacBook_E2', 'Dell_E2')]
for a, b in pairs:
    if a not in results or b not in results: continue
    preds_a = results[a][42]['svm_preds']
    preds_b = results[b][42]['svm_preds']
    n_same  = sum(pa == pb for pa, pb in zip(preds_a, preds_b))
    print(f'  {a} vs {b}: {n_same}/{len(preds_a)} identical SVM predictions '
          f'({n_same/len(preds_a)*100:.1f}%)')

# Summary table
print('\n=== ACCURACY SUMMARY ACROSS SEEDS ===')
print(f'  {"Dataset":<15} {"SVM_42":>8} {"SVM_123":>8} {"SVM_456":>8} {"SVM_mean±std":>15}'
      f' {"RF_42":>8} {"RF_123":>8} {"RF_456":>8} {"RF_mean±std":>15}')
for ds_name, sr in results.items():
    svms = [sr[s]['svm_acc'] for s in SEEDS if s in sr]
    rfs  = [sr[s]['rf_acc']  for s in SEEDS if s in sr]
    print(f'  {ds_name:<15} {svms[0]:>8.2f} {svms[1]:>8.2f} {svms[2]:>8.2f} '
          f'{np.mean(svms):>6.2f}±{np.std(svms):>4.2f}    '
          f'{rfs[0]:>8.2f} {rfs[1]:>8.2f} {rfs[2]:>8.2f} '
          f'{np.mean(rfs):>6.2f}±{np.std(rfs):>4.2f}')

with open('results/baseline/mfcc_verification.json', 'w') as f:
    # Save without actual predictions (too large)
    save = {k: {s: {m: v for m, v in sv.items() if 'preds' not in m}
                for s, sv in sr.items()} for k, sr in results.items()}
    json.dump(save, f, indent=2)
print('\nSaved: results/baseline/mfcc_verification.json')
