"""
print_results.py
Reads all multiseed JSON files from results/multiseed/ and prints a
paper-ready table, plus the t-test numbers needed for the IEEE Access revision.

Usage:
    python print_results.py
    python print_results.py --results_dir ./results/multiseed
"""

import json
import argparse
from pathlib import Path
from scipy import stats
import numpy as np

def load_results(results_dir: Path):
    rows = []
    for f in sorted(results_dir.glob("*_multiseed.json")):
        data = json.loads(f.read_text())
        rows.append(data)
    return rows

def fmt(mean, std):
    return f"{mean*100:.2f}% ± {std*100:.2f}%"

def print_table(rows):
    print("\n" + "="*80)
    print(f"  {'Model':<12} {'Dataset':<18} {'Val Mean±Std':>16} {'Test Mean±Std':>16} {'Range':>16}")
    print("-"*80)
    for r in rows:
        label = r.get("run_label", "?")
        print(
            f"  {r['model']:<12} {label:<18} "
            f"{fmt(r['val_mean'], r['val_std']):>16}  "
            f"{fmt(r['test_mean'], r['test_std']):>16}  "
            f"[{r['test_min']*100:.2f}%, {r['test_max']*100:.2f}%]"
        )
    print("="*80 + "\n")

def run_ttests(rows):
    """
    Print paired and independent t-tests where the paper needs them.
    """
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  T-TEST SUMMARY  (for paper update)")
    print("──────────────────────────────────────────────────────────────────────")

    # Group by dataset
    by_label = {}
    for r in rows:
        by_label.setdefault(r["run_label"], {})[r["model"]] = r

    def accs(r):
        return [x["test_acc"] for x in r["per_seed"]]

    # Custom E1: CoAtNet vs MaxViT (paired, same splits)
    if "custom_E1" in by_label:
        bl = by_label["custom_E1"]
        if "coatnet" in bl and "maxvit" in bl:
            a, b = accs(bl["coatnet"]), accs(bl["maxvit"])
            t, p = stats.ttest_rel(a, b)
            d = (np.mean(a) - np.mean(b)) / np.std([ai - bi for ai, bi in zip(a, b)], ddof=1)
            print(f"\n  Custom E1  CoAtNet vs MaxViT (paired t-test, n={len(a)})")
            print(f"    CoAtNet:  {np.mean(a)*100:.2f}% ± {np.std(a, ddof=1)*100:.2f}%")
            print(f"    MaxViT:   {np.mean(b)*100:.2f}% ± {np.std(b, ddof=1)*100:.2f}%")
            print(f"    t={t:.3f}, p={p:.3f}, Cohen's d={d:.3f}")

    # MKA: CoAtNet vs MaxViT (independent — different seeds may give different splits)
    if "mka" in by_label:
        bl = by_label["mka"]
        if "coatnet" in bl and "maxvit" in bl:
            a, b = accs(bl["coatnet"]), accs(bl["maxvit"])
            t, p = stats.ttest_ind(a, b)
            pooled_sd = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            d = (np.mean(a) - np.mean(b)) / pooled_sd
            ci_lo = (np.mean(a) - np.mean(b) - 1.96 * np.std(a + b, ddof=1) / np.sqrt(len(a))) * 100
            ci_hi = (np.mean(a) - np.mean(b) + 1.96 * np.std(a + b, ddof=1) / np.sqrt(len(a))) * 100
            print(f"\n  MKA  CoAtNet vs MaxViT (independent t-test, n={len(a)} each)")
            print(f"    CoAtNet:  {np.mean(a)*100:.2f}% ± {np.std(a, ddof=1)*100:.2f}%")
            print(f"    MaxViT:   {np.mean(b)*100:.2f}% ± {np.std(b, ddof=1)*100:.2f}%")
            print(f"    t={t:.3f}, p={p:.3f}, Cohen's d={d:.3f}")
            print(f"    95% CI (diff): [{ci_lo:.2f} pp, {ci_hi:.2f} pp]")

    # Custom E2 and E1+E2 ttests if both models available
    for lbl in ["custom_E2", "custom_both"]:
        if lbl in by_label:
            bl = by_label[lbl]
            if "coatnet" in bl and "maxvit" in bl:
                a, b = accs(bl["coatnet"]), accs(bl["maxvit"])
                t, p = stats.ttest_rel(a, b)
                d = (np.mean(a) - np.mean(b)) / np.std([ai - bi for ai, bi in zip(a, b)], ddof=1)
                print(f"\n  {lbl}  CoAtNet vs MaxViT (paired t-test, n={len(a)})")
                print(f"    CoAtNet:  {np.mean(a)*100:.2f}% ± {np.std(a, ddof=1)*100:.2f}%")
                print(f"    MaxViT:   {np.mean(b)*100:.2f}% ± {np.std(b, ddof=1)*100:.2f}%")
                print(f"    t={t:.3f}, p={p:.3f}, Cohen's d={d:.3f}")

    # Swin vs best model on each dataset
    for lbl in ["custom_E1", "custom_E2", "custom_both", "mka"]:
        if lbl in by_label:
            bl = by_label[lbl]
            if "swin" in bl:
                best_model = max(
                    [m for m in bl if m != "swin"],
                    key=lambda m: np.mean(accs(bl[m])),
                    default=None
                )
                if best_model:
                    a, b = accs(bl[best_model]), accs(bl["swin"])
                    t, p = stats.ttest_rel(a, b)
                    d = (np.mean(a) - np.mean(b)) / np.std([ai - bi for ai, bi in zip(a, b)], ddof=1)
                    print(f"\n  {lbl}  {best_model.upper()} vs Swin (paired t-test, n={len(a)})")
                    print(f"    {best_model.upper()}: {np.mean(a)*100:.2f}% ± {np.std(a, ddof=1)*100:.2f}%")
                    print(f"    Swin:     {np.mean(b)*100:.2f}% ± {np.std(b, ddof=1)*100:.2f}%")
                    print(f"    t={t:.3f}, p={p:.3f}, Cohen's d={d:.3f}")

    print("\n──────────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="./results/multiseed")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run ./run_missing_experiments.sh first.")
        exit(1)

    rows = load_results(results_dir)
    if not rows:
        print("No *_multiseed.json files found yet.")
        exit(0)

    print_table(rows)
    run_ttests(rows)
