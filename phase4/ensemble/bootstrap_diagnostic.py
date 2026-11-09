"""Conditional uncertainty check for stored OOF predictions, grouped by case.

This does not refit weights, thresholds, or checkpoints within a bootstrap sample.
Run from the project directory after ensemble.py.
"""

from pathlib import Path
import json

import numpy as np


def macro_f1(y, pred):
    tp = np.sum(pred & (y == 1), axis=-1)
    fp = np.sum(pred & (y == 0), axis=-1)
    fn = np.sum((~pred) & (y == 1), axis=-1)
    tn = np.sum((~pred) & (y == 0), axis=-1)
    return tp / (2*tp + fp + fn) + tn / (2*tn + fp + fn)


def run(results_dir="results"):
    r = Path(results_dir)
    a = np.load(r / "oof_predictions.npz", allow_pickle=False)
    y = a["gold"]
    rng = np.random.default_rng(7812026)
    results = {}
    for family in ["gold_plus_prefix", "select_any_candidate"]:
        base = a["efficient_gold"].astype(bool)
        blend = a[family].astype(bool)
        deltas = []
        for _ in range(25):
            ix = rng.integers(0, len(y), size=(200, len(y)))
            d = (macro_f1(y[ix][None, :, :], blend[:, ix])
                 - macro_f1(y[ix][None, :, :], base[:, ix])).mean(axis=0)
            deltas.extend(d.tolist())
        point = (macro_f1(y[None, :], blend) - macro_f1(y[None, :], base)).mean()
        results[family] = {"delta_macro_f1_mean": float(point),
                           "paired_bootstrap_95_percentile_interval": np.quantile(deltas, [.025, .975]).tolist(),
                           "resamples": 5000}
    results["method"] = (
        "Paired bootstrap of 994 original dev case IDs. Each resample retains predictions from all five repeats. "
        "The statistic is mean OOF macro-F1 difference across repeats. Fitted fold rules and base checkpoints are "
        "held fixed, so this is a conditional diagnostic, not a full pipeline confidence interval."
    )
    results["seed"] = 7812026
    (r / "bootstrap_diagnostic.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.results_dir)
