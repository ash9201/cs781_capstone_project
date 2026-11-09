"""Evaluate CJPE probability ensembles using existing diagnostics, without a GPU.

Usage: python ensemble.py --input-dir inputs --output-dir results
No leaderboard scores or test labels are used to select parameters.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


MODELS = ["efficient_gold", "efficient_kd", "prefix", "decoder", "encoder_decoder"]
RUNS = [
    "cjpe_efficient_gold_001", "cjpe_efficient_kd_001", "cjpe_prefix_lorma_001",
    "cjpe_decoder_partial_003", "cjpe_encoder_decoder_adapters_001",
]
THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)
SEEDS = [781, 782, 783, 784, 785]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def metrics(y, pred, probability=None):
    out = {
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_1": float(precision_score(y, pred, zero_division=0)),
        "recall_1": float(recall_score(y, pred, zero_division=0)),
        "predicted_positive_fraction": float(np.mean(pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }
    if probability is not None:
        out["roc_auc"] = float(roc_auc_score(y, probability))
    return out


def load_archives(folder):
    found = {}
    for path in sorted(Path(folder).glob("*.zip")):
        with zipfile.ZipFile(path) as z:
            for run in RUNS:
                root = run + "/"
                if root + "dev_probabilities.npz" not in z.namelist():
                    continue
                if run in found:
                    raise ValueError(f"Multiple diagnostic archives for {run}")
                entry = {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for split in ["dev", "test"]:
                    with np.load(io.BytesIO(z.read(root + split + "_probabilities.npz")), allow_pickle=False) as a:
                        entry[split] = {k: a[k] for k in a.files}
                    a = entry[split]
                    n = len(a["original_ids"])
                    assert a["probability_1"].shape == (n,)
                    assert a["logits"].shape == (n, 2)
                    assert len(set(a["row_uids"])) == n
                    assert np.isfinite(a["probability_1"]).all() and np.isfinite(a["logits"]).all()
                    assert ((a["probability_1"] >= 0) & (a["probability_1"] <= 1)).all()
                    delta = a["logits"][:, 1].astype(float) - a["logits"][:, 0].astype(float)
                    recovered = np.exp(-np.logaddexp(0, -delta))
                    assert np.allclose(recovered, a["probability_1"], atol=2e-7, rtol=2e-6)
                for name in ["dev_truth", "dev_prediction", "prediction", "selected_results", "inference_settings", "input_record"]:
                    entry[name] = json.loads(z.read(root + name + ".json"))
                entry["evaluator"] = z.read(root + "sources/evaluate_phase4.py")
                found[run] = entry
    assert set(found) == set(RUNS), f"Missing archives: {set(RUNS) - set(found)}"
    reference = found[RUNS[0]]
    dev_rows = reference["dev"]["row_uids"].tolist()
    dev_ids = reference["dev"]["original_ids"].tolist()
    test_ids = reference["test"]["original_ids"].tolist()
    assert len(set(test_ids)) == len(test_ids)
    truth = {r["id"]: r["gold"] for r in reference["dev_truth"]}
    assert len(truth) == len(reference["dev_truth"]) == len(dev_rows)
    assert set(truth) == set(dev_rows)
    assert all(type(x) is int and x in [0, 1] for x in truth.values())
    y = np.array([truth[k] for k in dev_rows], dtype=np.int64)
    seen = {}
    for i, case_id in enumerate(dev_ids):
        if case_id in seen:
            assert y[i] == y[seen[case_id]], "Conflicting labels for the same dev case"
        else:
            seen[case_id] = i
    first = np.array(list(seen.values()))
    assert len(set(Counter(dev_ids).values())) == 1, "Unequal repeat weights need an explicit policy"
    matrices = {"dev": [], "test": []}
    audit = []
    for key, run in zip(MODELS, RUNS):
        r = found[run]
        identity = r["input_record"]["data_identity"]
        assert identity == reference["input_record"]["data_identity"]
        assert {x["id"]: x["gold"] for x in r["dev_truth"]} == truth
        row = {"model": key, "run_id": run, "source_filename": r["filename"], "source_sha256": r["sha256"]}
        for split, ids, id_key in [("dev", dev_rows, "row_uids"), ("test", test_ids, "original_ids")]:
            a = r[split]
            lookup = {v: i for i, v in enumerate(a[id_key].tolist())}
            assert len(lookup) == len(ids) and set(lookup) == set(ids)
            indices = np.array([lookup[v] for v in ids])
            a = {k: v[indices] for k, v in a.items()}
            assert a["original_ids"].tolist() == (dev_ids if split == "dev" else test_ids)
            r[split] = a
            p = a["probability_1"].astype(np.float64)
            if split == "dev":
                for i, case_id in enumerate(dev_ids):
                    assert p[i] == p[seen[case_id]], "Repeated case has inconsistent probabilities"
            threshold = r["selected_results"]["selected_threshold"]
            exported = r["dev_prediction" if split == "dev" else "prediction"]
            ep = {x["id"]: x["pred"] for x in exported}
            assert len(exported) == len(ep) == len(ids) and set(ep) == set(ids)
            assert all(type(x) is int and x in [0, 1] for x in ep.values())
            assert np.array_equal(p >= threshold, [ep[x] for x in ids])
            matrices[split].append(p)
        row["export_threshold"] = threshold
        row["export_dev_macro_f1"] = metrics(y, matrices["dev"][-1] >= threshold)["macro_f1"]
        expected = r["selected_results"]["dev_at_selected_threshold"]["macro_f1"]
        assert abs(row["export_dev_macro_f1"] - expected) < 1e-12
        row["dev_roc_auc"] = float(roc_auc_score(y, matrices["dev"][-1]))
        row["dev_macro_f1_at_0_5"] = metrics(y, matrices["dev"][-1] >= 0.5)["macro_f1"]
        row["selected_epoch"] = r["selected_results"].get("selected_epoch")
        row["selected_update"] = r["selected_results"].get("selected_update")
        audit.append(row)
    return {
        "P": np.column_stack(matrices["dev"])[first], "T": np.column_stack(matrices["test"]),
        "y": y[first], "dev_ids": np.array(dev_ids)[first], "test_ids": np.array(test_ids),
        "dev_row_uids": np.array(dev_rows), "dev_row_ids": np.array(dev_ids), "dev_row_y": y,
        "dev_row_P": np.column_stack(matrices["dev"]), "audit": audit,
        "data_identity": reference["input_record"]["data_identity"],
        "evaluator": reference["evaluator"],
    }


def make_candidates():
    candidates = []
    for i, key in enumerate(MODELS):
        w = np.zeros(len(MODELS)); w[i] = 1
        candidates.append({"name": key, "family": key, "weights": w.tolist()})
    for i, key in enumerate(MODELS[1:], 1):
        for weight in [0.75, 0.50, 0.25]:
            w = np.zeros(len(MODELS)); w[0] = weight; w[i] = 1 - weight
            candidates.append({"name": f"gold_{weight:.2f}__{key}_{1-weight:.2f}", "family": f"gold_plus_{key}", "weights": w.tolist()})
    for w in [[0.5, 0.25, 0.25, 0, 0], [0.25, 0.5, 0.25, 0, 0], [0.25, 0.25, 0.5, 0, 0]]:
        candidates.append({"name": f"gold_kd_prefix_{w[0]:.2f}_{w[1]:.2f}_{w[2]:.2f}", "family": "gold_kd_prefix", "weights": w})
    candidates.append({"name": "all_five_equal", "family": "all_five_equal", "weights": [0.2]*5})
    return candidates


def score_grid(probabilities, y):
    # Dimensions are candidate, threshold, case. Both binary classes count in macro-F1.
    pred = probabilities.T[:, None, :] >= THRESHOLDS[None, :, None]
    tp = np.sum(pred & (y[None, None, :] == 1), axis=2)
    fp = np.sum(pred & (y[None, None, :] == 0), axis=2)
    fn = np.sum(y) - tp
    tn = len(y) - np.sum(y) - fp
    return tp / (2*tp + fp + fn) + tn / (2*tn + fp + fn)


def choose(scores, members):
    # Candidate order prefers a single model, then a larger gold weight on exact ties.
    subset = scores[members]
    best = np.max(subset)
    ties = np.argwhere(subset >= best - 1e-12)
    ranked = sorted(ties.tolist(), key=lambda x: (x[0], abs(THRESHOLDS[x[1]] - 0.5), THRESHOLDS[x[1]]))
    c, t = ranked[0]
    return int(members[c]), int(t)


def run(input_dir, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    data = load_archives(input_dir)
    P, T, y = data["P"], data["T"], data["y"]
    candidates = make_candidates()
    W = np.array([x["weights"] for x in candidates], dtype=np.float64)
    Q = P @ W.T
    families = list(dict.fromkeys(x["family"] for x in candidates))
    members = {f: np.array([i for i, c in enumerate(candidates) if c["family"] == f]) for f in families}
    members["select_best_single"] = np.arange(len(MODELS))
    members["select_any_candidate"] = np.arange(len(candidates))
    predictions = {f: np.zeros((len(SEEDS), len(y)), dtype=np.int8) for f in members}
    folds, aggregate = [], []
    for repeat, seed in enumerate(SEEDS):
        for fold, (fit, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(P, y)):
            scores = score_grid(Q[fit], y[fit])
            for family, indices in members.items():
                c, t = choose(scores, indices)
                pred = Q[valid, c] >= THRESHOLDS[t]
                predictions[family][repeat, valid] = pred
                folds.append({"seed": seed, "fold": fold, "family": family,
                              "candidate": candidates[c]["name"], "threshold": float(THRESHOLDS[t]),
                              "fit_macro_f1": float(scores[c, t]),
                              "heldout_macro_f1": metrics(y[valid], pred)["macro_f1"]})
        for family in members:
            aggregate.append({"seed": seed, "family": family, **metrics(y, predictions[family][repeat])})
    pd.DataFrame(folds).to_csv(out / "fold_results.csv", index=False)
    pd.DataFrame(aggregate).drop(columns="confusion_matrix").to_csv(out / "oof_results_by_seed.csv", index=False)
    np.savez_compressed(out / "oof_predictions.npz", ids=data["dev_ids"], gold=y,
                        seeds=np.array(SEEDS), **predictions)
    full_scores = score_grid(Q, y)
    summaries = []
    for family, indices in members.items():
        c, t = choose(full_scores, indices)
        values = [r["macro_f1"] for r in aggregate if r["family"] == family]
        baseline = [r["macro_f1"] for r in aggregate if r["family"] == "efficient_gold"]
        summaries.append({"family": family, "oof_macro_f1_mean": float(np.mean(values)),
                          "oof_macro_f1_sd_across_seeds": float(np.std(values, ddof=1)),
                          "oof_delta_vs_gold_mean": float(np.mean(np.array(values) - baseline)),
                          "seeds_better_than_gold": int(np.sum(np.array(values) > baseline)),
                          "full_dev_macro_f1": float(full_scores[c, t]),
                          "full_dev_candidate": candidates[c]["name"],
                          "full_dev_threshold": float(THRESHOLDS[t]),
                          "candidate_index": c, "threshold_index": t})
    pd.DataFrame(summaries).to_csv(out / "ensemble_comparison.csv", index=False)
    pd.DataFrame(data["audit"]).to_csv(out / "individual_models.csv", index=False)
    pd.DataFrame(np.corrcoef(P.T), index=MODELS, columns=MODELS).to_csv(out / "probability_correlations.csv")
    export_pred = P >= np.array([r["export_threshold"] for r in data["audit"]])
    disagreement = np.array([[np.mean(export_pred[:, i] != export_pred[:, j]) for j in range(5)] for i in range(5)])
    pd.DataFrame(disagreement, index=MODELS, columns=MODELS).to_csv(out / "prediction_disagreement.csv")
    audit = {"status": "passed", "dev_rows": len(data["dev_row_ids"]),
             "unique_dev_cases": len(y), "dev_repeats_per_case": dict(Counter(Counter(data["dev_row_ids"]).values())),
             "unique_dev_class_counts": np.bincount(y, minlength=2).tolist(), "test_cases": len(T),
             "dev_id_label_and_probability_alignment": "passed", "test_id_alignment": "passed",
             "original_exports_reproduced": "passed", "probabilities_match_logits": "passed",
             "data_identity": data["data_identity"], "inputs": data["audit"],
             "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}}
    dump(out / "input_audit.json", audit)
    dump(out / "search_protocol.json", {"model_order": MODELS, "threshold_grid": THRESHOLDS.tolist(),
        "candidates": candidates, "folds": 5, "seeds": SEEDS,
        "grouping": "one row per original dev case after verifying equal repeats, labels and probabilities",
        "selection": "maximize fit-part macro-F1; exact ties prefer candidate order, then threshold nearest 0.5, then lower threshold",
        "limitation": "Base checkpoints were previously selected using the full dev set. These folds assess postprocessing only; they are not an independent evaluation of the full training procedure."})
    # Store aligned inputs so the final choice can be exported without repeating a search.
    np.savez_compressed(out / "aligned_probabilities.npz", dev_ids=data["dev_ids"], gold=y,
                        dev=P, test=T, test_ids=data["test_ids"], model_order=np.array(MODELS),
                        dev_row_uids=data["dev_row_uids"], dev_row_ids=data["dev_row_ids"],
                        dev_row_gold=data["dev_row_y"], dev_row_probabilities=data["dev_row_P"])
    (out / "evaluate_phase4.py").write_bytes(data["evaluator"])
    print(pd.DataFrame(summaries).drop(columns=["candidate_index", "threshold_index"]).to_string(index=False))
    print("\nProbability correlations\n", np.round(np.corrcoef(P.T), 3))
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("inputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)
