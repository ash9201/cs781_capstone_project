"""Export the selected probability blend and validate its submission format.

Run after ensemble.py. The selection file stores the fixed final rule.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from ensemble import dump, metrics


def export(results_dir, selection_file, output_dir):
    results_dir, output_dir = Path(results_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads(Path(selection_file).read_text())
    with np.load(results_dir / "aligned_probabilities.npz", allow_pickle=False) as saved:
        a = {k: saved[k] for k in saved.files}
    model_order = a["model_order"].tolist()
    weights = np.array([selection["weights"].get(k, 0) for k in model_order], dtype=np.float64)
    assert np.all(weights >= 0) and abs(weights.sum() - 1) < 1e-12
    assert set(selection["weights"]) <= set(model_order)
    threshold = float(selection["threshold"])
    assert 0 <= threshold <= 1
    name = selection["export_name"]
    assert name.replace("_", "").isalnum()
    dev_p = a["dev"] @ weights
    test_p = a["test"] @ weights
    dev_row_p = a["dev_row_probabilities"] @ weights
    dev_metrics = metrics(a["gold"], dev_p >= threshold, dev_p)
    repeated_metrics = metrics(a["dev_row_gold"], dev_row_p >= threshold, dev_row_p)
    assert abs(dev_metrics["macro_f1"] - repeated_metrics["macro_f1"]) < 1e-12
    comparison = __import__("pandas").read_csv(results_dir / "ensemble_comparison.csv")
    expected = comparison[comparison.family == selection["family"]].iloc[0]
    assert abs(dev_metrics["macro_f1"] - expected.full_dev_macro_f1) < 1e-12
    assert abs(threshold - expected.full_dev_threshold) < 1e-12
    predictions = [{"id": str(case_id), "pred": int(p >= threshold)}
                   for case_id, p in zip(a["test_ids"], test_p)]
    assert len(predictions) == len({r["id"] for r in predictions}) == len(a["test_ids"])
    assert all(set(r) == {"id", "pred"} and type(r["pred"]) is int and r["pred"] in [0, 1] for r in predictions)
    payload = (json.dumps(predictions, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    zip_path = output_dir / (name + "_submission.zip")
    info = zipfile.ZipInfo("prediction.json", date_time=(2026, 10, 26, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(info, payload)
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["prediction.json"]
        assert z.testzip() is None
        assert json.loads(z.read("prediction.json")) == predictions
    dump(output_dir / "prediction.json", predictions)
    dump(results_dir / "ensemble_dev_prediction.json", [
        {"id": str(case_id), "pred": int(p >= threshold)} for case_id, p in zip(a["dev_row_uids"], dev_row_p)
    ])
    dump(results_dir / "ensemble_dev_truth.json", [
        {"id": str(case_id), "gold": int(y)} for case_id, y in zip(a["dev_row_uids"], a["dev_row_gold"])
    ])
    np.savez_compressed(results_dir / "ensemble_probabilities.npz", dev_ids=a["dev_ids"], gold=a["gold"],
                        dev_probability_1=dev_p, test_ids=a["test_ids"], test_probability_1=test_p)
    official = subprocess.run([
        sys.executable, str(results_dir / "evaluate_phase4.py"),
        "--truth", str(results_dir / "ensemble_dev_truth.json"),
        "--prediction", str(results_dir / "ensemble_dev_prediction.json"),
    ], check=True, capture_output=True, text=True)
    assert f"Macro-F1: {dev_metrics['macro_f1']:.6f}" in official.stdout
    (results_dir / "official_dev_evaluation.txt").write_text(official.stdout)
    audit = json.loads((results_dir / "input_audit.json").read_text())
    gold_threshold = next(x["export_threshold"] for x in audit["inputs"] if x["model"] == "efficient_gold")
    gold_pred = a["test"][:, model_order.index("efficient_gold")] >= gold_threshold
    summary = {"selection": selection, "dev": dev_metrics, "dev_repeated_rows": repeated_metrics,
               "test_cases": len(predictions), "test_predicted_0": int(sum(r["pred"] == 0 for r in predictions)),
               "test_predicted_1": int(sum(r["pred"] == 1 for r in predictions)),
               "test_changes_vs_original_gold": int(np.sum(gold_pred != (test_p >= threshold))),
               "submission_filename": zip_path.name,
               "submission_sha256": hashlib.sha256(zip_path.read_bytes()).hexdigest(),
               "prediction_json_sha256": hashlib.sha256(payload).hexdigest(),
               "official_dev_evaluator": "passed", "zip_root_and_prediction_schema": "passed"}
    dump(results_dir / "export_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--selection", type=Path, default=Path("selection.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("deliverables"))
    args = parser.parse_args()
    export(args.results_dir, args.selection, args.output_dir)
