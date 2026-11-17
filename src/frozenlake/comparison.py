"""A matched initialization control must actually isolate initialization."""
import csv
import json
from pathlib import Path
import numpy as np


def assert_matched(base, candidate):
    for key in base:
        if key not in ["name", "variant", "initialization"] and base[key] != candidate[key]:
            raise ValueError(f"Unmatched protocol field: {key}")
    if base["initialization"]["method"] != "zero" or candidate["initialization"]["method"] != "optimistic":
        raise ValueError("Expected zero baseline and optimistic candidate")


def compare(baseline, candidate, output):
    baseline, candidate, output = map(Path, [baseline, candidate, output])
    cfg = json.loads((baseline / "config.json").read_text())
    alt = json.loads((candidate / "config.json").read_text())
    assert_matched(cfg, alt)
    for root in [baseline, candidate]:
        if json.loads((root / "verification.json").read_text())["status"] != "passed":
            raise ValueError("Both experiments must be verified")
    a, b = np.load(baseline / "inputs.npz"), np.load(candidate / "inputs.npz")
    for key in a.files:
        np.testing.assert_array_equal(a[key], b[key])
    ma, mb = [json.loads((p / "manifest.json").read_text()) for p in [baseline, candidate]]
    for field in ["backend", "jax_version", "numpy_version", "q_dtype"]:
        if ma[field] != mb[field]:
            raise ValueError(f"Unmatched runtime field: {field}")
    for source in ["learner.py", "initialization.py"]:
        if ma["source_sha256"][source] != mb["source_sha256"][source]:
            raise ValueError(f"Training code changed: {source}; retrain a matched baseline")
    def records(path):
        with (path / "summary.csv").open() as handle:
            return {(r["group"], r["map_seed"], r["training_seed"]): r for r in csv.DictReader(handle)}
    left, right = records(baseline), records(candidate)
    if left.keys() != right.keys():
        raise ValueError("Unmatched map/training seeds")
    pairs = []
    for key in sorted(left):
        x, y = float(left[key]["success_probability"]), float(right[key]["success_probability"])
        pairs.append(dict(group=key[0], map_seed=key[1], training_seed=key[2],
                          baseline_success=x, candidate_success=y, difference=y - x))
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    selected = [r for r in pairs if r["group"] == "main"]
    result = dict(pairs=len(pairs), main_pairs=len(selected),
                  baseline_mean=float(np.mean([r["baseline_success"] for r in selected])),
                  candidate_mean=float(np.mean([r["candidate_success"] for r in selected])),
                  mean_difference=float(np.mean([r["difference"] for r in selected])),
                  note="Matched descriptive results, not a universal performance guarantee")
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
