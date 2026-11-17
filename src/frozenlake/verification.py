"""Audit the actual saved experiments before publishing their numbers."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from .configuration import alpha_at, epsilon_at, validate
from .evaluation import exact_policy


def read(path):
    return json.loads(Path(path).read_text())


def verify(output):
    output = Path(output)
    cfg, runs, manifest = [read(output / f) for f in ["config.json", "runs.json", "manifest.json"]]
    validate(cfg)
    assert manifest["status"] == "complete" and manifest["backend"] == "gpu"
    assert manifest["uid"] != 0 and all("cuda" in name for name in manifest["final_array_devices"])
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((output / "source" / name).read_bytes()).hexdigest() == digest, name
    prepared = read(output / "input_manifest.json")
    assert hashlib.sha256((output / "inputs.npz").read_bytes()).hexdigest() == prepared["input_sha256"]
    assert hashlib.sha256((output / "config.json").read_bytes()).hexdigest() == prepared["config_sha256"]
    inputs, snapshots, final = [np.load(output / f) for f in ["inputs.npz", "snapshots.npz", "final_state.npz"]]
    expected = sum(len(g["map_seeds"]) * len(g["training_seeds"]) for g in cfg["groups"].values())
    assert expected == len(runs) == manifest["learners"]
    assert len({(r["group"], r["map_seed"], r["training_seed"]) for r in runs}) == expected
    states, budget = cfg["environment"]["size"] ** 2, cfg["training"]["interactions"]
    assert snapshots["q"].shape == (len(snapshots["steps"]), expected, states, 4)
    np.testing.assert_array_equal(snapshots["visits"].sum(axis=(2, 3)), np.repeat(snapshots["steps"][:, None], expected, axis=1))
    assert np.isfinite(snapshots["q"]).all() and (snapshots["q"] >= 0).all() and (snapshots["q"] <= 1 + 1e-12).all()
    assert np.allclose(inputs["probability"].sum(axis=-1), 1)
    for i in range(expected):
        q0 = snapshots["q"][0, i]
        terminal = inputs["terminal_states"][i]
        assert (q0[~terminal] == cfg["initialization"]["value"]).all() and (q0[terminal] == 0).all()
        assert (snapshots["q"][:, i, terminal] == 0).all()
    goals = episodes = 0
    for i, run in enumerate(runs):
        directory = output / run["group"] / f'map_{run["map_seed"]}' / f'seed_{run["training_seed"]}'
        with gzip.open(directory / "episodes.csv.gz", "rt") as handle:
            log = np.atleast_1d(np.genfromtxt(handle, delimiter=",", names=True))
        assert len(log) == final["episodes"][i] == manifest["completed_episodes"][i]
        assert log["start_step"][0] == 0 and np.array_equal(log["start_step"][1:], log["end_step"][:-1])
        assert (log["length"] == log["end_step"] - log["start_step"]).all()
        assert int(log["length"].sum() + final["length"][i]) == budget
        assert (log["length"] <= cfg["environment"]["horizon"]).all()
        assert (log["reward"] == (log["outcome"] == 1)).all()
        assert int((log["outcome"] == 1).sum()) == final["goals"][i]
        np.testing.assert_allclose(log["discounted_return"], log["reward"] * cfg["training"]["gamma"] ** (log["length"] - 1), atol=1e-12)
        np.testing.assert_array_equal(log["learning_rate"], [alpha_at(cfg, int(step) - 1) for step in log["end_step"]])
        np.testing.assert_allclose(log["epsilon"], [epsilon_at(cfg, int(step) - 1) for step in log["end_step"]], atol=1e-14)
        q = snapshots["q"][-1, i]
        if cfg["initialization"]["method"] == "zero" and final["goals"][i] == 0:
            assert (q == 0).all()
        summary = read(directory / "summary.json")
        model = tuple(inputs[key][i] for key in ["probability", "next_state", "reward", "terminal"])
        fresh = exact_policy(q, model, cfg["environment"]["horizon"], cfg["training"]["gamma"])
        assert abs(fresh["success_probability"] - summary["success_probability"]) < 1e-12
        assert abs(fresh["success_probability"] + fresh["failure_probability"] + fresh["truncation_probability"] - 1) < 1e-10
        sampled = np.atleast_1d(np.genfromtxt(directory / "evaluation.csv", delimiter=",", names=True))
        assert len(sampled) == cfg["evaluation"]["episodes"]
        p = fresh["success_probability"]
        tolerance = 5 * np.sqrt(p * (1 - p) / len(sampled)) + 5 / len(sampled)
        assert abs(sampled["success"].mean() - p) < tolerance
        with (directory / "q_table.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        np.testing.assert_array_equal([[float(r[k]) for k in ["Q_left", "Q_down", "Q_right", "Q_up"]] for r in rows], q)
        goals += int(final["goals"][i])
        episodes += len(log)
    assert read(output / "reference_check.json")["status"] == "passed"
    cp = read(output / "checkpoints.json")
    assert cp[-1]["interactions"] == budget and all(a["compute_seconds"] <= b["compute_seconds"] for a, b in zip(cp, cp[1:]))
    assert 0 < manifest["timing"]["synchronized_compute_seconds"] <= manifest["timing"]["training_and_artifact_seconds"]
    result = dict(status="passed", experiment=cfg["name"], runs=expected, training_interactions=expected * budget,
                  completed_episodes=episodes, training_goals=goals, initialization=cfg["initialization"],
                  numerical_reference="passed", raw_logs_and_snapshots="passed", model_evaluation="passed")
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
