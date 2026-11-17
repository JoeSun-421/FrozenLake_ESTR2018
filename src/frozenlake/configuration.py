"""Resolve small JSON overlays and keep published protocols explicit."""
from copy import deepcopy
import json
from pathlib import Path


def merge(base, overlay):
    result = deepcopy(base)
    for key, value in overlay.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def load(path, _parents=()):
    path = Path(path).resolve()
    if path in _parents:
        raise ValueError("Configuration inheritance cycle")
    raw = json.loads(path.read_text())
    parent = raw.pop("extends", None)
    cfg = merge(load(path.parent / parent, _parents + (path,)), raw) if parent else raw
    validate(cfg)
    return cfg


def validate(cfg):
    env, train, init = cfg["environment"], cfg["training"], cfg["initialization"]
    if not isinstance(env["size"], int) or env["size"] < 2:
        raise ValueError("size must be an integer >= 2")
    if not 0 <= env["hole_fraction"] < 1 or not 0 <= env["intended_probability"] <= 1:
        raise ValueError("Invalid environment probability")
    if env["horizon"] <= 0:
        raise ValueError("horizon must be positive")
    if train["interactions"] <= 0 or train["chunk_steps"] <= 0 or train["interactions"] % train["chunk_steps"]:
        raise ValueError("interaction budget must be a positive multiple of chunk_steps")
    if not 0 <= train["gamma"] < 1:
        raise ValueError("gamma must be in [0, 1)")
    for key in ["initial", "final"]:
        if not 0 < train["alpha"][key] <= 1 or not 0 <= train["epsilon"][key] <= 1:
            raise ValueError("Invalid learning/exploration rate")
    if train["epsilon"]["decay_steps"] <= 0 or train["alpha"]["switch_step"] < 0:
        raise ValueError("Invalid absolute schedule threshold")
    if cfg["variant"] not in ["baseline", "optimistic"]:
        raise ValueError("variant must be baseline or optimistic")
    if init["method"] not in ["zero", "optimistic"] or not 0 <= init["value"] <= 1:
        raise ValueError("initialization must be zero or optimistic, with value in [0, 1]")
    if init["method"] == "zero" and init["value"] != 0:
        raise ValueError("zero initialization requires value=0")
    if cfg["variant"] == "baseline" and init["method"] != "zero":
        raise ValueError("baseline results must remain zero-initialized")
    if not any(g["map_seeds"] and g["training_seeds"] for g in cfg["groups"].values()):
        raise ValueError("At least one nonempty experiment group is required")
    pairs = []
    for group in cfg["groups"].values():
        if len(set(group["map_seeds"])) != len(group["map_seeds"]) or len(set(group["training_seeds"])) != len(group["training_seeds"]):
            raise ValueError("Duplicate seeds would repeat a learner")
        if any(not isinstance(s, int) or s < 0 for s in group["map_seeds"] + group["training_seeds"]):
            raise ValueError("Seeds must be nonnegative integers")
        pairs.extend((m, s) for m in group["map_seeds"] for s in group["training_seeds"])
    if len(set(pairs)) != len(pairs) or len({m * 100000 + s for m, s in pairs}) != len(pairs):
        raise ValueError("Groups must use distinct map/seed pairs and random keys")
    if cfg["evaluation"]["episodes"] <= 0:
        raise ValueError("evaluation episodes must be positive")
    if any(s < 0 or s > train["interactions"] or s % train["chunk_steps"] for s in cfg["presentation"]["snapshots"]):
        raise ValueError("presentation snapshots must be saved checkpoints")


def alpha_at(cfg, step):
    schedule = cfg["training"]["alpha"]
    return schedule["final"] if step >= schedule["switch_step"] else schedule["initial"]


def epsilon_at(cfg, step):
    schedule = cfg["training"]["epsilon"]
    return schedule["initial"] + (schedule["final"] - schedule["initial"]) * min(1, step / schedule["decay_steps"])
