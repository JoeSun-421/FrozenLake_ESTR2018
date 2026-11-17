"""Connected fixed-density maps and Gymnasium's authoritative transition model."""
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np


def shortest_path(rows):
    n = len(rows)
    queue, seen = deque([(0, 0)]), {0}
    while queue:
        state, distance = queue.popleft()
        if state == n * n - 1:
            return distance
        r, c = divmod(state, n)
        for dr, dc in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            rr, cc = r + dr, c + dc
            ns = rr * n + cc
            if 0 <= rr < n and 0 <= cc < n and rows[rr][cc] != "H" and ns not in seen:
                seen.add(ns)
                queue.append((ns, distance + 1))
    return None


def generate_map(size, seed, hole_fraction):
    rng = np.random.default_rng(seed)
    count = round(hole_fraction * (size * size - 2))
    for attempt in range(1, 10001):
        tiles = np.full(size * size, "F", dtype="<U1")
        tiles[rng.choice(np.arange(1, size * size - 1), count, replace=False)] = "H"
        tiles[0], tiles[-1] = "S", "G"
        rows = ["".join(row) for row in tiles.reshape(size, size)]
        if shortest_path(rows) is not None:
            return rows, attempt
    raise RuntimeError("Could not generate a connected map")


def build_model(rows, intended_probability):
    import gymnasium as gym
    env = gym.make("FrozenLake-v1", desc=rows, is_slippery=True, success_rate=intended_probability)
    states = len(rows) ** 2
    p = np.zeros((states, 4, 3), dtype=np.float64)
    ns = np.zeros((states, 4, 3), dtype=np.int32)
    reward = np.zeros((states, 4, 3), dtype=np.float64)
    done = np.zeros((states, 4, 3), dtype=bool)
    for state in range(states):
        for action in range(4):
            for k, (prob, nxt, r, end) in enumerate(env.unwrapped.P[state][action]):
                p[state, action, k], ns[state, action, k] = prob, nxt
                reward[state, action, k], done[state, action, k] = r, end
    env.close()
    if not np.allclose(p.sum(axis=-1), 1):
        raise ValueError("Transition probabilities do not sum to one")
    return p, ns, reward, done


def prepare(cfg, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    models, runs = [], []
    env = cfg["environment"]
    for group, spec in cfg["groups"].items():
        for seed in spec["map_seeds"]:
            rows, attempts = generate_map(env["size"], seed, env["hole_fraction"])
            model = build_model(rows, env["intended_probability"])
            for training_seed in spec["training_seeds"]:
                models.append(model)
                runs.append(dict(group=group, map_seed=seed, training_seed=training_seed, rows=rows,
                                 generation_attempts=attempts, shortest_path=shortest_path(rows)))
    np.savez_compressed(output / "inputs.npz", probability=np.stack([m[0] for m in models]),
                        next_state=np.stack([m[1] for m in models]), reward=np.stack([m[2] for m in models]),
                        terminal=np.stack([m[3] for m in models]),
                        terminal_states=np.array([[c in "HG" for row in run["rows"] for c in row] for run in runs]))
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (output / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    provenance = dict(input_sha256=hashlib.sha256((output / "inputs.npz").read_bytes()).hexdigest(),
                      config_sha256=hashlib.sha256((output / "config.json").read_bytes()).hexdigest())
    (output / "input_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return len(runs)
