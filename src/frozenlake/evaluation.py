"""Frozen-policy evaluation, separated from learning and independent of JAX."""
import csv
import json
from pathlib import Path
import time
import numpy as np


def greedy_policy(q):
    mask = q == q.max(axis=1, keepdims=True)
    return mask / mask.sum(axis=1, keepdims=True)


def exact_policy(q, model, horizon, gamma):
    """Propagate probability mass through the finite cutoff, in float64.

    Incoming terminal transitions pay once and remove mass from continuation.
    This is model-based evaluation of a frozen policy, never a training target.
    """
    p, ns, reward, terminal = model
    weights = p * greedy_policy(q)[:, :, None]
    goal_rate = (weights * reward).sum(axis=(1, 2))
    failure_rate = (weights * terminal * (reward == 0)).sum(axis=(1, 2))
    continuation = (weights * (~terminal)).reshape(len(p), -1)
    edges = ns.ravel()
    mass = np.zeros(len(p))
    mass[0] = 1.
    success = failure = discounted = length = success_length = 0.
    for step in range(horizon):
        goal = float(mass @ goal_rate)
        success += goal
        failure += float(mass @ failure_rate)
        discounted += gamma ** step * goal
        success_length += (step + 1) * goal
        length += float(mass.sum())
        mass = np.bincount(edges, weights=(mass[:, None] * continuation).ravel(), minlength=len(p))
    return dict(success_probability=success, failure_probability=failure,
                truncation_probability=float(mass.sum()), discounted_return=discounted,
                mean_episode_length=length, successful_episode_length=success_length / success if success else None)


def monte_carlo(q, model, horizon, gamma, episodes, seed):
    """Vectorized independent rollouts; inactive episodes cannot gain rewards."""
    p, ns, rewards, terminal = model
    rng = np.random.default_rng(seed)
    states = np.zeros(episodes, dtype=np.int32)
    active = np.ones(episodes, dtype=bool)
    result = np.zeros((episodes, 4))
    for step in range(horizon):
        values = q[states]
        ties = values == values.max(axis=1, keepdims=True)
        rank = np.floor(rng.random(episodes) * ties.sum(axis=1)).astype(int) + 1
        action = np.argmax(ties & (np.cumsum(ties, axis=1) == rank[:, None]), axis=1)
        branches = np.minimum(2, (rng.random(episodes)[:, None] >= np.cumsum(p[states, action], axis=1)).sum(axis=1))
        nxt = ns[states, action, branches]
        reward = rewards[states, action, branches]
        done = terminal[states, action, branches]
        result[:, 0] += active * reward
        result[:, 1] += active * gamma ** step * reward
        result[active, 2] = step + 1
        ended = active & done
        result[ended, 3] = np.where(reward[ended] > 0, 1, -1)
        active &= ~done
        states = np.where(active, nxt, 0)
        if not active.any():
            break
    return result


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_experiment(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("Training is not complete")
    cfg = json.loads((output / "config.json").read_text())
    runs = json.loads((output / "runs.json").read_text())
    inputs = np.load(output / "inputs.npz")
    snapshots = np.load(output / "snapshots.npz")
    final = np.load(output / "final_state.npz")
    horizon, gamma = cfg["environment"]["horizon"], cfg["training"]["gamma"]
    models = [tuple(inputs[key][i] for key in ["probability", "next_state", "reward", "terminal"]) for i in range(len(runs))]
    records, at, references = [], {}, {}
    start = time.perf_counter()
    for j, step in enumerate(snapshots["steps"]):
        for i, run in enumerate(runs):
            values = exact_policy(snapshots["q"][j, i], models[i], horizon, gamma)
            record = dict(index=i, group=run["group"], map_seed=run["map_seed"], training_seed=run["training_seed"],
                          interactions=int(step), **values)
            at[(int(step), i)] = record
            records.append(record)
    exact_seconds = time.perf_counter() - start
    start = time.perf_counter()
    summaries = []
    for i, run in enumerate(runs):
        directory = output / run["group"] / f'map_{run["map_seed"]}' / f'seed_{run["training_seed"]}'
        q = snapshots["q"][-1, i]
        ref_key = (run["group"], run["map_seed"])
        if ref_key not in references:
            references[ref_key] = exact_policy(np.zeros_like(q), models[i], horizon, gamma)
        samples = monte_carlo(q, models[i], horizon, gamma, cfg["evaluation"]["episodes"],
                              run["map_seed"] * 1000 + run["training_seed"] + cfg["evaluation"]["seed_offset"])
        with (directory / "evaluation.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["success", "discounted_return", "length", "outcome"])
            writer.writerows(samples)
        summary = dict(at[(cfg["training"]["interactions"], i)],
                       monte_carlo_success=float(samples[:, 0].mean()), evaluation_episodes=len(samples),
                       random_success=references[ref_key]["success_probability"],
                       random_return=references[ref_key]["discounted_return"],
                       completed_training_episodes=int(final["episodes"][i]), training_goals=int(final["goals"][i]),
                       nonzero_q_entries=int(np.count_nonzero(q)),
                       visited_state_action_fraction=float((snapshots["visits"][-1, i, ~inputs["terminal_states"][i]] > 0).mean()))
        summaries.append(summary)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        rows = []
        size = cfg["environment"]["size"]
        for state in range(size * size):
            row, col = divmod(state, size)
            record = dict(state=state, row=row + 1, column=col + 1, tile=run["rows"][row][col])
            record.update(dict(zip(["Q_left", "Q_down", "Q_right", "Q_up"], map(float, q[state]))))
            record["visits"] = int(snapshots["visits"][-1, i, state].sum())
            rows.append(record)
        write_csv(directory / "q_table.csv", rows)
    sampling_seconds = time.perf_counter() - start
    groups = {}
    for name in cfg["groups"]:
        selected = [s for s in summaries if s["group"] == name]
        if not selected:
            continue
        groups[name] = dict(runs=len(selected), mean_success=float(np.mean([s["success_probability"] for s in selected])),
            minimum_success=min(s["success_probability"] for s in selected), maximum_success=max(s["success_probability"] for s in selected),
            monte_carlo_mean=float(np.mean([s["monte_carlo_success"] for s in selected])),
            mean_return=float(np.mean([s["discounted_return"] for s in selected])),
            random_success=float(np.mean([s["random_success"] for s in selected])),
            random_return=float(np.mean([s["random_return"] for s in selected])),
            training_goals=sum(s["training_goals"] for s in selected),
            zero_q_tables=sum(s["nonzero_q_entries"] == 0 for s in selected))
    pres = cfg["presentation"]
    index = next(i for i, r in enumerate(runs) if r["group"] == "main" and r["map_seed"] == pres["map_seed"] and r["training_seed"] == pres["training_seed"])
    aggregate = dict(experiment=cfg["name"], variant=cfg["variant"], size=cfg["environment"]["size"],
                     initialization=cfg["initialization"], groups=groups, representative=summaries[index],
                     evaluation=dict(horizon=horizon, gamma=gamma, exploratory=False, ties="uniform", exact_seconds=exact_seconds,
                                     monte_carlo_and_serialization_seconds=sampling_seconds))
    write_csv(output / "evaluation_curve.csv", records)
    write_csv(output / "summary.csv", summaries)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    # Lightweight, language-neutral data for the read-only browser viewer.
    curve = [at[(int(s), index)]["success_probability"] for s in snapshots["steps"]]
    means = [float(np.mean([at[(int(s), i)]["success_probability"] for i, r in enumerate(runs) if r["group"] == "main"])) for s in snapshots["steps"]]
    viewer = dict(name=cfg["name"], size=cfg["environment"]["size"], rows=runs[index]["rows"],
                  steps=snapshots["steps"].tolist(), q=snapshots["q"][:, index].tolist(),
                  visits=snapshots["visits"][:, index].tolist(), success=curve, mean_success=means,
                  config=cfg, aggregate=aggregate)
    (output / "viewer.json").write_text(json.dumps(viewer, separators=(",", ":")) + "\n")
    return aggregate
