"""Train prepared experiments; no public CPU-training fallback exists."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from .configuration import alpha_at, epsilon_at, validate


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def train(output):
    from .learner import jax, initial_state, make_kernel, synchronize
    output = Path(output)
    cfg = json.loads((output / "config.json").read_text())
    runs = json.loads((output / "runs.json").read_text())
    validate(cfg)
    if (output / "manifest.json").exists():
        raise FileExistsError("A training manifest already exists; preserve this run and use another output path")
    try:
        device = jax.devices("gpu")[0]
    except RuntimeError as error:
        raise RuntimeError("Training requires the configured CUDA runtime; use the documented container command") from error
    if device.platform != "gpu" or not jax.config.read("jax_enable_x64"):
        raise RuntimeError("Expected a CUDA device and float64 mode")
    data = np.load(output / "inputs.npz")
    arrays = tuple(data[key] for key in ["probability", "next_state", "reward", "terminal"])
    inputs = tuple(jax.device_put(x, device) for x in arrays)
    initial = initial_state(cfg, runs, data["terminal_states"], device)
    if initial["q"].devices() != {device}:
        raise RuntimeError("Q table is not on the requested device")
    try:
        hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        hardware = device.device_kind
    source = Path(__file__).parent
    manifest = dict(status="running", experiment=cfg["name"], variant=cfg["variant"],
                    backend=device.platform, device=str(device), device_kind=device.device_kind, hardware=hardware,
                    python_version=platform.python_version(), jax_version=jax.__version__, numpy_version=np.__version__,
                    uid=os.getuid(), gid=os.getgid(), learners=len(runs),
                    interactions_per_learner=cfg["training"]["interactions"], q_dtype="float64",
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))})
    archive = output / "source"
    archive.mkdir()
    for path in source.glob("*.py"):
        (archive / path.name).write_bytes(path.read_bytes())
    write_json(output / "manifest.json", manifest)
    start = time.perf_counter()
    kernel = make_kernel(cfg, len(runs))
    t = time.perf_counter()
    compiled = jax.jit(kernel, backend="gpu").lower(initial, *inputs).compile()
    compile_seconds = time.perf_counter() - t
    t = time.perf_counter()
    trial, events = compiled(initial, *inputs)
    synchronize((trial, events))
    warmup_seconds = time.perf_counter() - t
    # This short prefix is a numerical cross-check, not a second training mode.
    cpu = jax.devices("cpu")[0]
    t = time.perf_counter()
    cpu_initial = initial_state(cfg, runs, data["terminal_states"], cpu)
    cpu_inputs = tuple(jax.device_put(x, cpu) for x in arrays)
    reference_fn = jax.jit(kernel, backend="cpu").lower(cpu_initial, *cpu_inputs).compile()
    reference, reference_events = reference_fn(cpu_initial, *cpu_inputs)
    synchronize((reference, reference_events))
    reference_seconds = time.perf_counter() - t
    np.testing.assert_allclose(np.asarray(trial["q"]), np.asarray(reference["q"]), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.asarray(trial["visits"]), np.asarray(reference["visits"]))
    np.testing.assert_array_equal(np.asarray(events[0]), np.asarray(reference_events[0]))
    write_json(output / "reference_check.json", dict(status="passed", prefix_steps=cfg["training"]["chunk_steps"],
        nonzero_q_entries=int(np.count_nonzero(np.asarray(trial["q"]))),
        max_q_difference=float(np.max(np.abs(np.asarray(trial["q"]) - np.asarray(reference["q"]))))))
    carry = initial_state(cfg, runs, data["terminal_states"], device)
    synchronize(carry)
    files, writers = [], []
    counts = np.zeros(len(runs), dtype=np.int64)
    starts = counts.copy()
    fields = ["episode", "start_step", "end_step", "length", "reward", "discounted_return", "epsilon", "outcome", "mean_abs_td", "max_abs_td", "learning_rate"]
    for index, run in enumerate(runs):
        directory = output / run["group"] / f'map_{run["map_seed"]}' / f'seed_{run["training_seed"]}'
        directory.mkdir(parents=True)
        write_json(directory / "run.json", dict(index=index, **run))
        handle = gzip.open(directory / "episodes.csv.gz", "wt", newline="")
        writer = csv.writer(handle)
        writer.writerow(fields)
        files.append(handle)
        writers.append(writer)
    qshots, vshots, steps = [np.asarray(carry["q"])], [np.asarray(carry["visits"])], [0]
    checkpoints = []
    compute = transfer = logging = 0.
    training_start = time.perf_counter()
    budget, chunk = cfg["training"]["interactions"], cfg["training"]["chunk_steps"]
    for end in range(chunk, budget + 1, chunk):
        t = time.perf_counter()
        carry, events = compiled(carry, *inputs)
        synchronize((carry, events))
        elapsed = time.perf_counter() - t
        compute += elapsed
        t = time.perf_counter()
        host_events = tuple(np.asarray(x) for x in events)
        q, visits = np.asarray(carry["q"]), np.asarray(carry["visits"])
        transfer += time.perf_counter() - t
        t = time.perf_counter()
        for local, index in np.argwhere(host_events[0]):
            step = end - chunk + int(local)
            counts[index] += 1
            writers[index].writerow([counts[index], starts[index], step + 1, int(host_events[1][local, index]),
                host_events[2][local, index], host_events[3][local, index], epsilon_at(cfg, step),
                int(host_events[4][local, index]), host_events[5][local, index], host_events[6][local, index], alpha_at(cfg, step)])
            starts[index] = step + 1
        logging += time.perf_counter() - t
        np.testing.assert_array_equal(visits.sum(axis=(1, 2)), np.full(len(runs), end))
        if not np.isfinite(q).all() or (q < 0).any() or (q > 1 + 1e-12).any():
            raise RuntimeError("Invalid Q estimate")
        qshots.append(q.copy())
        vshots.append(visits.copy())
        steps.append(end)
        checkpoints.append(dict(interactions=end, compute_seconds=compute, transfer_seconds=transfer,
                                logging_seconds=logging, completed_episodes=counts.tolist()))
        print(f'{cfg["name"]}: {end:,}/{budget:,} interactions per learner', flush=True)
    t = time.perf_counter()
    for handle in files:
        handle.close()
    logging += time.perf_counter() - t
    np.savez_compressed(output / "snapshots.npz", steps=np.array(steps), q=np.stack(qshots), visits=np.stack(vshots))
    # Keeping the complete carry makes continuation possible without resetting an
    # unfinished episode or discarding its random key and diagnostic accumulators.
    np.savez_compressed(output / "final_state.npz", **{key: np.asarray(value) for key, value in carry.items()})
    write_json(output / "checkpoints.json", checkpoints)
    manifest.update(status="complete", completed_episodes=counts.tolist(),
                    total_training_interactions=budget * len(runs),
                    timing=dict(compile_seconds=compile_seconds, discarded_warmup_seconds=warmup_seconds,
                        reference_check_seconds=reference_seconds, synchronized_compute_seconds=compute,
                        transfer_seconds=transfer, episode_logging_seconds=logging,
                        training_and_artifact_seconds=time.perf_counter() - training_start,
                        compile_reference_training_seconds=time.perf_counter() - start),
                    final_array_devices=[str(x) for x in carry["q"].devices()])
    write_json(output / "manifest.json", manifest)
    return manifest
