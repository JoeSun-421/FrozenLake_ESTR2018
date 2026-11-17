"""Container orchestration; host analysis never imports the training runtime."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

IMAGE = "estr2018-frozenlake:latest"


def docker(arguments, **kwargs):
    return subprocess.run(["sg", "docker", "-c", shlex.join(["docker", *map(str, arguments)])], check=True, **kwargs)


def build_runtime(root, image=IMAGE):
    root = Path(root).resolve()
    # Reuse the already-installed CUDA/JAX environment on this workstation.
    # On another machine the Dockerfile defaults to a public Python image and
    # installs the same pinned JAX runtime instead of requiring the robotics repo.
    arguments = ["build", "-f", root / "docker/Dockerfile", "-t", image]
    cached = subprocess.run(["sg", "docker", "-c", shlex.join(["docker", "image", "inspect", "estr2018-qlearning-jax:latest"])],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if cached.returncode == 0:
        arguments += ["--build-arg", "BASE_IMAGE=estr2018-qlearning-jax:latest", "--build-arg", "PYTHON_BIN=/.venv/bin/python"]
    docker([*arguments, root / "docker"])


def train_in_container(root, output, image=IMAGE):
    root, output = Path(root).resolve(), Path(output).resolve()
    arguments = ["run", "--rm", "--gpus", "all", "--user", f"{os.getuid()}:{os.getgid()}",
                 "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
                 "-v", f"{root}:/ws:ro", "-v", f"{output}:/results:rw",
                 image, "-m", "frozenlake", "train", "--output", "/results"]
    start = time.perf_counter()
    with (output / "train.log").open("w") as handle:
        docker(arguments, stdout=handle, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - start
    (output / "container_timing.json").write_text(json.dumps(dict(container_wall_seconds=seconds,
        includes="container start, Python/runtime initialization, validation, training and output files"), indent=2) + "\n")
    return seconds
