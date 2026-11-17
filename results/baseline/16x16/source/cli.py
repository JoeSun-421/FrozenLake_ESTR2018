"""Small command surface for preparation, training, analysis and inspection."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(prog="frozenlake", description="Learn and inspect policies on stochastic FrozenLake")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("runtime", help="Build the pinned training runtime")
    for name in ["prepare", "run"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--output", required=True)
    for name in ["train", "evaluate", "verify", "plot"]:
        p = sub.add_parser(name)
        p.add_argument("--output", required=True)
        if name == "plot":
            p.add_argument("--figures", default=str(ROOT / "figure"))
    p = sub.add_parser("publish", help="Register an evaluated result for the viewer and report")
    p.add_argument("--output", required=True)
    p.add_argument("--catalog", default=str(ROOT / "results/catalog.json"))
    p = sub.add_parser("compare", help="Compare a matched baseline and initialization control")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("view", help="Open the read-only result viewer")
    p.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "runtime":
        from .runtime import build_runtime
        build_runtime(ROOT)
    elif args.command in ["prepare", "run"]:
        from .configuration import load
        from .environment import prepare
        cfg = load(args.config)
        output = Path(args.output)
        if args.command == "run" and (output / "config.json").exists() and not (output / "manifest.json").exists():
            if json.loads((output / "config.json").read_text()) != cfg:
                raise ValueError("Existing prepared configuration differs")
        else:
            count = prepare(cfg, output)
            print(f"Prepared {count} independent learners")
        if args.command == "run":
            from .runtime import train_in_container
            from .evaluation import evaluate_experiment
            seconds = train_in_container(ROOT, output)
            result = evaluate_experiment(output)
            print(json.dumps(dict(container_wall_seconds=seconds, groups=result["groups"]), indent=2))
    elif args.command == "train":
        from .train import train
        result = train(args.output)
        print(json.dumps(dict(status=result["status"], timing=result["timing"]), indent=2))
    elif args.command == "evaluate":
        from .evaluation import evaluate_experiment
        print(json.dumps(evaluate_experiment(args.output)["groups"], indent=2))
    elif args.command == "verify":
        from .verification import verify
        print(json.dumps(verify(args.output), indent=2))
    elif args.command == "plot":
        from .figures import render_experiment
        render_experiment(args.output, args.figures)
    elif args.command == "publish":
        from .publication import publish
        publish(args.output, args.catalog)
    elif args.command == "compare":
        from .comparison import compare
        print(json.dumps(compare(args.baseline, args.candidate, args.output), indent=2))
    elif args.command == "view":
        from .web import serve
        serve(ROOT, args.port)
