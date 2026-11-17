"""Curated scientific figures; plotting is independent of the training runtime."""
import csv
import gzip
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
COLORS = {8: "#16697A", 12: "#D58A19", 16: "#B54B57"}


def read(path):
    return json.loads(Path(path).read_text())


def save(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        fig.savefig(output / f"{name}.{ext}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def map_axis(ax, rows, q=None, visits=None):
    n = len(rows)
    values = np.zeros((n, n)) if q is None else q.max(axis=1).reshape(n, n)
    im = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1)
    for r in range(n):
        for c in range(n):
            tile, state = rows[r][c], r * n + c
            if tile in "HG":
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor="#223344" if tile == "H" else "#2B8061", edgecolor="white"))
                ax.text(c, r, tile, ha="center", va="center", color="white", weight="bold", fontsize=10 if n <= 8 else 7)
            elif q is not None:
                choices = np.flatnonzero(q[state] == q[state].max())
                arrow = ["←", "↓", "→", "↑"][int(choices[0])] if len(choices) == 1 else "·"
                ax.text(c, r, f"{values[r,c]:.2f}\n{arrow}", ha="center", va="center", fontsize=10 if n <= 8 else 6,
                        color="white" if values[r, c] > .6 else "#102B3A")
                if visits is not None and visits[state].sum() == 0:
                    ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, fill=False, hatch="///", edgecolor="#999999", linewidth=0))
            if tile == "S":
                ax.add_patch(Rectangle((c - .46, r - .46), .92, .92, fill=q is None,
                                      facecolor="#E7AA4A" if q is None else "none", edgecolor="#D58A19", linewidth=2))
                if q is None:
                    ax.text(c, r, "S", ha="center", va="center", weight="bold", fontsize=16)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.6)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    return im


def curve_data(output):
    cfg = read(output / "config.json")
    runs = read(output / "runs.json")
    with (output / "evaluation_curve.csv").open() as handle:
        raw = list(csv.DictReader(handle))
    steps = sorted({int(r["interactions"]) for r in raw})
    at = {(int(r["interactions"]), int(r["index"])): float(r["success_probability"]) for r in raw}
    ids = [i for i, r in enumerate(runs) if r["group"] == "main"]
    map_seeds = cfg["groups"]["main"]["map_seeds"]
    maps = np.array([np.mean([[at[(step, i)] for step in steps] for i in ids if runs[i]["map_seed"] == m], axis=0) for m in map_seeds])
    pres = cfg["presentation"]
    index = next(i for i, r in enumerate(runs) if r["group"] == "main" and r["map_seed"] == pres["map_seed"] and r["training_seed"] == pres["training_seed"])
    return cfg, runs, steps, at, maps, index


def render_experiment(output, figure_dir):
    output, figure_dir = Path(output), Path(figure_dir)
    cfg, runs, steps, at, maps, index = curve_data(output)
    n = cfg["environment"]["size"]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.plot(np.array(steps) / 1000, maps.mean(axis=0) * 100, color=COLORS.get(n, "#16697A"), linewidth=2.4, label="Mean: 3 maps × 3 training seeds")
    ax.fill_between(np.array(steps) / 1000, maps.min(axis=0) * 100, maps.max(axis=0) * 100, color=COLORS.get(n, "#16697A"), alpha=.13, label="Range of map means")
    ax.plot(np.array(steps) / 1000, [at[(s, index)] * 100 for s in steps], ":", color="#D58A19", label="Predesignated representative run")
    ax.axhline(95, color="#777777", linestyle="--", linewidth=1, label="95% reference")
    ax.set(title=f"{n}×{n} FrozenLake: policy learning", xlabel="Training interactions (thousands)", ylabel="Success probability (%)", ylim=(-2, 104))
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.22)
    fig.text(.5, .03, "Frozen greedy-policy evaluation; slipping remains and tied actions are sampled uniformly.\n"
             "Shading is the range of map means, not a confidence interval. All points are saved observations.", ha="center", fontsize=10)
    save(fig, figure_dir, f"learning_curve_{n}x{n}")
    if n != 8:
        return
    snapshots = np.load(output / "snapshots.npz")
    desc = runs[index]["rows"]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    map_axis(ax, desc)
    ax.set_title("The FrozenLake navigation problem", pad=15)
    fig.subplots_adjust(bottom=.15)
    fig.text(.5, .035, "S: start     H: hole     G: goal\nThe chosen action and actual movement can differ.", ha="center", fontsize=11)
    save(fig, figure_dir, "problem_8x8")
    cps = read(output / "checkpoints.json")
    episodes = {0: 0, **{r["interactions"]: r["completed_episodes"][index] for r in cps}}
    pres = cfg["presentation"]["snapshots"]
    for suffix, selected, title in [("early", pres[:2], "Early learning"), ("late", pres[2:], "Later learning")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 6.3))
        for ax, step in zip(axes, selected):
            j = int(np.flatnonzero(snapshots["steps"] == step)[0])
            im = map_axis(ax, desc, snapshots["q"][j, index], snapshots["visits"][j, index])
            ax.set_title(f'{step:,} interactions\n{episodes[step]:,} completed episodes\nPolicy success: {100*at[(step,index)]:.2f}%', fontsize=12, pad=12)
        fig.suptitle(f"8×8 Q-map: {title.lower()}", fontsize=21, y=.98)
        fig.subplots_adjust(left=.025, right=.885, top=.78, bottom=.19, wspace=.12)
        cax = fig.add_axes([.915, .25, .018, .46])
        fig.colorbar(im, cax=cax, label=r"$\max_a Q(s,a)$", ticks=np.linspace(0, 1, 6))
        fig.text(.46, .073, "Cell values are max Q, not success probabilities. Arrows: unique greedy actions; dots: ties.\n"
                 "H/G: terminal Q = 0. Hatch: unvisited. Both figures use the same 0–1 color scale.", ha="center", fontsize=10)
        fig.text(.46, .025, f'Map {runs[index]["map_seed"]} / seed {runs[index]["training_seed"]}; initial Q = {cfg["initialization"]["value"]:g}', ha="center", fontsize=10, color="#555555")
        save(fig, figure_dir, f"qmap_8x8_{suffix}")
    run = runs[index]
    path = output / run["group"] / f'map_{run["map_seed"]}' / f'seed_{run["training_seed"]}'
    with gzip.open(path / "episodes.csv.gz", "rt") as handle:
        data = np.atleast_1d(np.genfromtxt(handle, delimiter=",", names=True))
    if len(data) >= 100:
        smooth = np.convolve(data["reward"], np.ones(100) / 100, mode="valid")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.plot(np.arange(100, len(data) + 1), smooth, color="#2563B8", linewidth=1.3)
        ax.set(title="8×8 FrozenLake: training behavior", xlabel="Completed training episode", ylabel="Mean reward over the last 100 episodes", ylim=(-.02, 1.02))
        ax.grid(alpha=.25)
        fig.subplots_adjust(bottom=.22)
        fig.text(.5, .03, "Exploratory training includes ε-greedy actions and slipping; this is not frozen-policy success.\n"
                 "The first trailing-average point ends at episode 100.", ha="center", fontsize=10)
        save(fig, figure_dir, "training_return_8x8")
    equations = {
        "q_learning_update": r"Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha_t\left[r_{t+1}+\gamma(1-d_t)\max_{a^\prime}Q(s_{t+1},a^\prime)-Q(s_t,a_t)\right]",
        "bellman_optimality": r"Q^*(s,a)=\sum_{s^\prime}P(s^\prime\mid s,a)\left[r(s,a,s^\prime)+\gamma\max_{a^\prime}Q^*(s^\prime,a^\prime)\right]"
    }
    for name, equation in equations.items():
        fig = plt.figure(figsize=(16, 1.2))
        fig.text(.5, .5, f"${equation}$", ha="center", va="center", fontsize=23)
        save(fig, figure_dir, name)


def render_summary(catalog, figure_dir):
    catalog = Path(catalog)
    root = catalog.resolve().parent.parent
    items = [r for r in read(catalog)["experiments"] if r["variant"] == "baseline"]
    results = [(r, root / r["path"], read(root / r["path"] / "aggregate.json")) for r in items]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    xx = np.arange(len(results))
    for ax, key, reference, scale, title in [(axes[0], "mean_success", "random_success", 100, "Do learned policies reach the goal?"),
        (axes[1], "mean_return", "random_return", 1, "Expected discounted return")]:
        for offset, field, label, color in [(-.17, key, "Q-learning", "#16697A"), (.17, reference, "Uniform random", "#B9C4CC")]:
            values = [a["groups"]["main"][field] * scale for _, _, a in results]
            ax.bar(xx + offset, values, .34, label=label, color=color)
            for x, value in zip(xx + offset, values):
                if value < (.01 if scale == 100 else .001):
                    continue
                ax.text(x, value + (1.5 if scale == 100 else .012), f"{value:.2f}%" if scale == 100 else f"{value:.4f}", ha="center", fontsize=10)
        ax.set(xticks=xx, xticklabels=[f'{r["size"]}×{r["size"]}' for r, _, _ in results], title=title,
               ylabel="Success probability (%)" if scale == 100 else "Discounted return", ylim=(0, 114 if scale == 100 else .65))
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=9, loc="upper center")
    fig.subplots_adjust(bottom=.23, wspace=.28)
    fig.text(.5, .035, "Three maps × three training seeds per environment; all main Q tables start at zero.\n"
             "Budgets and learning-rate schedules differ. This tests policy quality, not a controlled size or speed scaling law.\n"
             "Tiny baseline/16×16 bars may be below 0.01%; full precision is retained in the numerical tables.", ha="center", fontsize=10)
    save(fig, figure_dir, "policy_quality")
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5.2), squeeze=False)
    for ax, (item, output, _) in zip(axes[0], results):
        cfg, _, steps, _, maps, _ = curve_data(output)
        ax.plot(np.array(steps) / 1000, maps.mean(axis=0) * 100, color=COLORS[item["size"]], linewidth=2)
        ax.fill_between(np.array(steps) / 1000, maps.min(axis=0) * 100, maps.max(axis=0) * 100, color=COLORS[item["size"]], alpha=.13)
        ax.set(title=f'{item["size"]}×{item["size"]}: {cfg["training"]["interactions"]:,} interactions', xlabel="Interactions (thousands)", ylabel="Policy success (%)", ylim=(-2, 104))
        ax.grid(alpha=.2)
    fig.subplots_adjust(bottom=.21, wspace=.30)
    fig.text(.5, .03, "Mean of nine runs; shading is the range of map means, not a confidence interval.\n"
             "Each task is trained independently under its stated protocol; no cross-size policy transfer is tested.", ha="center", fontsize=10)
    save(fig, figure_dir, "learning_curves")
    sixteen = next(((p, a) for r, p, a in results if r["size"] == 16), None)
    if sixteen:
        output, aggregate = sixteen
        cfg, runs, _, _, _, index = curve_data(output)
        snapshots = np.load(output / "snapshots.npz")
        visits = snapshots["visits"][-1, index].sum(axis=1).reshape(16, 16)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.3))
        im = axes[0].imshow(np.log10(visits + 1), cmap="YlGnBu")
        axes[0].set(title="16×16: uneven state visitation", xlabel="Column", ylabel="Row")
        fig.colorbar(im, ax=axes[0], label="log10(visits + 1)")
        rows = []
        for i, run in enumerate(runs):
            if run["group"] == "main":
                summary = read(output / "main" / f'map_{run["map_seed"]}' / f'seed_{run["training_seed"]}' / "summary.json")
                rows.append(summary["training_goals"])
        axes[1].bar(range(len(rows)), rows, color="#B54B57")
        axes[1].set(title="Positive reward discovery in nine runs", xlabel="Main training run", ylabel="Goal rewards observed", xticks=range(len(rows)), xticklabels=range(1, len(rows) + 1))
        fig.subplots_adjust(bottom=.23, wspace=.30)
        stats = aggregate["groups"]["main"]
        fig.text(.5, .035, f'Nine runs: {stats["training_goals"]} goal rewards; {stats["zero_q_tables"]} final Q tables remain zero.\n'
                 "Visitation panel: fixed map 2018 / seed 11. Diagnostics do not isolate all causal factors.", ha="center", fontsize=10)
        save(fig, figure_dir, "diagnostics_16x16")
