"""Write the English overview and report from verified published observations."""
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def percentage(value):
    return f"{100*value:.4f}%" if 0 < value < .0001 else f"{100*value:.2f}%"


def write_documents(root):
    root = Path(root)
    catalog = read(root / "results/catalog.json")
    entries = [r for r in catalog["experiments"] if r["variant"] == "baseline"]
    data = {r["size"]: dict(path=root / r["path"], aggregate=read(root / r["path"] / "aggregate.json"),
                          config=read(root / r["path"] / "config.json"), manifest=read(root / r["path"] / "manifest.json")) for r in entries}
    if set(data) != {8, 12, 16}:
        raise ValueError("The retained report requires verified 8×8, 12×12 and 16×16 baselines")
    for item in data.values():
        if read(item["path"] / "verification.json")["status"] != "passed":
            raise ValueError("Unverified experiment")
    def main(n):
        return data[n]["aggregate"]["groups"]["main"]
    def val(n):
        return data[n]["aggregate"]["groups"]["validation"]
    results_table = "\n".join(f'| {n}×{n} | {data[n]["config"]["training"]["interactions"]:,} | {percentage(main(n)["mean_success"])} | {percentage(main(n)["random_success"])} | {percentage(val(n)["mean_success"])} |' for n in [8, 12, 16])
    rep = data[8]["aggregate"]["representative"]
    with (data[8]["path"] / "evaluation_curve.csv").open() as handle:
        records = list(csv.DictReader(handle))
    at50 = float(np.mean([float(r["success_probability"]) for r in records if r["group"] == "main" and int(r["interactions"]) == 50000]))
    timing_table = "\n".join(f'| {n}×{n} | {data[n]["manifest"]["learners"]} | {data[n]["manifest"]["timing"]["synchronized_compute_seconds"]:.3f} | {data[n]["manifest"]["timing"]["training_and_artifact_seconds"]:.3f} | {read(data[n]["path"] / "container_timing.json")["container_wall_seconds"]:.3f} |' for n in [8, 12, 16])
    alpha_text = lambda n: (f'{data[n]["config"]["training"]["alpha"]["initial"]} → {data[n]["config"]["training"]["alpha"]["final"]} at {data[n]["config"]["training"]["alpha"]["switch_step"]:,}' if n == 8 else str(data[n]["config"]["training"]["alpha"]["initial"]))
    protocol_table = "\n".join(f'| {n}×{n} | {n*n} | {n*n*4} | {round(.1*(n*n-2))} | {data[n]["config"]["environment"]["horizon"]} | {data[n]["config"]["training"]["interactions"]:,} | {alpha_text(n)} | {data[n]["config"]["training"]["epsilon"]["decay_steps"]:,} |' for n in [8, 12, 16])
    main_goals = main(16)["training_goals"]
    zero_tables = main(16)["zero_q_tables"]
    candidate = next((r for r in catalog["experiments"] if r["id"] == "optimistic_16"), None)
    if candidate:
        from .comparison import compare
        control = compare(data[16]["path"], root / candidate["path"], root / "results/optimistic/comparison")
        control_status = (f'A separately verified matched control is now published: mean success changes from '
            f'{percentage(control["baseline_mean"])} to {percentage(control["candidate_mean"])}. '
            'Its paired observations are in results/optimistic/comparison; it is not part of the baseline table.')
    else:
        control_status = "No matched optimistic-control result is published yet. The configured control is future work, not a measured improvement."
    readme = f'''# Learning Policies on FrozenLake

**Can Q-learning learn useful navigation policies when movement is uncertain?**

This ESTR2018 project models FrozenLake as a finite Markov decision process
(MDP) and learns tabular action values from sampled experience. We test 8×8,
12×12 and 16×16 lakes. The aim is policy effectiveness, rather than a controlled
comparison of training speed across grid sizes.

![Policy learning on an 8×8 lake](figure/learning_curve_8x8.png)

## Results

Each main result is the mean of **nine independent training runs: three maps ×
three training seeds**. Every baseline starts with **Q₀ = 0**. The table uses
frozen-policy success probabilities computed from the environment model;
independent Monte Carlo evaluation is also retained. Slipping remains enabled.

| Lake | Interactions per run | Mean policy success | Uniform random | Additional-map check |
| --- | ---: | ---: | ---: | ---: |
{results_table}

The 8×8 mean rises from **{percentage(at50)} at 50,000 interactions** to
**{percentage(main(8)["mean_success"])} at 100,000** within the same runs. The
predesignated example has **{percentage(rep["success_probability"])}** exact
success and **{percentage(rep["monte_carlo_success"])}** success over 5,000
independent evaluation episodes. One example is not the nine-run mean.

The 16×16 baseline remains close to random under its stated budget. Its nine
main runs discover only **{main_goals} goal rewards**, and **{zero_tables} final Q
tables remain zero**. This is an observed limitation of this preset, not proof
that every 16×16 lake is unsolvable with Q-learning.

![Policy quality across the three environments](figure/policy_quality.png)

**Interpretation matters:** budgets, learning-rate schedules and episode cutoffs
differ by size. Additional-map checks retrain from scratch; they do not test
direct transfer of one policy to unseen maps. The report describes these limits.

## Task and method

The agent starts in the upper-left and aims for the lower-right goal while
avoiding holes. A chosen action moves in the intended direction or either
perpendicular direction with probability 1/3 each. Boundary moves stay in place.
Entering the goal pays +1 once; every other transition pays 0. Goal and hole
states terminate an episode.

Q-learning updates one state–action entry per interaction. It does not read a
transition expectation or use the separately computed evaluation as training
feedback. Training uses ε-greedy exploration; evaluation freezes Q and removes
ε exploration, with uniform tie-breaking among maximal actions.

![Q-learning sample update](figure/q_learning_update.svg)

The terminal indicator dₜ is one only at the goal or a hole. External time-limit
truncation retains bootstrapping. Q-map colors show estimated discounted return,
not success probability.

![Later policy estimates](figure/qmap_8x8_late.png)

## Project layout

```text
src/frozenlake/     configuration, environments, learner, evaluation and CLI
config/            three baselines and a matched optimistic-16 control
figure/            curated PNG/SVG assets used by these documents
test/              numerical, protocol and browser checks
gui/               read-only checkpoint and four-action Q-table viewer
docker/            pinned training environment
results/baseline/  configurations, logs, Q snapshots, evaluations and provenance
report/            experiment report and presentation text
references/        bibliography, course instructions and the replay paper
```

There is one training implementation. Historical alternative implementations,
parameter searches and duplicate images are excluded from the active project.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (0.13.8 or
later), then sync the locked environment. The project supports Python 3.10–3.12;
`.python-version` selects 3.10 for local analysis and the viewer.

```bash
uv sync --locked
uv run --locked python -m unittest discover -s test -v
```

`pyproject.toml` declares dependencies and `uv.lock` pins their resolved versions.
Commit both files together. The `train` dependency group pins JAX 0.5.3 and NumPy
1.26.4 for the training container; the default sync installs the analysis/viewer
environment. Docker synchronizes only the training group from the same lock.
The `run` command starts training inside that container.
Use `uv add <package>` to add a dependency and `uv lock --upgrade-package <package>`
to deliberately update one. See [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/).

Build the training environment once, then use a **new output directory**:

```bash
uv run --locked frozenlake runtime
uv run --locked frozenlake run --config config/baseline_8.json --output results/runs/example_8
uv run --locked frozenlake verify --output results/runs/example_8
uv run --locked frozenlake plot --output results/runs/example_8 --figures results/runs/example_8/figure
```

After a result has been verified, `publish --output <directory>` registers it in
the read-only viewer. `uv run --locked frozenlake report` regenerates this
README, the report and summary figures from the published catalog.

The runtime requires a CUDA-capable NVIDIA device and NVIDIA Container Toolkit.
The retained runs used an RTX 4090 D, JAX 0.5.3 and float64 action values. The
runtime builder can start from a local cached image or a public Python base.
Both paths run `uv sync --locked --only-group train` before training. Container
commands use `sg docker -c` and execute as the host UID/GID. The repository mount
is read-only and result files are owned by the host user.

Existing results are protected from overwriting. `prepare`, `train`, `evaluate`,
`verify` and `publish` are also available as separate stages. Run
`uv run --locked frozenlake --help` for the command list.

## Inspect a policy

```bash
uv run --locked frozenlake view
```

Open **http://127.0.0.1:8765/gui/**. Choose a lake, move through checkpoints,
switch between maximum Q and any of the four action values, and select a cell
to inspect full-precision values and visit counts. This viewer is read-only.

## Planned 16×16 control

The baseline remains zero-initialized. The separate optimistic configuration
inherits the same maps, seeds, dynamics, budget and schedules, changing only the
nonterminal initial Q value. It is **not included in the published baseline**.

```bash
uv run --locked frozenlake run --config config/optimistic_16.json --output results/optimistic/16x16
uv run --locked frozenlake verify --output results/optimistic/16x16
uv run --locked frozenlake compare --baseline results/baseline/16x16 --candidate results/optimistic/16x16 --output results/optimistic/comparison
```

The comparison rejects changed maps, schedules, budgets or training code, and
writes paired results. After verification, publish the candidate to add it as a
separate viewer entry and include the matched result in a regenerated report.
Replay and DQN remain later research options; neither is an implemented solution.

{control_status}

## Evidence and documents

- [Experiment report](report/report.md): probability model, protocol, numerical
  results, measured timing, limitations and the next experiment.
- [Presentation text](report/presentation.md): bilingual material for a title and
  five content slides.
- [Published experiment catalog](results/catalog.json): links to the raw results.
- Each result has `config.json`, `runs.json`, `inputs.npz`, `snapshots.npz`,
  `final_state.npz`, `evaluation_curve.csv`, `summary.csv`, `aggregate.json`,
  runtime/provenance manifests and a verification record. Per-run folders retain
  every completed episode and the full final four-action `q_table.csv`.

## References

- Watkins and Dayan (1992), [Q-learning](https://doi.org/10.1007/BF00992698).
- Sutton and Barto (2018), [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/9780262039246/reinforcement-learning/), 2nd ed.
- Farama Foundation, [FrozenLake specification](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).
- We also consulted [Stanford CS234: Reinforcement Learning](https://web.stanford.edu/class/archive/cs/cs234/cs234.1264/modules.html), especially Lectures 2 and 4.
- The 100-episode moving-average plot was informed by the [Sesen.ai FrozenLake tutorial](https://sesen.ai/blog/q-learning-frozen-lake-from-scratch); its 4×4 results and training budgets are not our results.
- Full citations, including planned replay/DQN references, are in [references.bib](references/references.bib).
'''
    (root / "README.md").write_text(readme)
    report = f'''# Learning Navigation Policies on Stochastic FrozenLake

**ESTR2018 — Markov Decision Processes in Reinforcement Learning**

## Abstract

We investigate whether one-step tabular Q-learning can learn useful policies
from interaction in stochastic FrozenLake environments. Three map sizes are
used as task examples, with separately specified budgets and learning-rate
schedules. All main Q tables start at zero. The main-map mean final success
probabilities are {percentage(main(8)["mean_success"])}, {percentage(main(12)["mean_success"])}
and {percentage(main(16)["mean_success"])} for 8×8, 12×12 and 16×16 respectively.
The largest baseline discovers little positive feedback. We retain raw logs,
action-value checkpoints, model-based policy evaluation, independent sampled
evaluation and timing. A matched optimistic-initialization control is configured
as a separate experiment; it does not alter the zero-initialized baseline table.

## 1. Research question

Can an agent learn an effective navigation policy through experience when its
actions have uncertain outcomes? The three sizes provide environments in which
to examine learning and failure. The contribution is a transparent implementation
and preliminary numerical study, not a new algorithm or a universal scaling law.

The project follows Topic 10 of the [course instructions](../references/project_instructions.pdf).
MDP and Q-learning background follows Watkins and Dayan [1], Sutton and Barto [2],
and the [Stanford CS234 materials](https://web.stanford.edu/class/archive/cs/cs234/cs234.1264/modules.html).

## 2. Problem formulation and probability

The current cell is the state; actions are left, down, right and up. The lake
contains a start S, safe cells F, terminal holes H and terminal goal G. The start
is upper-left and the goal lower-right. Maps use a fixed rounded hole count,
`round(0.10 × (n² − 2))`, giving 6, 14 and 25 holes. Generation rejects maps
without a hole-free path; connectivity does not guarantee safe execution.

![Task example](../figure/problem_8x8.png)

For an action pointing right, actual right, up and down movement each have
probability 1/3. Other actions have the analogous perpendicular alternatives.
Out-of-bounds movement remains in place, so multiple branches may merge. For
example, at the upper-left corner, choosing left stays with probability 2/3 and
moves down with probability 1/3. Thus arrows toward a boundary can be purposeful.
Transition and reward semantics are taken from Gymnasium 1.4.0 [3].

Entering G pays +1 once; other transitions pay zero. G and H terminate. The
objective is expected discounted return: immediate reward plus gamma times the
continuation return, averaged over possible future outcomes. If the only reward
occurs on move T, that episode returns γ^(T−1). With γ < 1, this objective differs
from undiscounted success probability; both metrics are reported separately.

The Markov property makes the next-state distribution depend on the current
state and chosen action. Conditioning on that next state yields the Bellman
expectation, while selecting the best available continuation action yields the
Bellman optimality relation [1,2]. Q-learning approximates this relation from
individual samples, without a model expectation in its update.

![Bellman optimality relation](../figure/bellman_optimality.svg)

## 3. Learning algorithm

![One-step Q-learning update](../figure/q_learning_update.svg)

Here dₜ is the true termination indicator. It is one at G/H, so there is no
continuation target after termination. The sampling cutoff H=4n² is external
truncation: it restarts sampling but does not remove the bootstrap term. The
current position is the policy observation; no shortest path, VI value or
evaluation score is provided to the learner.

Training uses ε-greedy actions and uniform tie-breaking among equally maximal
values. Each run has its own Q table, visits, state and random key. Independent
runs execute in parallel, without sharing transitions or values. Warmup and a
short numerical reference check are discarded before every learner is reset.
The single retained training implementation uses compiled JAX on CUDA; the
short CPU reference is a verification check, not an alternative training entry.

Q=0 is used in every published baseline. A zero table receiving only zero
rewards remains zero. This explains why sparse reward discovery is essential;
small TD errors by themselves do not establish successful learning.

## 4. Experimental protocol

| Lake | States | Q entries | Holes | Episode cutoff | Interactions/run | α | ε decay ends at |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
{protocol_table}

All runs use γ=.99 and ε decreasing linearly from 1 to .05, then held at .05.
The 8×8 thresholds are absolute: α changes at 37,500 steps and ε reaches .05 at
25,000, including when the total budget is 100,000. A step is one environment
interaction and one Q update; a completed episode can contain many steps. The
representative 8×8 run completes {rep["completed_training_episodes"]:,} episodes.

Main cases use map seeds 2018/2019/2020 and training seeds 11/22/33: nine learned
policies, not nine maps and not one policy transferred between maps. A second
group uses maps 2021/2022/2023 and seeds 44/55/66, retraining from scratch. These
maps have been inspected during development, so this is a sensitivity check,
not a pristine held-out generalization benchmark.

Presets were informed by exploratory development before the retained runs.
Search cost is not included in the per-run interaction budget. We make no claim
that these are globally optimal parameters or minimum sufficient budgets.
Different budgets, schedules and cutoffs prevent attributing differences only
to grid size. All retained training results use the same implementation and
runtime; historical implementation variants are outside the active project.

At saved checkpoints, we freeze Q, remove ε exploration and propagate state
probability mass through H steps to evaluate policy success, failure, truncation,
discounted return and episode length. Environmental slipping remains. This
analytic model evaluation never changes Q. Final policies also have 5,000
independent Monte Carlo episodes; sampled fractions fluctuate around analytic
probabilities. Every configuration and source hash is retained with the run.

## 5. Numerical results

| Lake | Interactions/run | Mean exact success | Uniform random | Additional-map mean |
| --- | ---: | ---: | ---: | ---: |
{results_table}

![Learning curves](../figure/learning_curves.png)

All aggregate means use nine runs. Curve shading is the range of three map
means after averaging training seeds within each map; it is not a confidence
interval treating nine runs as nine independent maps. The final recorded
checkpoint is used, rather than selecting each run's best intermediate score.

![Policy quality](../figure/policy_quality.png)

### 5.1 The 8×8 case

Within the same retained trajectories, mean success changes from
{percentage(at50)} at 50,000 steps to {percentage(main(8)["mean_success"])} at 100,000.
The predesignated map2018/seed11 policy has {percentage(rep["success_probability"])}
analytic success and {percentage(rep["monte_carlo_success"])} success in 5,000
independent rollouts. The nine-run mean and representative example are different
statistics. The minimum main-run score is {percentage(main(8)["minimum_success"])};
a mean above 95% does not mean every run exceeds 95%.

![Early action values](../figure/qmap_8x8_early.png)

![Later action values](../figure/qmap_8x8_late.png)

Q-map numbers show max Q, not policy success probability. Dots mark tied actions;
hatched cells were unvisited. Terminal values remain zero because reward is paid
on entering G. Values and actions can change after useful behavior appears; a
similar-looking map does not prove exact convergence with a constant late rate.

![Exploratory behavior](../figure/training_return_8x8.png)

The last figure is a trailing 100-completed-episode average of training reward,
inspired by the plotting style of [the Sesen.ai tutorial](https://sesen.ai/blog/q-learning-frozen-lake-from-scratch).
It includes ε exploration and is not frozen-policy success. The first point ends
at episode100. No monotonic smoothing or fabricated intermediate measurements
are used. The tutorial's 4×4 setting and 20,000-episode budget are not our protocol.

### 5.2 The 16×16 limitation

The nine main runs receive **{main_goals} goal rewards in 4,500,000 interactions**,
with **{zero_tables} final zero Q tables**. The baseline's final score remains close
to uniform random choice. This supports a finite-budget reward-discovery and
value-propagation limitation. It does not isolate exploration, learning rate,
map topology and budget as separate causal factors.

![16×16 diagnostics](../figure/diagnostics_16x16.png)

The table contains only 1,024 scalar action values, about 8 KiB in float64. Storage
capacity is not the demonstrated bottleneck. A transition with a goal reward
updates one action value once; enough relevant subsequent samples are needed
to make that information useful at the start. Repeated starts also concentrate
visits near the initial region. The complete visit arrays preserve that evidence.

Rare-event sampling also needs care: a finite evaluation can observe no
successes even when the model assigns a small nonzero success probability.
The analytic probabilities and sampled fractions are retained separately.

## 6. Measured runtime and reproducibility

| Lake | Parallel learners | Synchronized training compute (s) | Training, transfer, logs, snapshots (s) | Container wall time (s) |
| --- | ---: | ---: | ---: | ---: |
{timing_table}

These are total batch times for 18 independent learners (nine main and nine
additional-map cases), not single-learner latency divided by 18. Device work is
synchronized before timers stop. Compilation, discarded warmup and the numerical
reference check are recorded separately in each manifest. Container wall time
includes startup and runtime initialization; plotting and host evaluation are
outside training compute. These observations do not establish a CPU/GPU speedup.

The recorded runtime uses an RTX4090D, JAX0.5.3, float64 values and a non-root
host UID/GID. Raw manifests contain the exact versions and hardware. The runtime
builder can reuse the local dependency image or start from a public Python base.
Both paths synchronize the `train` dependency group from `uv.lock`; the new
environment workflow does not rewrite the recorded historical runs. The
repository mount is read-only during training.

Maps, configurations, probability arrays, source snapshots, random keys, complete
final state, episode logs and evaluation samples remain under `results/baseline`.
The [catalog](../results/catalog.json) identifies the published baseline directories.
The read-only viewer exposes every representative checkpoint and all four Q
values per selected cell. Baseline verification checks source/input hashes,
counts, schedules, terminal masks, returns, sampled evaluation and Q exports.

## 7. Next experiment: optimistic initialization on 16×16

The baseline stays Q₀=0. The proposed configuration is
[`config/optimistic_16.json`](../config/optimistic_16.json), inheriting the full
16×16 baseline and changing only its name, variant and nonterminal initial value
to one. Terminal Q stays zero. Because an episode pays at most one +1 reward,
one is a valid optimistic upper bound on discounted return. The exploration
idea follows Sutton and Barto, §2.6 [2]; it is not an algorithmic novelty claim.

{control_status}

Use the commands in
the README, verify both runs, then compare paired maps/seeds. The comparison
rejects changed dynamics, rewards, cutoffs, schedules, budgets and core learner
code. Report both success and discounted return, not only the metric that improves.

If rare successful transitions are discovered, uniform replay and then prioritized
replay [4] can test sample reuse. Replay cannot invent rewards never observed;
replaying zero-reward/zero-value samples alone may leave an uninformative table.
If a neural approximation is explored later, compare DQN [5] with a tabular-replay
baseline at the same interaction budget, and also record replayed samples, update
counts, compute and wall time. A neural network is not an established solution
to this small-table exploration problem. Replay and DQN are not implemented here.

## 8. Conclusions

The retained implementation learns useful policies in the smaller task examples,
while the 16×16 zero-initialized preset receives little informative reward and
remains ineffective under its budget. The additional-map group reveals sensitivity
to topology and training randomness. Results support learning under particular
conditions, not guaranteed convergence, unseen-map transfer or a universal
sample count. The next step is a matched initialization control, with all original
baseline evidence preserved.

## References

[1] C. J. C. H. Watkins and P. Dayan, “Q-learning,” *Machine Learning*, vol. 8,
pp. 279–292, 1992. [DOI](https://doi.org/10.1007/BF00992698).

[2] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*,
2nd ed., MIT Press, 2018. Ch. 3, §2.6 and §6.5.
[Book](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).

[3] Farama Foundation, [FrozenLake documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/), accessed Nov. 16, 2026.

[4] T. Schaul, J. Quan, I. Antonoglou and D. Silver, “Prioritized experience
replay,” ICLR, 2016, §3.2–3.4. [Paper PDF](https://arxiv.org/pdf/1511.05952) ·
[Local PDF](../references/prioritized_experience_replay_2016.pdf).

[5] V. Mnih et al., “Human-level control through deep reinforcement learning,”
*Nature*, vol. 518, pp. 529–533, 2015. Methods: Training algorithm for deep Q-networks.
[DOI](https://doi.org/10.1038/nature14236).

We also referred to [Stanford CS234: Reinforcement Learning, Winter2026](https://web.stanford.edu/class/archive/cs/cs234/cs234.1264/modules.html)
for the MDP and Q-learning background. All reported numerical performance is
from the retained local experiments, not from these references.
'''
    # Preserve readable author/year and number spacing in generated English.
    for a, b in [("map2018", "map 2018"), ("/seed11", "/seed 11"), ("episode100", "episode 100"),
                 ("RTX4090D", "RTX 4090 D"), ("JAX0.5.3", "JAX 0.5.3"), ("Winter2026", "Winter 2026")]:
        report = report.replace(a, b)
    (root / "report/report.md").write_text(report)
    slides = f'''# Markov Decision Process (MDP) in Reinforcement Learning
# 强化学习中的马尔可夫决策过程

Q-Learning for Stochastic FrozenLake Navigation

---

## 1. Introduction / 问题介绍

An agent must reach a goal while avoiding holes on slippery ice. We model this
as an MDP and investigate whether Q-learning learns a useful policy from
interaction in 8×8, 12×12 and 16×16 environments.

智能体需要在会打滑的冰面上避开冰洞、到达终点。我们将任务建模为MDP，使用Q-learning研究智能体能否通过交互学到有效策略，并准备8×8、12×12、16×16三类环境。

![Task](../figure/problem_8x8.png)

Reference: [FrozenLake specification](https://gymnasium.farama.org/environments/toy_text/frozen_lake/).

---

## 2. MDP and Q-Learning / 模型与算法

- State: current cell. Actions: left, down, right, up.
- Intended and two perpendicular directions each have probability 1/3; boundary moves stay in place.
- Entering the goal pays +1, otherwise 0. Goal/hole terminate. Baseline Q₀=0.
- ε-greedy exploration; one observed transition updates one state–action value.

状态为当前位置，动作是四个方向。预期方向及两个垂直方向各1/3；越界留原地。进入终点奖励1，其他0；终点/冰洞结束回合。主实验Q₀=0，使用ε-greedy探索。

![Q-learning update](../figure/q_learning_update.png)

![Bellman optimality](../figure/bellman_optimality.png)

Watkins & Dayan (1992), §2. [DOI](https://doi.org/10.1007/BF00992698). We also consulted [Stanford CS234](https://web.stanford.edu/class/archive/cs/cs234/cs234.1264/modules.html).

---

## 3. Our Experiment / 实验过程

| Setting / 设置 | Value / 数值 |
| --- | --- |
| Initialization / 初始化 | Q₀=0 |
| Training budget / 预算 | 100,000 interactions per run |
| Learning rate / 学习率 | α=.7 until 37,500 interactions, then .1 |
| Discount / 折扣 | γ=.99 |
| Exploration / 探索 | ε: 1.00 to .05 over 25,000 interactions |

![Early values](../figure/qmap_8x8_early.png)

![Later values](../figure/qmap_8x8_late.png)

![Policy success](../figure/learning_curve_8x8.png)

Nine-run mean exact success: **{percentage(main(8)["mean_success"])}** (3 maps × 3 training seeds).
Predesignated policy: **{percentage(rep["success_probability"])}** exact success;
**{percentage(rep["monte_carlo_success"])}** over 5,000 independent evaluation episodes.

九次运行平均精确成功概率为{percentage(main(8)["mean_success"])}。图上数字为最大Q值，不是成功概率；箭头表示选择动作，实际移动仍会打滑。

---

## 4. Policy Assessment / 策略效果

![Results](../figure/policy_quality.png)

| Lake | Interactions/run | Mean exact success | Uniform random | Additional-map check |
| --- | ---: | ---: | ---: | ---: |
{results_table}

Different budgets and schedules are disclosed; these are task examples, not a
pure grid-size comparison. Policies are retrained on each map.

各环境使用不同预算与超参数，重点是策略是否有效，不将差异完全归因于地图大小；每张地图独立训练。

---

## 5. Limitations and Future Work / 局限与后续工作

- 16×16 receives **{main_goals} goal rewards across 4.5 million interactions**; **{zero_tables}/9** final Q tables stay zero.
- Sparse rewards and uneven visits restrict value propagation under this preset.
- The report's next experiment is matched optimistic initialization; the main baseline remains Q₀=0.
- Replay and DQN are later options, not implemented or established solutions here.

16×16的九次主训练共450万交互步，仅获得{main_goals}次终点奖励，其中{zero_tables}次最终Q表全零。后续report将用匹配对照检验乐观初始化，再讨论回放和DQN；不预先保证能解决问题。

References: Sutton & Barto (2018), §2.6; [Schaul et al. (2016)](https://arxiv.org/pdf/1511.05952); [Mnih et al. (2015)](https://doi.org/10.1038/nature14236).
'''
    (root / "report/presentation.md").write_text(slides)
    return data
