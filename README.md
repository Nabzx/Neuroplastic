# Neuroplastic Collective Intelligence

**Investigating whether neuroplastic communication mechanisms inspired by
computational neuroscience can improve emergent coordination and functional
specialisation in collective AI systems.**

> **Status: research scaffolding.** This repository defines the architecture,
> interfaces, configuration system and experiment plan. The optimisation loop and
> the learned components are intentionally *not* implemented yet — every deferred
> piece is a clearly-marked placeholder that raises `NotImplementedError` with a
> pointer to [`docs/experiment_plan.md`](docs/experiment_plan.md). The structural
> pieces that don't need training (config loading, the interaction graph,
> graph-theoretic and information-theoretic metrics, plasticity update maths) are
> already functional and tested.

---

## Motivation

Biological neural systems do not learn over fixed wiring. Synaptic strengths
change as a function of local activity — *cells that fire together, wire
together* — and this **plasticity**, together with **homeostatic** and
**neuromodulatory** control, is central to how brains self-organise, allocate
representational resources, and develop specialised sub-circuits.

Most multi-agent AI systems, by contrast, communicate over **static** channels:
who-talks-to-whom is fixed, and the "strength" of a connection is not itself an
adaptive variable shaped by the ongoing interaction. This project asks a simple
question:

> If we let the *communication topology itself* be plastic — rewiring and
> reweighting inter-agent connections using Hebbian-inspired, activity-dependent
> rules — do collectives of agents coordinate better and differentiate into
> functional roles more readily than with conventional fixed communication?

## Research question & hypotheses

**RQ.** Do neuroplastic communication mechanisms improve emergent coordination
and functional specialisation in cooperative multi-agent systems, relative to
static-communication baselines?

- **H1 (plasticity helps).** Hebbian-inspired plasticity on the communication
  weights improves coordination over an identical architecture with plasticity
  disabled.
- **H2 (dynamics help).** A *dynamic* interaction topology outperforms a *static*
  one under matched plasticity.
- **H3 (specialisation emerges).** Neuroplastic communication yields greater
  functional specialisation (more distinct, stable agent roles), measurable with
  graph- and information-theoretic tools.

The full protocol — variables, conditions, metrics, statistics and milestones —
is in [`docs/experiment_plan.md`](docs/experiment_plan.md).

## Approach

A cooperative multi-agent framework with three interacting subsystems:

1. **Adaptive graph-based communication.** Agents are nodes in a directed,
   weighted interaction graph. A *topology* strategy decides who communicates
   each step (fully-connected · k-nearest · static · learned/adaptive); a
   *protocol* decides how received messages are aggregated (mean · attention ·
   GNN).
2. **Hebbian-inspired plasticity.** The graph's edge weights are adapted online
   from communication activity, with optional **neuromodulation** (reward-gated /
   three-factor) and **homeostasis** (synaptic scaling) to keep learning stable.
3. **Emergent-behaviour evaluation.** The discovered structure and information
   flow are characterised with **graph-theoretic** (density, modularity,
   clustering, path length, centralisation) and **information-theoretic**
   (entropy, mutual information, transfer entropy) measures, alongside task-level
   coordination and role-specialisation metrics.

See [`docs/architecture.md`](docs/architecture.md) for how the pieces fit
together and where to extend them.

## Repository layout

```
neuroplastic/
├── agents/          # policy networks + the agent interface driven by training
├── environments/    # cooperative PettingZoo environments (registry + benchmarks)
├── communication/   # interaction graph, topologies, protocols, message routing
├── plasticity/      # Hebbian/Oja rules, neuromodulation, homeostasis
├── training/        # trainer, MARL algorithm interface, rollout, CLI (loop deferred)
├── evaluation/      # graph-theoretic / information-theoretic / coordination metrics
├── analysis/        # cross-run interpretation: topology evolution, specialisation, stats
├── visualisation/   # interaction-graph and training-dynamics plots
├── configs/         # config system: dataclass schema + YAML (with ablations)
├── docs/            # research motivation, experiment plan, architecture
├── core/            # shared primitives: registry, seeding, types  (supporting)
└── tests/           # unit tests for the functional pieces         (supporting)
```

`core/` and `tests/` are small supporting packages added alongside the ten
requested domain directories; every domain package is a top-level, importable
Python package, selected by name from config via a lightweight registry.

## Installation

Requires Python ≥ 3.10.

```bash
# Core install (config + graph + metrics tooling; no heavy RL deps):
pip install -e .

# Full reinforcement-learning stack (PyTorch + PettingZoo) for experiments:
pip install -e ".[rl]"

# Optional extras:
pip install -e ".[viz]"        # matplotlib plotting
pip install -e ".[analysis]"   # scipy/pandas statistics
pip install -e ".[dev]"        # pytest, ruff, black, mypy
```

## Quickstart

Validate a config and print the fully-resolved component wiring — this works with
just the core install, no PyTorch required:

```bash
nci --config configs/default.yaml --dry-run
nci --config configs/ablations/no_plasticity.yaml -o seed=1 --dry-run
```

Use a functional metric on a synthetic interaction graph:

```python
from itertools import permutations

from communication.graph import InteractionGraph
from evaluation.graph_metrics import compute_graph_metrics

g = InteractionGraph(["agent_0", "agent_1", "agent_2"])
g.set_edges(permutations(g.agents, 2))            # fully connected
print(compute_graph_metrics(g, ["density", "modularity", "global_clustering"]))
```

Load and override configuration:

```python
from configs import load_config

cfg = load_config("configs/default.yaml", overrides=["communication.topology=k_nearest"])
print(cfg.plasticity.rule, cfg.communication.topology)
```

### Train the baselines

Three fixed-communication baselines (no plasticity), sharing an identical
network so they are directly comparable — the reference points for the eventual
neuroplastic model:

```bash
nci --config configs/baselines/no_comm.yaml          # 1. independent agents
nci --config configs/baselines/fully_connected.yaml  # 2. everyone-to-everyone
nci --config configs/baselines/sparse.yaml           # 3. fixed ring graph
```

Reward and episode-length curves stream to the console and to
`runs/<name>/metrics.csv`. Shrink a run for a quick smoke check with overrides:

```bash
nci --config configs/baselines/fully_connected.yaml \
    -o training.total_steps=20000 -o training.rollout_length=500
```

The learner is parameter-shared PPO; the three settings differ only in a fixed,
differentiable communication mask (`none` / dense / ring), so messages are
learned end-to-end while the *topology* stays fixed. See
[`docs/experiment_plan.md`](docs/experiment_plan.md) §7 for the model details.

### Adaptive communication

The adaptive setting learns a **weighted edge matrix** (attention over the graph)
instead of a fixed uniform mask — messages are still learned, but *who is
listened to* now adapts to the agents' state:

```bash
nci --config configs/adaptive.yaml
```

Graph statistics (edge density, weight entropy, effective degree) stream to the
console and `runs/<name>/metrics.csv`. The learned graph exports to NetworkX:

```python
from configs import load_config
from training.trainer import Trainer

trainer = Trainer(load_config("configs/adaptive.yaml"))
trainer.train()
graph = trainer.export_communication_graph()   # InteractionGraph (weighted)
nx_graph = graph.to_networkx()                  # networkx.DiGraph
```

This is dynamic *weighted* communication only — the edge weights are learned by
the policy loss. Hebbian plasticity (updating a persistent weight matrix from
activity) is a later milestone and reuses the same edge-matrix representation.

## Tech stack

Python · PyTorch · PettingZoo · NetworkX · NumPy
(plus PyYAML for configs; SciPy/Matplotlib as optional analysis/plotting extras).

## Roadmap

- [x] Architecture, interfaces, config system, experiment plan (this scaffold)
- [x] Functional interaction graph + graph/information metrics + plasticity maths
- [x] Environment layer (PettingZoo/SuperSuit) with reproducible seeding
- [x] Learning baseline: shared-parameter PPO + fixed communication (none / full / sparse)
- [x] Adaptive graph communication: learned attention edge-weights, graph stats, NetworkX export
- [ ] Recurrent policies + multi-round GNN protocols; learned edge-*existence* gating
- [ ] Hebbian plasticity coupling (the neuroplastic model)
- [ ] Full evaluation over recorded rollouts + role-clustering
- [ ] Benchmark sweep across conditions and seeds; write-up

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
