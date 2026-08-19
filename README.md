# Neuroplastic — Biological Plasticity Against Loss of Plasticity

**Can biologically-inspired neuroplasticity mechanisms keep artificial neural
networks learning, where standard ones stop?**

Deep networks trained on a *sequence* of tasks gradually **lose the ability to
learn** — neurons go dormant, representational rank collapses, weights drift
(Dohare & Sutton et al., *Nature* 2024; Lyle et al. 2023; Sokar et al. 2023).
Biological brains stay plastic for life through **homeostatic** and **structural**
plasticity. This project replicates those mechanisms in artificial networks and
asks, rigorously, whether — and *how* — they preserve plasticity.

> **Status: Phase 1 complete.** Loss of plasticity is reproduced on a small,
> CPU-friendly synthetic benchmark (Gate 1 ✅). The biological mechanisms and the
> multi-seed study follow. See [`docs/preregistration.md`](docs/preregistration.md)
> for the fixed hypotheses and protocol.

## The phenomenon (reproduced)

Vanilla SGD over 300 permuted-regression tasks measurably loses plasticity —
per-task fitting error rises, dormant units triple (0.10 → 0.33), and effective
rank halves (17 → 9):

```bash
python scripts/reproduce_plasticity_loss.py --seeds 3 --num-tasks 300
```

## Approach

- **Benchmark** ([`data/`](data)): permuted-input regression — a fixed nonlinear
  teacher with a fresh input permutation per task; synthetic, reproducible, no
  downloads. Per-task *late* loss measures the learner's *current* plasticity.
- **Diagnostics** ([`diagnostics/`](diagnostics)): dormant-unit fraction,
  effective rank, weight magnitude — the standard loss-of-plasticity signatures,
  which also let us attribute each mechanism to the failure mode it repairs.
- **Mechanisms** ([`mechanisms/`](mechanisms), Phase 2): homeostatic synaptic
  scaling and utility-gated structural plasticity (pruning + neurogenesis), vs
  baselines (L2, shrink-and-perturb, Continual Backprop, ReDo).

## Repository layout

```
neuroplastic/
├── data/          # continual task streams
├── models/        # instrumented networks (expose hidden activations)
├── diagnostics/   # plasticity metrics
├── mechanisms/    # biological plasticity mechanisms + baselines (Phase 2)
├── training/      # online continual training loop
├── core/          # registry, seeding, types (shared primitives)
├── scripts/       # runnable experiments
├── docs/          # preregistration, plan, results
└── tests/
```

## Prior work in this repo

The earlier multi-agent *neuroplastic communication* project (MARL) is archived at
tag `marl-communication-v1` and branch `archive/marl-communication` — recover with
`git checkout marl-communication-v1`.

## Install

```bash
pip install -e ".[viz,dev]"   # torch, numpy, matplotlib, pytest
```

## License

MIT — see [`LICENSE`](LICENSE).
