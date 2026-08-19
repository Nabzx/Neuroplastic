# Neuroplastic — Biological Plasticity Against Loss of Plasticity

**Can biologically-inspired neuroplasticity mechanisms keep artificial neural
networks learning, where standard ones stop?**

Deep networks trained on a *sequence* of tasks gradually **lose the ability to
learn** — neurons go dormant, representational rank collapses, weights drift
(Dohare & Sutton et al., *Nature* 2024; Lyle et al. 2023; Sokar et al. 2023).
Biological brains stay plastic for life through **homeostatic** and **structural**
plasticity. This project replicates those mechanisms in artificial networks and
asks, rigorously, whether — and *how* — they preserve plasticity.

> **Status: main study complete — a positive result.** Across 10 seeds,
> biologically-inspired **homeostatic synaptic scaling** retains significantly more
> plasticity than a vanilla network *and significantly exceeds the state-of-the-art
> remedy* (Continual Backprop); **combining** it with structural plasticity is best.
> Structural and homeostatic mechanisms repair complementary failure modes.
> See [`docs/study_results.md`](docs/study_results.md) and the write-up in
> [`docs/paper_draft.md`](docs/paper_draft.md); hypotheses were fixed in advance in
> [`docs/preregistration.md`](docs/preregistration.md).

## Headline result (10 seeds)

| method | final per-task loss ↓ | vs vanilla | vs SOTA (Continual Backprop) |
|---|---|---|---|
| vanilla | 0.82 (loses plasticity) | — | — |
| Continual Backprop (SOTA) | 0.71 | p=0.001 | — |
| **homeostatic (ours)** | **0.58** | **p=2×10⁻⁴** | **p=0.02 (beats SOTA)** |
| **combined (ours)** | **0.55** | **p=1×10⁻⁴** | **p=3×10⁻⁴ (beats SOTA)** |

Reproduce the full study:

```bash
python scripts/run_study.py --seeds 10 --num-tasks 250 --output results/study
```

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
