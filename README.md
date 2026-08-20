# Neuroplastic — Biological Plasticity Against Loss of Plasticity

**Can biologically-inspired neuroplasticity mechanisms keep artificial neural
networks learning, where standard ones stop?**

Deep networks trained on a *sequence* of tasks gradually **lose the ability to
learn** — neurons go dormant, representational rank collapses, weights drift
(Dohare & Sutton et al., *Nature* 2024; Lyle et al. 2023; Sokar et al. 2023).
Biological brains stay plastic for life through **homeostatic** and **structural**
plasticity. This project replicates those mechanisms in artificial networks and
asks, rigorously, whether — and *how* — they preserve plasticity.

> **Status: study complete on two benchmarks — a positive result.**
> Biologically-inspired **homeostatic synaptic scaling** significantly exceeds the
> state-of-the-art remedy (Continual Backprop) on *both* a synthetic
> permuted-regression task and **Continual Permuted-MNIST**. Structural and
> homeostatic mechanisms repair complementary failure modes. Reported honestly
> (the best-of-ours method is benchmark-dependent; a strong baseline is competitive
> on MNIST). See [`docs/study_results.md`](docs/study_results.md), the write-up in
> [`docs/paper_draft.md`](docs/paper_draft.md), and the preregistered hypotheses in
> [`docs/preregistration.md`](docs/preregistration.md).

## Headline result — homeostatic scaling beats SOTA on both benchmarks

| | vanilla | Continual Backprop (SOTA) | **homeostatic (ours)** |
|---|---|---|---|
| synthetic — late loss ↓ | 0.82 | 0.71 | **0.58** (p=0.02 vs SOTA) |
| Permuted-MNIST — accuracy ↑ | 0.59 | 0.70 | **0.76** (p<10⁻³ vs SOTA) |

Reproduce:

```bash
python scripts/run_study.py --seeds 10 --num-tasks 250 --output results/study
python scripts/run_study.py --benchmark permuted_mnist --seeds 8 --num-tasks 300 \
    --task-length 800 --batch-size 16 --hidden-dim 32 --lr 0.05 --output results/study_mnist
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
