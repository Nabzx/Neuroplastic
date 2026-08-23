# Preregistration — Biological Neuroplasticity Mechanisms Against Loss of Plasticity

*Registered before running the main study. Hypotheses, benchmarks, mechanisms,
metrics and analysis are fixed here; results are reported against these regardless
of outcome. Negative or mixed results are acceptable and will be reported.*

## Background

Deep networks trained on a *sequence* of tasks lose the ability to keep learning —
**loss of plasticity**: neurons go dormant, representational (effective) rank
collapses, and weight magnitudes drift (Dohare & Sutton et al., *Nature* 2024;
Lyle et al. 2023; Sokar et al. 2023). Biological brains stay plastic across a
lifetime via **homeostatic plasticity** (synaptic scaling; Turrigiano),
**structural plasticity** (synaptic pruning + neurogenesis), and
**metaplasticity** (Abraham & Bear). This project asks whether replicating those
biological mechanisms in artificial networks preserves plasticity, and — as a
mechanistic study — *which failure mode each mechanism repairs*.

## Research questions & hypotheses

- **RQ.** Do biologically-inspired neuroplasticity mechanisms prevent loss of
  plasticity in continually-trained networks, and by what mechanism?
- **H1 (preservation).** A network with homeostatic synaptic scaling and/or
  utility-gated structural plasticity (pruning + neurogenesis) retains per-task
  fitting ability over a long task sequence, where a vanilla network does not.
- **H2 (mechanism attribution).** Each mechanism improves a *specific* diagnostic:
  structural plasticity chiefly reduces the dormant-unit fraction; homeostatic
  scaling chiefly bounds weight magnitude; both help sustain effective rank.
- **H3 (competitiveness).** A principled combination matches or exceeds the
  strongest published remedy (Continual Backprop) on retained plasticity, while
  being simpler / more interpretable.

## Benchmarks (small, synthetic, CPU-reproducible)

1. **Permuted-input regression** (primary; `data/streams.py`): a fixed nonlinear
   teacher; each task permutes the input features. Per-task late loss measures the
   learner's *current* fitting ability. *(Gate 1 confirmed vanilla nets lose
   plasticity here: late-loss ratio ~1.24x, dormant fraction 0.10→0.33, effective
   rank 17→9 over 300 tasks.)*
2. *(Optional)* bit-flipping / Continual Permuted-MNIST if data access allows.

## Mechanisms

- **Baselines:** vanilla SGD; L2 regularisation; shrink-and-perturb (Ash & Adams);
  **Continual Backprop** (Dohare et al., SOTA remedy); ReDo-style dormant recycling
  (Sokar et al.).
- **Biological (ours), hero:** homeostatic synaptic scaling; utility-gated
  structural plasticity (pruning + neurogenesis); and their combination.
- **Follow-on (post-base):** metaplasticity (per-weight consolidation).

## Metrics (per task, over the sequence)

- Plasticity: **per-task late loss** (primary), area under the loss-vs-task curve.
- Diagnostics: **dormant/dead unit fraction**, **effective rank** of features,
  **mean weight magnitude**. (All in `diagnostics/plasticity.py`.)

## Protocol & analysis

- **>= 10 seeds** per condition; report mean, std, bootstrap CI.
- **Permutation tests** for each mechanism vs vanilla and vs Continual Backprop on
  the primary metric; report effect sizes, not just p-values; note multiple
  comparisons.
- Attribute mechanisms to diagnostics (H2) with correlation / ablation.
- Honest auto-generated summary; the framework reports whatever it finds.

## Falsifiable predictions

- If H1 holds: mechanism late-loss stays ~flat while vanilla climbs.
- If H2 holds: the diagnostic each mechanism targets improves selectively.
- If they fail: mechanisms do not beat vanilla / Continual Backprop — reported as
  a negative result with analysis of why.

## Amendment (added after the main study, before running these)

Robustness and rigor analyses, motivated by the main result and specified here
before being run:

- **A1 — plasticity-gain probe** (`scripts/probe_plasticity_gain.py`): re-run on an
  *independent-teacher* stream (fresh teacher per task). Prediction: if homeostatic's
  ratio < 1 is genuine plasticity it persists; if it was shared-structure
  accumulation the ratio rises toward 1. Either outcome is reported.
- **A2 — optimiser/scale robustness** (`scripts/run_robustness.py`): sweep optimiser
  (SGD/Adam) × network width. Prediction to test: homeostatic scaling continues to
  exceed Continual Backprop; if the advantage vanishes under Adam or at scale, that
  is reported.
- **A3 — hyper-parameter sensitivity** (`scripts/run_sensitivity.py`): sweep each
  mechanism's key coefficient across a grid. Prediction: the advantage over
  Continual Backprop is stable across a broad range (not a tuned point).
- **A4 — statistical rigor**: report Cohen's d effect sizes and Holm-Bonferroni
  corrected p-values (family-wise error control), and run MNIST at >= 10 seeds.

## Amendment — metaplasticity follow-on (registered before running)

The third biological mechanism, **metaplasticity** (Abraham & Bear; BCM sliding
threshold), as a *smooth, per-synapse* alternative to the discrete unit resets of
Continual Backprop / ReDo. Design is fixed in [`metaplasticity_plan.md`](metaplasticity_plan.md)
and implemented (`mechanisms/metaplasticity.py`); the mechanism modulates each unit's
*effective learning rate* by its activity history — boosting under-active units,
damping hyper-active ones — via rescaling the realised update (exact per-unit LR for
any optimiser). Same benchmarks, metrics, seeds and statistics as the main study.
Directional predictions, fixed here before any run; all outcomes reported:

- **HM1 — maintenance.** `metaplastic` preserves plasticity vs vanilla (lower final
  late loss / higher accuracy), significantly.
- **HM2 — attribution.** It works by *reawakening units*: lower dormant fraction and
  higher effective rank than vanilla — the smooth analog of structural neurogenesis,
  with no discrete resets.
- **HM3 — competitiveness.** It matches or beats the discrete SOTA (Continual
  Backprop) and our own structural mechanism.
- **HM4 — synergy.** `metaplastic_homeostatic` >= homeostatic alone (activity-gated
  learning rates and set-point rescaling target different failure modes).
- **HM5 — direction matters (negative control).** The forgetting-oriented
  `metaplastic_consolidation` (which *lowers* plasticity of settled weights) does
  **not** help — and may hurt — loss of plasticity. Confirming HM5 shows the benefit
  is specifically the *reawakening* direction, not metaplastic bookkeeping per se.

Falsifiable: if HM3 fails, smooth reawakening does not match discrete resets on some
configs (as homeostatic did not beat CBP on the large SGD net in A2) — reported
honestly. If HM5's foil *helps*, the direction hypothesis is wrong.
