# Metaplasticity — follow-on plan (draft)

Extension to *Biological Plasticity Against Loss of Plasticity*. The base study is
complete: **homeostatic synaptic scaling** beats the SOTA remedy (Continual Backprop,
CBP) on both benchmarks, and **structural plasticity** (prune + neurogenesis) repairs
a complementary failure mode. This document plans the third biological mechanism —
**metaplasticity** — as a preregistration-style spec to lock in *before* any run.

Status: **design only, no experiments run.** Compute is gated on explicit go-ahead.

---

## 1. Scientific motivation & the novel angle

**Metaplasticity** (Abraham & Bear, 1996) is the *plasticity of plasticity*: a
synapse's history of activity modulates its *future capacity to change*, rather than
its current weight. The classic instantiation is the **BCM sliding threshold**
(Bienenstock–Cooper–Munro, 1982): a unit's recent activity sets a modification
threshold θ that shifts how easily it potentiates or depresses next.

In machine learning, metaplasticity has so far been used almost exclusively for the
**opposite** problem to ours — *catastrophic forgetting*. Methods like synaptic
metaplasticity in binarized nets (Laborieux et al., 2021), Synaptic Intelligence
(Zenke et al., 2017), and EWC (Kirkpatrick et al., 2017) **raise** a weight's
consolidation to **protect** important weights and *reduce* their plasticity.

Our problem is **loss of plasticity** — networks becoming *unable to learn* new
tasks: units go dormant, effective rank collapses, weights saturate. This asks for
the **inverse** metaplastic pressure: *increase* the effective plasticity of
saturated / dormant units to reawaken them. The loss-of-plasticity literature's
remedies for this (CBP, ReDo) are **discrete** — they detect low-utility units and
**hard-reset** them. 

> **The novel claim.** Metaplasticity gives a **smooth, per-synapse, optimiser-level**
> alternative to discrete unit replacement: modulate each weight's *effective learning
> rate* by its unit's activity history — boost the dormant back toward their active
> regime, gently damp the saturated — without ever resetting a unit. We ask whether
> this smooth reawakening **matches or beats** discrete structural replacement (CBP /
> ReDo / our structural mechanism) and whether it **complements** the homeostatic
> winner.

This reframes an idea the field uses for *stability* (forgetting) as a tool for
*plasticity* — and directly contrasts smooth vs discrete plasticity repair. That
contrast is the paper's contribution.

---

## 2. Hypotheses (preregistered)

Primary metric: per-task **late loss** (regression) / **accuracy** (MNIST), same
windowing and stats machinery as the base study (bootstrap CI, permutation test,
Cohen's d, Holm–Bonferroni). Directional predictions fixed before running:

- **HM1 — maintenance.** Metaplastic LR modulation preserves plasticity vs vanilla
  (lower final late loss / higher accuracy), significantly.
- **HM2 — attribution.** It works *by reawakening units*: dormant-unit fraction
  falls and effective rank rises relative to vanilla — the smooth analog of
  structural neurogenesis, achieved without discrete resets.
- **HM3 — competitiveness.** It matches or beats the discrete SOTA (CBP) and our own
  structural mechanism, at equal or lower intervention (no unit resets).
- **HM4 — synergy.** Metaplasticity **+** homeostatic scaling ≥ homeostatic alone —
  activity-gated learning rates and set-point rescaling target different failure
  modes (trainability vs weight drift), so they should stack.
- **HM5 — direction matters (honest negative control).** The *forgetting*-oriented
  variant (consolidation metaplasticity, which *lowers* plasticity of settled
  weights) does **not** help — and may *hurt* — loss of plasticity. Confirming HM5
  is as valuable as HM1: it shows the mechanism's benefit is specifically its
  *reawakening* direction, not metaplastic bookkeeping per se.

Any of these may come back negative or mixed; all outcomes are reportable.

---

## 3. Mechanism design

Two variants, both reusing the existing per-unit activity/utility EMA infrastructure
(`observe(activations)`), both **optimiser-agnostic** by construction (see §4).

### 3a. `MetaplasticLR` — activity-history modulation (HERO)

Per hidden unit *i*, maintain a slow EMA of its activity `a_i` (mean post-ReLU
output magnitude), with set-point `a*` taken from the initial-epoch mean (mirrors how
homeostatic scaling anchors to the initial norm). Per-unit metaplastic gain:

```
g_i = clip( (a* / (a_i + eps)) ** beta ,  g_min ,  g_max )
```

- `a_i << a*` (dormant / saturated-off) → `g_i > 1` → **boost** its learning rate →
  gradient can move it back into the active regime (smooth reawakening).
- `a_i >> a*` (hyperactive) → `g_i < 1` → **damp** → protect against runaway drift.

The gain scales the **realized weight update** of unit *i*'s incoming synapses (row
*i* of the layer weight + its bias). The readout/head is left unmodulated by default.
Config: `beta` (strength, default 1.0), `g_min`/`g_max` (bounds, default 0.5 / 5.0),
`ema_decay`, `scope ∈ {incoming, incoming+outgoing}` (default incoming).

*Biological reading:* a direct BCM sliding-threshold rule — activity history sets each
unit's plasticity, pushing under-active units to change more.

### 3b. `ConsolidationMetaplasticity` — weight-history modulation (FOIL for HM5)

Per **weight** `w_ij`, maintain a consolidation state `c_ij` that grows when updates
are sign-consistent and the weight is large (the Laborieux/SI "settled" signal);
effective LR scaled by `1 / (1 + lambda * c_ij)`. This is the forgetting-oriented
direction: it *protects* consolidated weights. Included **only** as the HM5 control —
we predict it does not cure loss of plasticity.

### 3c. Combinations

- `metaplastic + homeostatic` (HM4) via the existing `Combined` composition path,
  generalized to take an arbitrary list of sub-mechanisms.
- `metaplastic + structural` as an ablation (do smooth and discrete reawakening
  stack, or are they redundant?).

---

## 4. Framework change (minimal, backward-compatible)

Metaplasticity modulates the *learning rate per parameter*, which the current
`after_optimizer_step` hook cannot do alone (the step has already applied a uniform
LR). Clean, optimiser-agnostic solution: **rescale the realized update**.

Add **one** optional no-op hook, `before_optimizer_step(model, step_index)`, called in
`training/continual.py` between `loss.backward()` and `optimizer.step()`:

```
zero_grad(); loss.backward()
mechanism.before_optimizer_step(model, step_index)   # NEW: snapshot W, B (clone)
optimizer.step()
mechanism.observe(activations)                        # updates activity EMA
mechanism.after_optimizer_step(model, step_index)     # W ← W_snap + g ⊙ (W − W_snap)
```

In `after_optimizer_step`, per layer: `delta = W_now − W_snap`, then
`W_now ← W_snap + g[:,None] * delta` under `torch.no_grad()` (bias likewise). Because
it rescales the **applied delta**, it yields an exact per-unit effective learning rate
for **any optimiser** — SGD *and* Adam — with no interaction with Adam's internal
normalization. This is a nice property to state in the paper and makes the A2-style
optimiser robustness sweep especially clean.

Backward compatibility: `Mechanism.before_optimizer_step` defaults to a no-op, so
every existing mechanism and all 23 tests are unaffected. `MetaplasticLR` sets
`requires_activations = True`.

Registry names: `metaplastic` (hero), `metaplastic_consolidation` (foil). Added to
`METHODS` for the studies.

---

## 5. Experiments (reuse existing machinery — no new harness)

Same two benchmarks, seeds, and stats as the base study; new methods slot into
`run_study.py` / `run_sensitivity.py` / `run_robustness.py` unchanged.

| Stage | What | New methods vs baselines | Purpose |
|---|---|---|---|
| **E1** | Synthetic permuted-regression, 10 seeds, 250 tasks | `metaplastic`, `metaplastic_consolidation` vs vanilla / CBP / homeostatic / structural | HM1, HM2, HM3, HM5 |
| **E2** | `metaplastic + homeostatic` on E1 config | vs homeostatic alone | HM4 (synergy) |
| **E3** | Permuted-MNIST, 8 seeds, 300 tasks (hid 32, lr 0.05) | same method set | confirm HM1/HM3 on classification |
| **E4** | `beta` sensitivity sweep (A3-style): 0.25, 0.5, 1, 2, 4 | vs vanilla / CBP reference lines | robustness to strength (not a lucky β) |
| **E5** | Optimiser × width sweep (A2-style): SGD/Adam × 32/128 | metaplastic vs CBP | robustness across scale/optimiser |

Diagnostics already logged (dormant fraction, effective rank, weight magnitude) give
HM2 directly. Figures reuse `visualisation/plasticity_figures.py` (the mechanism
attribution scatter is exactly the smooth-vs-discrete story).

---

## 6. Deliverables & staging (gated)

- **S1 — code + tests, NO compute.** `mechanisms/metaplasticity.py` (both variants +
  registry), the `before_optimizer_step` hook, `Combined` generalization, unit tests
  (dormant unit → gain > 1; saturated → gain < 1; delta-rescale equals a per-unit LR;
  registry/round-trip; foil lowers plasticity on a toy). Update the preregistration
  amendment with HM1–HM5. This is the "just write code / improve the paper" tier —
  runnable without your CPU go-ahead, like the A1–A4 code stage.
- **S2 — run E1–E2** (synthetic). *Requires go-ahead.*
- **S3 — run E3** (MNIST confirmation). *Requires go-ahead.*
- **S4 — run E4–E5** (sensitivity + robustness). *Requires go-ahead.*
- **S5 — paper.** New "Metaplasticity" subsection (mechanism + math), results table,
  attribution figure, honest limitations; fold into `paper_draft.md`,
  `robustness_results.md`, README.

---

## 7. Rough compute estimate (CPU, gated)

- S1: no experiments — dev + `pytest` only.
- E1+E2 (synthetic, ~10 seeds): order **10–25 min**.
- E3 (MNIST, 8 seeds × 300 tasks): the heavy one — order **30–60 min**.
- E4 (β sweep, 5×5) + E5 (4 combos × 5 seeds): order **20–40 min**.

Total run time across S2–S4 ≈ **1.5–2.5 hours** of background CPU, run in stages so
each result is checked before the next. All stages are interruptible.

---

## 8. Risks & honest failure modes

- **Metaplasticity may just re-derive homeostatic scaling.** Both use activity set-
  points. Mitigation: they act on *different objects* — homeostatic rescales weight
  *norms* toward a set-point (a stability pressure), metaplastic rescales *update
  magnitude* by activity (a learning-rate pressure). HM4 (do they stack?) and the
  attribution diagnostics test whether they are genuinely distinct.
- **Boosting dormant units may add noise / instability** (large `g_max`). E4's β sweep
  and the `g_min/g_max` bounds probe this; report if it's brittle.
- **HM3 may fail** — smooth reawakening might not match discrete resets on some
  configs (as homeostatic didn't beat CBP on the large SGD net in A2). That's a
  reportable, honest result, not a blocker.
- **Adam interaction** is handled by rescaling the realized delta (§4), but we will
  verify empirically in E5 rather than assume it.

---

## 9. References (to add to `references.bib`)

- Abraham & Bear (1996). *Metaplasticity: the plasticity of synaptic plasticity.* TINS.
- Bienenstock, Cooper & Munro (1982). *Theory for the development of neuron
  selectivity (BCM).* J. Neuroscience.
- Laborieux et al. (2021). *Synaptic metaplasticity in binarized neural networks.*
  Nature Communications.
- Zenke, Poole & Ganguli (2017). *Continual learning through synaptic intelligence.* ICML.
- Kirkpatrick et al. (2017). *Overcoming catastrophic forgetting (EWC).* PNAS.
