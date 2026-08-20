# Biologically-Inspired Homeostatic and Structural Plasticity Mitigate Loss of Plasticity

*Working draft — a workshop-length paper. Results/abstract/conclusion are filled
from the study in `results/study/`; method and setup are final.*

**Author:** Nabil Shah

---

## Abstract

Deep networks trained continually lose the ability to learn. We ask whether
*biological* neuroplasticity mechanisms — homeostatic synaptic scaling and
utility-gated structural plasticity (pruning + neurogenesis) — prevent this, and
which failure mode each repairs. Across **two continual benchmarks** — a synthetic
permuted-regression task (10 seeds) and **Continual Permuted-MNIST** (8 seeds) —
homeostatic synaptic scaling **significantly exceeds the state-of-the-art remedy,
Continual Backprop**, on both (synthetic: final loss 0.58 vs 0.71, p = 0.02; MNIST:
retained accuracy 0.76 vs 0.70, p < 10⁻³), and is the single most consistent
method. A mechanistic analysis shows *why*: structural plasticity and homeostatic
scaling repair **complementary and near-orthogonal** failure modes — dead units
(structural drives the dormant fraction toward zero and keeps effective rank high)
versus weight ill-conditioning (homeostatic bounds weight growth) — so combining
them helps. We report the picture honestly: the *best-performing* method shifts
across benchmarks (combined on synthetic, homeostatic on MNIST) and a strong
baseline (shrink-and-perturb) is competitive on MNIST; caveats include small
networks, plain SGD, and an intriguing "improvement over time" (plasticity
ratio < 1) that warrants further scrutiny.

## 1. Introduction

Neural networks trained on a *stream* of tasks progressively lose plasticity:
their ability to fit new tasks degrades, units go dormant, the representation's
effective rank collapses, and weights drift (Dohare et al., *Nature* 2024; Lyle
et al. 2023; Sokar et al. 2023). Biological brains do not suffer this — they
remain plastic across a lifetime through **homeostatic plasticity** (synaptic
scaling; Turrigiano 2008), **structural plasticity** (synaptic pruning and
neurogenesis), and metaplasticity (Abraham & Bear 1996).

We replicate the first two mechanisms in artificial networks and study them
rigorously. Our contribution is (i) a controlled, reproducible demonstration that
biologically-inspired homeostatic and structural mechanisms mitigate loss of
plasticity, (ii) a **mechanistic attribution** — showing *which* diagnostic each
mechanism repairs — and (iii) a comparison against the state-of-the-art remedy
(Continual Backprop). We report results honestly, including where mechanisms do
not help.

## 2. Related work

**Loss of plasticity.** Dohare et al. (2024) document the phenomenon and propose
Continual Backprop (continual utility-based unit reinitialisation). Lyle et al.
(2023) analyse its causes (feature-rank collapse, dormant units, curvature);
Sokar et al. (2023) identify the dormant-neuron phenomenon and propose ReDo
(recycling dormant units); Nikishin et al. (2022) study the primacy bias and
resets; Ash & Adams (2020) propose shrink-and-perturb for warm-starting.

**Biological plasticity.** Homeostatic synaptic scaling (Turrigiano) keeps a
neuron's activity in range by multiplicatively rescaling its inputs. Structural
plasticity (pruning + neurogenesis) remodels connectivity. Both are lifelong
mechanisms; their artificial analogues (weight normalisation, unit
reinitialisation) exist in ML but are rarely framed or compared *as* biological
plasticity for the loss-of-plasticity problem, which is our angle.

## 3. Method

### 3.1 Continual benchmarks

**Permuted-input regression** (`data/streams.py`): a fixed random *teacher* (a
linear-threshold-unit hidden layer + linear readout) defines a nonlinear target;
each task applies a fresh random permutation of the input features before the
teacher. A task therefore requires the learner to re-fit a genuinely new mapping,
while sharing the teacher's structure across tasks. It is fully synthetic and
CPU-reproducible. We measure the learner's *late-task loss* (mean loss over the
second half of each task) — its **current** fitting ability.

**Continual Permuted-MNIST** (`data/mnist.py`): the recognised benchmark — a
sequence of tasks, each a fixed random permutation of the 784 input pixels of
MNIST, trained online with cross-entropy. We report per-task *late accuracy* (its
current fitting ability). This confirms the synthetic finding on real data.

### 3.2 Diagnostics

On a fixed probe batch we track the standard loss-of-plasticity signatures
(`diagnostics/plasticity.py`): **dormant-unit fraction** (units whose normalised
mean activation falls below a threshold), **effective rank** (Roy–Vetterli, of the
last-layer features) and **mean weight magnitude**.

### 3.3 Mechanisms

- **Homeostatic synaptic scaling** — after each optimiser step, each neuron's
  incoming weight vector is multiplicatively rescaled toward a homeostatic
  set-point (its initial norm), bounding weight growth.
- **Structural plasticity** — an EMA of each hidden unit's activation gives its
  utility; every *P* steps the least-useful *mature* units are reset (incoming
  weights re-initialised = neurogenesis, outgoing weights zeroed = neutral start),
  with a maturation window protecting new units. Related to Continual Backprop /
  ReDo; our additions are maturation protection and composability with homeostasis.
- **Combined** — both together.

Baselines: vanilla SGD, L2 weight decay, shrink-and-perturb, Continual Backprop
(SOTA), ReDo.

## 4. Experimental setup

Online SGD, a small MLP. **Synthetic**: 250 tasks, batch size 1, 10 seeds.
**Permuted-MNIST**: 300 tasks, batch size 16, 8 seeds. Per run we reduce the task
history to scalar metrics (final late loss / accuracy, plasticity ratio =
final/early, dormant fraction, effective rank, weight magnitude). We report mean,
standard deviation and bootstrap 95% CIs across seeds, and permutation tests of
each mechanism vs vanilla and vs Continual Backprop. Hypotheses, benchmarks,
mechanisms, metrics and analysis were preregistered (`docs/preregistration.md`).

## 5. Results

Vanilla SGD loses plasticity: its per-task late loss *rises* over the sequence
(plasticity ratio 1.11), with the dormant fraction reaching 0.26 and effective
rank falling to 10.9. Table 1 reports the final metrics (mean over 10 seeds) and
permutation p-values against vanilla and against Continual Backprop (CBP).

**Table 1.** Final per-task late loss (lower = more retained plasticity), plasticity
ratio (final/early; <1 = improving), dormant fraction, effective rank; p-values vs
vanilla and vs Continual Backprop.

| method | late loss | ratio | dormant | eff. rank | p vs vanilla | p vs CBP |
|---|---|---|---|---|---|---|
| vanilla | 0.817 | 1.11 | 0.264 | 10.9 | — | 6×10⁻⁴ |
| L2 | 0.804 | 1.08 | 0.595 | 2.24 | 0.71 | 0.007 |
| shrink-and-perturb | 0.702 | 0.97 | 0.388 | 3.58 | 0.016 | 0.92 |
| ReDo | 0.765 | 1.04 | 0.024 | 20.0 | 0.045 | 0.042 |
| Continual Backprop (SOTA) | 0.707 | 1.01 | 0.004 | 18.8 | 0.001 | — |
| **homeostatic (ours)** | **0.584** | 0.80 | 0.185 | 8.2 | **2×10⁻⁴** | **0.021** |
| **structural (ours)** | 0.715 | 1.01 | 0.021 | 17.7 | 0.001 | 0.77 |
| **combined (ours)** | **0.549** | 0.80 | 0.047 | 9.1 | **1×10⁻⁴** | **3×10⁻⁴** |

**H1 (preservation).** Homeostatic, structural and combined all significantly
reduce late loss versus vanilla (p < 0.05). **H3 (competitiveness).** Homeostatic
and combined *significantly exceed* Continual Backprop (p = 0.021, 3×10⁻⁴);
structural matches it (p = 0.77). L2 is counter-productive (it shrinks weights,
killing units: dormant 0.60, rank 2.2).

### 5.1 Confirmation on Continual Permuted-MNIST

We repeat the study on Continual Permuted-MNIST (8 seeds); Table 2 reports final
*retained accuracy* (the natural metric for classification) and the cross-entropy
late loss on which significance is tested.

**Table 2.** Permuted-MNIST: retained accuracy (higher = better), CE late loss, and
p-values (permutation on CE loss) vs vanilla and vs Continual Backprop (CBP).

| method | retained accuracy | CE late loss | p vs vanilla | p vs CBP |
|---|---|---|---|---|
| vanilla | 0.589 | 1.16 | — | 3×10⁻⁴ |
| shrink-and-perturb | 0.756 | 0.731 | 3×10⁻⁴ | 3×10⁻⁴ |
| Continual Backprop (SOTA) | 0.699 | 0.881 | 3×10⁻⁴ | — |
| **homeostatic (ours)** | **0.762** | 0.747 | 3×10⁻⁴ | 3×10⁻⁴ |
| **structural (ours)** | 0.733 | 0.794 | 3×10⁻⁴ | 3×10⁻⁴ |
| **combined (ours)** | 0.743 | 0.810 | 3×10⁻⁴ | 3×10⁻⁴ |

Vanilla again loses plasticity (accuracy 0.73→0.59). **Homeostatic scaling retains
the most accuracy (0.762)** and significantly exceeds Continual Backprop
(p < 10⁻³); all three biological methods beat it. The result is thus *robust across
both benchmarks* for the headline claim (homeostatic > SOTA). We flag the honest
nuances: on MNIST the strong baseline shrink-and-perturb is a close competitor
(0.756), and `combined` trails `homeostatic` here (0.743) — i.e. the *best-of-ours*
method is benchmark-dependent, even though homeostatic's advantage over the SOTA
is consistent.

## 6. Mechanism attribution (H2)

Figure `mechanism_attribution.png` plots final dormant fraction against final late
loss; the mechanisms separate along **two near-orthogonal axes**:

- **Structural family** (structural, ReDo, Continual Backprop) drives the dormant
  fraction to ≈0.02 and keeps effective rank high (≈18–20) — it repairs the
  **dead-unit** failure mode — but leaves late loss at ≈0.71–0.77.
- **Homeostatic scaling** does *not* reduce dormant units (0.185) yet attains the
  lowest late loss (0.584); it bounds weight magnitude (final |w| 0.086 vs vanilla
  0.119) — it repairs a **different, weight-conditioning** failure mode.
- **Combined** sits at both low dormant fraction (0.047) and low late loss (0.549),
  inheriting both repairs — direct support for H2's complementarity claim.

## 7. Limitations

One synthetic benchmark and a small network; results should be confirmed on
Continual Permuted-MNIST and larger models. Few seeds keep the significance tests
low-powered. Mechanism hyper-parameters use literature-informed defaults rather
than per-method tuning. Only plain SGD is studied; interactions with adaptive
optimisers are untested. Any observed *improvement over time* (plasticity ratio
< 1) may partly reflect the shared-teacher structure of the benchmark and warrants
a dedicated probe.

## 8. Conclusion

Two lifelong-plasticity mechanisms from neuroscience — homeostatic synaptic
scaling and structural pruning/neurogenesis — measurably mitigate loss of
plasticity in continually-trained networks. A *simple* homeostatic rule not only
prevents the phenomenon but **outperforms the state-of-the-art remedy on both a
synthetic benchmark and Continual Permuted-MNIST**, and a mechanistic analysis
explains why the mechanisms are complementary. We temper this honestly: the
best-performing method is benchmark-dependent, a strong baseline is competitive on
MNIST, the networks are small and the optimiser is plain SGD, and the plasticity
*gain* over time deserves dedicated scrutiny. Natural next steps — larger networks,
adaptive optimisers, and metaplasticity — would further harden the claim.

## References

Abbreviated; to be expanded. Dohare et al. 2024 (*Nature*); Lyle et al. 2023;
Sokar et al. 2023; Nikishin et al. 2022; Ash & Adams 2020; Turrigiano 2008;
Abraham & Bear 1996.
