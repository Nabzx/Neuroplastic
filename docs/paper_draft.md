# Biologically-Inspired Homeostatic and Structural Plasticity Mitigate Loss of Plasticity

*Working draft — a workshop-length paper. Results/abstract/conclusion are filled
from the study in `results/study/`; method and setup are final.*

**Author:** Nabil Shah

---

## Abstract

*(to be finalised from results)* Deep networks trained continually lose the
ability to learn. We ask whether *biological* neuroplasticity mechanisms —
homeostatic synaptic scaling and utility-gated structural plasticity (pruning +
neurogenesis) — prevent this, and which failure mode each repairs. On a
permuted-regression continual benchmark (10 seeds) we find that [KEY RESULT:
homeostatic scaling retains the most plasticity, competitive with / exceeding
Continual Backprop], and that structural and homeostatic mechanisms address
*complementary* failure modes (dead units vs weight ill-conditioning), so their
combination inherits both. [Honest caveats.]

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

### 3.1 Continual benchmark

**Permuted-input regression** (`data/streams.py`): a fixed random *teacher* (a
linear-threshold-unit hidden layer + linear readout) defines a nonlinear target;
each task applies a fresh random permutation of the input features before the
teacher. A task therefore requires the learner to re-fit a genuinely new mapping
(as in Continual Permuted MNIST), while sharing the teacher's structure across
tasks. It is fully synthetic and CPU-reproducible. We measure the learner's
*late-task loss* (mean loss over the second half of each task) — its **current**
fitting ability.

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

Online SGD (batch size 1), a small MLP, over 250 permuted tasks. **10 seeds** per
mechanism. Per run we reduce the task history to scalar metrics (final late loss,
plasticity ratio = final/early late loss, dormant fraction, effective rank, weight
magnitude). We report mean, standard deviation and bootstrap 95% CIs across seeds,
and permutation tests of each mechanism vs vanilla and vs Continual Backprop.
Hypotheses, benchmarks, mechanisms, metrics and analysis were preregistered
(`docs/preregistration.md`).

## 5. Results

*(filled from `results/study/`)*

## 6. Mechanism attribution

*(filled: which diagnostic each mechanism repairs)*

## 7. Limitations

One synthetic benchmark and a small network; results should be confirmed on
Continual Permuted-MNIST and larger models. Few seeds keep the significance tests
low-powered. Mechanism hyper-parameters use literature-informed defaults rather
than per-method tuning. Only plain SGD is studied; interactions with adaptive
optimisers are untested. Any observed *improvement over time* (plasticity ratio
< 1) may partly reflect the shared-teacher structure of the benchmark and warrants
a dedicated probe.

## 8. Conclusion

*(filled from results)*

## References

Abbreviated; to be expanded. Dohare et al. 2024 (*Nature*); Lyle et al. 2023;
Sokar et al. 2023; Nikishin et al. 2022; Ash & Adams 2020; Turrigiano 2008;
Abraham & Bear 1996.
