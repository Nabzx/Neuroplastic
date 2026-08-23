<!-- Committed record of the A1-A3 robustness runs (5 seeds each). -->
# Robustness results (A1–A3)

Follow-ups to the main study, preregistered in [`preregistration.md`](preregistration.md)
(amendment). Reproduce with `scripts/probe_plasticity_gain.py`,
`scripts/run_robustness.py`, `scripts/run_sensitivity.py`.

## A1 — is the plasticity gain genuine? (independent teachers)

Plasticity ratio (final/early late loss; **< 1 = the network keeps improving**),
shared teacher vs a fresh teacher per task (no shared structure to accumulate):

| method | shared teacher | independent teachers |
|---|---|---|
| vanilla | 1.177 ± 0.053 | 1.148 ± 0.041 |
| **homeostatic** | 0.863 ± 0.113 | **0.729 ± 0.061** |
| combined | 0.840 ± 0.077 | 0.785 ± 0.065 |

**Verdict:** homeostatic's ratio stays below 1 with *independent* teachers — in fact
*lower* than with a shared teacher — while vanilla loses plasticity in both. The
improvement is **genuine retained plasticity, not shared-structure accumulation.**
This resolves the main caveat of the study.

## A2 — robust to Adam and larger networks?

Final per-task late loss (lower = better), and whether homeostatic beats Continual
Backprop (CBP; Holm-corrected p):

| optimiser × width | vanilla | CBP | **homeostatic** | combined | homeo < CBP? |
|---|---|---|---|---|---|
| SGD, 32 | 0.836 | 0.700 | **0.533** | 0.537 | yes (p = 0.02) |
| SGD, 128 | 0.743 | 0.586 | 0.624 | 0.594 | **no** (n.s.) |
| Adam, 32 | 0.987 | 0.740 | **0.511** | 0.544 | yes (p = 0.02) |
| Adam, 128 | 0.990 | 0.933 | **0.538** | 0.617 | yes (p = 0.02) |

**Verdict:** the advantage is **robust to the optimiser** — homeostatic beats CBP
significantly under Adam at both widths (and CBP *degrades sharply* under Adam while
homeostatic does not), and at the small SGD network. The **one exception** is the
large network under plain SGD, where CBP matches homeostatic (combined stays
competitive). Reported honestly: the effect is strongest when capacity is limited or
the optimiser is adaptive.

## A3 — a lucky hyper-parameter?

Homeostatic scaling `rate` swept; final late loss (references: vanilla 0.886,
Continual Backprop 0.728):

| rate | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| late loss | 0.589 | 0.565 | 0.606 | 0.550 | 0.548 |

**Verdict:** below Continual Backprop at **all 5 settings** (spread 0.06) — the
advantage is **insensitive to the coefficient**, not a lucky tuning. See
`results/sensitivity/sensitivity_homeostatic_rate.png`.
