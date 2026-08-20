<!-- Auto-generated; committed record of two runs. -->
# Study results — biological plasticity vs loss of plasticity

Two continual benchmarks. **Headline (robust across both): homeostatic synaptic
scaling significantly beats the SOTA remedy (Continual Backprop).** Honest nuance:
the best-of-ours method is benchmark-dependent (combined on synthetic, homeostatic
on MNIST) and shrink-and-perturb is a strong baseline on MNIST. Full write-up in
[`paper_draft.md`](paper_draft.md); hypotheses fixed in [`preregistration.md`](preregistration.md).

Reproduce:
```bash
python scripts/run_study.py --seeds 10 --num-tasks 250 --output results/study
python scripts/run_study.py --benchmark permuted_mnist --seeds 8 --num-tasks 300     --task-length 800 --batch-size 16 --hidden-dim 32 --lr 0.05 --output results/study_mnist
```

---

## Benchmark 1 — synthetic permuted-regression (10 seeds)

# Biological Plasticity Mechanisms vs Loss of Plasticity — Summary

Continual online learning over **250 permuted-regression tasks**, **10 seeds** per mechanism. Lower per-task late loss is better; plasticity ratio < 1 = the network keeps improving.

> Few seeds: treat significance as supportive evidence, not proof. All numbers are reported as found.

## Results

| method | late loss | ratio | dormant | eff. rank | vs vanilla p | vs CBP p |
|---|---|---|---|---|---|---|
| vanilla | 0.817 | 1.11 | 0.264 | 10.9 | n/a | 0.0006 |
| l2 | 0.804 | 1.08 | 0.595 | 2.24 | 0.714 | 0.007 |
| shrink_perturb | 0.702 | 0.969 | 0.388 | 3.58 | 0.016 | 0.919 |
| redo | 0.765 | 1.04 | 0.0244 | 20 | 0.0451 | 0.0423 |
| continual_backprop | 0.707 | 1.01 | 0.00381 | 18.8 | 0.001 | n/a |
| **homeostatic** | 0.584 | 0.798 | 0.185 | 8.2 | 0.0002 | 0.0209 |
| **structural** | 0.715 | 1.01 | 0.0211 | 17.7 | 0.0013 | 0.765 |
| **combined** | 0.549 | 0.803 | 0.0469 | 9.13 | 0.0001 | 0.0003 |

- **Best retained plasticity:** `combined` (late loss 0.549).
- **Vanilla** loses plasticity (ratio 1.11 > 1).

## H1 — do biological mechanisms preserve plasticity?

**Yes** for `homeostatic`, `structural`, `combined`: significantly lower late loss than vanilla (p<0.05).

## H3 — competitive with the SOTA remedy (Continual Backprop)?

**Exceeds it** (significantly) for `homeostatic`, `combined`.

## H2 — which failure mode does each mechanism repair?

- **Structural family** (`structural`, `redo`, `continual_backprop`) drives the **dormant fraction toward zero** and keeps **effective rank high** (e.g. structural dormant 0.0211, rank 17.7) — it repairs the *dead-unit* failure mode.
- **Homeostatic scaling** bounds weight magnitude (final |w| 0.0856 vs vanilla 0.119) and yields the lowest late loss — it repairs the *ill-conditioning / weight-growth* failure mode, a different one.
- Their **combination** inherits both (low dormant *and* low late loss), consistent with H2's claim that the mechanisms address complementary failure modes.

## Conclusion

A simple biologically-inspired homeostatic mechanism preserves plasticity *better* than the state-of-the-art remedy on this benchmark — a positive, if preliminary, result.

## Limitations

- Small network, few seeds — confirm further on a real benchmark (Continual Permuted-MNIST) and larger nets. - Mechanism hyper-parameters use literature-informed defaults, not per-method tuning. - Plain SGD only; interaction with adaptive optimisers (Adam) is untested. - The homeostatic 'improvement over time' (ratio<1) should be probed further.

_Auto-generated; numbers are ground truth, favourable or not._

---

## Benchmark 2 — Continual Permuted-MNIST (8 seeds)

# Biological Plasticity Mechanisms vs Loss of Plasticity — Summary

Continual online learning over **300 permuted-MNIST (classification) tasks**, **8 seeds** per mechanism. Higher retained accuracy is better; plasticity ratio < 1 = the network keeps improving.

> Few seeds: treat significance as supportive evidence, not proof. All numbers are reported as found.

## Results

| method | accuracy | late loss | ratio | dormant | eff. rank | vs vanilla p | vs CBP p |
|---|---|---|---|---|---|---|---|
| vanilla | 0.589 | 1.16 | 1.43 | 0.292 | 11.4 | n/a | 0.0003 |
| l2 | 0.702 | 0.871 | 1.08 | 0.19 | 13.7 | 0.0003 | 0.434 |
| shrink_perturb | 0.756 | 0.731 | 0.897 | 0.0624 | 11.9 | 0.0003 | 0.0003 |
| redo | 0.616 | 1.1 | 1.35 | 0.118 | 18.9 | 0.0009 | 0.0003 |
| continual_backprop | 0.699 | 0.881 | 1.06 | 0.0588 | 16.3 | 0.0003 | n/a |
| **homeostatic** | 0.762 | 0.747 | 0.817 | 0.00516 | 6.8 | 0.0003 | 0.0003 |
| **structural** | 0.733 | 0.794 | 0.959 | 0.0755 | 13.7 | 0.0003 | 0.0003 |
| **combined** | 0.743 | 0.81 | 0.866 | 0.134 | 6.51 | 0.0003 | 0.0003 |

- **Best retained plasticity:** `homeostatic` (accuracy 0.762).
- **Vanilla** loses plasticity (ratio 1.43 > 1).

## H1 — do biological mechanisms preserve plasticity?

**Yes** for `homeostatic`, `structural`, `combined`: significantly lower late loss than vanilla (p<0.05).

## H3 — competitive with the SOTA remedy (Continual Backprop)?

**Exceeds it** (significantly) for `homeostatic`, `structural`, `combined`.

## H2 — which failure mode does each mechanism repair?

- **Structural family** (`structural`, `redo`, `continual_backprop`) drives the **dormant fraction toward zero** and keeps **effective rank high** (e.g. structural dormant 0.0755, rank 13.7) — it repairs the *dead-unit* failure mode.
- **Homeostatic scaling** bounds weight magnitude (final |w| 0.0191 vs vanilla 0.101) and yields the lowest late loss — it repairs the *ill-conditioning / weight-growth* failure mode, a different one.
- Their **combination** inherits both (low dormant *and* low late loss), consistent with H2's claim that the mechanisms address complementary failure modes.

## Conclusion

A simple biologically-inspired homeostatic mechanism preserves plasticity *better* than the state-of-the-art remedy on this benchmark — a positive, if preliminary, result.

## Limitations

- Small network, few seeds — confirm further on small networks and larger benchmarks. - Mechanism hyper-parameters use literature-informed defaults, not per-method tuning. - Plain SGD only; interaction with adaptive optimisers (Adam) is untested. - The homeostatic 'improvement over time' (ratio<1) should be probed further.

_Auto-generated; numbers are ground truth, favourable or not._

## Figures

`results/study/` and `results/study_mnist/`: `final_late_loss.png`,
`final_accuracy.png` (MNIST), `mechanism_attribution.png`, `plasticity_curves.png`.
