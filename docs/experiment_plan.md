# Experiment Plan — Neuroplastic Collective Intelligence

This document is the reference specification for the study. Placeholders across
the codebase (`NotImplementedError`) point here for their intended design.

---

## 1. Research question

Do **neuroplastic communication mechanisms** — activity-dependent, Hebbian-inspired
adaptation of *who communicates with whom* and *how strongly* — improve
**emergent coordination** and **functional specialisation** in cooperative
multi-agent systems, compared with conventional static-communication baselines?

## 2. Hypotheses

| ID | Hypothesis | Isolating comparison |
|----|------------|----------------------|
| **H1** | Hebbian plasticity on communication weights improves coordination. | `default` vs `no_plasticity` |
| **H2** | A dynamic (adaptive) topology beats a static one under matched plasticity. | `default` vs `static_topology` |
| **H3** | Neuroplastic communication increases functional specialisation (more distinct, stable roles). | `default` vs `dense_static_baseline` |
| **H0 (sanity)** | Communication of any kind beats none. | `dense_static_baseline` vs `no_communication` |

## 3. Variables

**Independent (manipulated).**
- *Plasticity*: on / off; rule ∈ {hebbian, oja}; modulation ∈ {none, reward-gated, three-factor}.
- *Topology*: fully-connected · k-nearest · static · adaptive (learned).
- *Protocol*: mean · attention · GNN.

**Dependent (measured).** See §6.

**Controlled (held fixed across a comparison).** Environment and task, number of
agents, policy architecture and parameter count, RL algorithm and its
hyper-parameters, training budget, and the set of seeds. Ablations are expressed
as *minimal diffs* against `configs/default.yaml` so exactly one factor changes
at a time.

## 4. Conditions (configs)

| Config | Topology | Plasticity | Role |
|--------|----------|------------|------|
| `configs/default.yaml` | adaptive | hebbian | main model |
| `configs/ablations/no_plasticity.yaml` | adaptive | off | tests H1 |
| `configs/ablations/static_topology.yaml` | static | hebbian | tests H2 |
| `configs/ablations/dense_static_baseline.yaml` | fully-connected | off | conventional-comms baseline |
| `configs/ablations/no_communication.yaml` | — (disabled) | off | lower bound |

## 5. Benchmarks

Cooperative tasks from PettingZoo MPE, chosen because coordination and (for some)
explicit communication are required for success:

- **`simple_spread`** — N agents must cover N landmarks without colliding (pure
  coordination; no privileged communication channel).
- **`simple_reference`** — each agent must be guided to its goal landmark by
  another agent (requires communication).
- **`simple_speaker_listener`** — asymmetric roles: a non-moving speaker must
  communicate the goal to a moving listener (a natural specialisation probe).

Selection criteria for extending the suite: cooperative reward, varying degrees
of required communication, and — where available — a ground-truth interaction
structure to validate discovered topologies against.

## 6. Metrics

All metrics are computed per episode and aggregated across evaluation episodes
and seeds. Registries live in `evaluation/`.

### 6.1 Coordination (task-level) — `evaluation/coordination.py`
- **Episode return** (team reward). *(functional)*
- **Success rate** (task-specific completion). *(deferred: needs env success signal)*
- **Coordination index**: normalised gain over an independent-learner baseline on
  the same task. *(deferred)*

### 6.2 Graph-theoretic (structure) — `evaluation/graph_metrics.py` *(functional today)*
Computed on the (time-averaged or snapshotted) interaction graph:
- **Density** — how much of the possible communication is used.
- **Modularity** — sub-group / community structure (proxy for functional modules).
- **Global clustering** — local interconnectedness.
- **Characteristic path length** — communication efficiency.
- **Degree centralisation** — hub formation vs. egalitarian communication.

### 6.3 Information-theoretic (flow) — `evaluation/information_metrics.py`
- **Message entropy** — how much information messages carry. *(functional)*
- **Mutual information** between senders/receivers (or messages and actions/goals). *(functional, plug-in estimator)*
- **Transfer entropy** — *directed* information flow along edges. *(deferred: needs a
  bias-corrected lagged conditional-MI estimator; e.g. KSG.)*

### 6.4 Functional specialisation — `analysis/specialisation_analysis.py`
From per-agent **role descriptors** (`agents/specialisation.py`) aggregated over a
rollout window (action-distribution summary + message-emission statistics +
graph centrality):
- **Role entropy** — diversity of the role distribution. *(functional given labels)*
- **Role cluster count** — number of distinct emergent roles. *(functional given labels)*
- **Role stability** — persistence of role assignments across training. *(deferred)*
- Descriptor **clustering** (silhouette-selected k-means). *(deferred)*

## 7. Model design (to be implemented)

### 7.1 Agent forward pass — `agents/policy.py`, `agents/recurrent_policy.py`
Per step, per agent *i*:
1. `h_enc = Encoder(obs_i)`.
2. `m_i = MessageHead(h_i^{prev})` — outgoing message.
3. `c_i = Protocol.aggregate(i, {m_j : j ∈ neighbours(i)}, graph)` — integrate
   incoming messages, gated by the plastic edge weights `w_{j→i}`.
4. `h_i = Core([h_enc; c_i], h_i^{prev})` — recurrent update.
5. `logits_i, v_i = ActorCritic(h_i)`; sample action.

### 7.2 Adaptive topology — `communication/topology.py::AdaptiveTopology`
Each step, produce edge-existence/gates from a learned scoring function over
agent states (e.g. a bilinear or attention score), optionally sparsified to
`max_neighbours` (top-k) or thresholded. Differentiable relaxation (Gumbel-softmax
/ concrete edges) for gradient flow; hard edges at evaluation.

### 7.3 Plasticity — `plasticity/`
- **Hebbian (reference, implemented):** `Δw_{j→i} = η · g · pre_j · post_i − λ w_{j→i}`,
  with scalar co-activity summarised as message/context norms.
- **Vector/Oja variants (deferred):** outer-product Hebbian on message/context
  vectors; Oja normalisation for stability without explicit decay.
- **Neuromodulation (deferred):** three-factor rule where `g` is a reward-derived
  eligibility-gated signal (`plasticity/modulation.py`).
- **Homeostasis (implemented):** per-receiver synaptic scaling
  (`plasticity/homeostasis.py`) keeps incoming weights bounded.

Plasticity operates on the **communication graph weights** (fast, online,
activity-driven) and is *separate* from the policy's gradient learning (slow,
reward-driven) — a deliberate two-timescale design.

### 7.4 Learning algorithm — `training/algorithms.py`
Baseline: **Independent PPO (IPPO)** — each agent optimises a clipped-surrogate
objective; the communication substrate is shared. Rollouts additionally record
interaction-graph snapshots and exchanged messages for post-hoc analysis
(`training/rollout.py`).

## 8. Protocol

1. **Seeds.** ≥ 5 seeds per condition (scale up for headline results).
2. **Budget.** Identical `training.total_steps` across a comparison.
3. **Evaluation.** Fixed held-out episodes with `evaluation.deterministic = true`.
4. **Logging.** Snapshot the interaction graph every `evaluation.log_topology_every`
   steps for temporal analysis (`analysis/topology_analysis.py`).
5. **Statistics.** Report mean with a bootstrap CI over seeds
   (`analysis/statistics.py::bootstrap_ci`) and a Mann–Whitney U test for between-
   condition differences; report effect sizes, not just p-values.
6. **Reproducibility.** Every run is fully determined by `(config, seed)`; seeding
   via `core.seeding.set_global_seed`.

## 9. Threats to validity

- **Confounds:** plastic vs. static models must be matched on parameter count and
  compute — otherwise gains may reflect capacity, not plasticity.
- **Estimator bias:** plug-in information estimators are biased at small sample
  sizes; use as *relative* measures across matched conditions and prefer
  bias-corrected estimators for headline claims.
- **Stability:** pure Hebbian learning diverges; homeostasis/decay are required
  and their settings are themselves a factor to check.
- **Task ceiling/floor:** benchmarks where communication is unnecessary (or
  impossible to exploit) cannot separate the hypotheses; hence the mixed suite.

## 10. Milestones

- **M0 — Scaffold (this repo):** architecture, interfaces, config system, plan,
  functional graph/metrics/plasticity maths, tests. ✅
- **M1 — Forward pass:** feedforward actor-critic (§7.1) with a single-round mean
  communication protocol over a **fixed** graph; trains on `simple_spread`. ✅
- **M2 — Learning baselines:** shared-parameter PPO with three fixed communication
  settings (none / fully-connected / sparse ring), reward + episode-length logging
  (`configs/baselines/`, `training/`). ✅  *Pending:* rollout recording of
  graphs/messages, checkpoints, recurrent policies, learned attention/GNN protocols.
- **M3 — Adaptive topology & plasticity coupling:** §7.2 + neuromodulated §7.3.
- **M4 — Evaluation & analysis:** full metric suite over recorded rollouts,
  role-clustering, transfer entropy.
- **M5 — Study:** sweep all conditions × seeds × benchmarks; test H1–H3; write-up.
