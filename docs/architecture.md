# Architecture

How the packages fit together, the data flow of one environment step, and where
to extend the system.

## Design principles

1. **Config-driven & extensible.** Components (agents, environments, topologies,
   protocols, plasticity rules, algorithms, metrics) register under a string key
   in a small `core.Registry` and are selected *by name* from a YAML config.
   Adding a variant = write a class + one decorator; never edit the training loop.
2. **Interfaces before implementations.** Abstract base classes fix the contracts;
   compute-heavy internals are placeholders that raise `NotImplementedError` with
   a pointer to the experiment plan.
3. **Light core, heavy extras.** `pip install -e .` pulls only NumPy / NetworkX /
   PyYAML, so config validation and graph/metric analysis run anywhere. PyTorch
   and PettingZoo live in the `[rl]` extra and are imported lazily.
4. **Two learning timescales.** *Plasticity* adapts communication-graph weights
   fast and online from activity; the *RL algorithm* adapts policy parameters
   slowly from reward. They are separate subsystems by design.

## Dependency direction

```
        core  (registry, seeding, types — depends on nothing internal)
          ▲
          │
   ┌──────┼─────────────┬───────────────┐
communication        agents          environments
  (graph,             (policies,       (PettingZoo
   topology,           agent iface)     adapter,
   protocol,              ▲             registry)
   channel) ◀── plasticity│                ▲
     ▲          (hebbian,  │                │
     │           homeostasis)               │
     └─────────────┬───────┴────────────────┘
                   │
               training  (trainer, algorithms, rollout, cli)
                   │
      ┌────────────┼───────────────┐
 evaluation     analysis      visualisation
 (metrics)   (interpretation)   (plots)
```

Arrows point from a dependency to its dependents. `core` is the only shared base;
`training` composes the core subsystems; `evaluation`/`analysis`/`visualisation`
consume their outputs. There are no import cycles: package `__init__` files avoid
eager cross-domain imports, and registration happens via lazy imports inside
factories (`make_*`).

## Data flow — one environment step

```
              ┌─────────────────────── CommunicationChannel.step ───────────────────────┐
 obs_i ──▶ Agent_i.encode_message ──▶ m_i ─┐                                              │
                                           ▼                                              │
                              Topology.update(graph)   ──▶  who talks to whom this step   │
                                           │                                              │
                              Protocol.aggregate(i, msgs, graph) ──▶ context c_i ─────────┼──▶ Agent_i.integrate
                                           │                                              │            │
                              PlasticityRule.apply(graph, msgs, contexts) ──▶ Δw          │            ▼
                              (Hebbian + neuromodulation + homeostasis)                   │      Agent_i.act ──▶ action_i
              └──────────────────────────────────────────────────────────────────────────┘
                                           │
                       env.step(actions) ──▶ rewards, next obs, done
```

Recorded per step for later analysis: the interaction-graph snapshot and the
messages, so `evaluation` and `analysis` can reconstruct structure and information
flow offline.

## Package responsibilities

| Package | Responsibility | Key types |
|---------|----------------|-----------|
| `core` | Shared primitives | `Registry`, `set_global_seed`, type aliases |
| `configs` | Typed schema + YAML loader (inheritance, overrides) | `ExperimentConfig`, `load_config` |
| `agents` | Policy networks + the agent interface | `BaseAgent`, `PolicyNetwork`, `AGENT_REGISTRY` |
| `environments` | Cooperative env interface + benchmarks | `CooperativeEnv`, `ENV_REGISTRY` |
| `communication` | Interaction graph, topology, protocol, routing | `InteractionGraph`, `Topology`, `Protocol`, `CommunicationChannel` |
| `plasticity` | Activity-driven weight adaptation | `PlasticityRule`, `HebbianRule`, homeostasis, modulation |
| `training` | Compose subsystems; algorithm/rollout/CLI | `Trainer`, `Algorithm`, `RolloutBuffer` |
| `evaluation` | Emergent-behaviour metrics | graph / information / coordination metric registries, `Evaluator` |
| `analysis` | Cross-run interpretation | topology trajectories, specialisation, statistics |
| `visualisation` | Plots | interaction graph, metric trajectories |

## Extension points (cookbook)

- **New topology:** subclass `communication.topology.Topology`, decorate with
  `@TOPOLOGY_REGISTRY.register("my_topology")`, set `communication.topology:
  my_topology` in a config.
- **New plasticity rule:** subclass `plasticity.base.PlasticityRule`, register,
  set `plasticity.rule`.
- **New protocol / agent / algorithm / environment / metric:** identical pattern
  against `PROTOCOL_REGISTRY` / `AGENT_REGISTRY` / `ALGORITHM_REGISTRY` /
  `ENV_REGISTRY` / the metric registries.

Because everything is selected by name, `nci --dry-run` can fully resolve and
report a run's wiring before any training code exists — the fastest way to check
a new component is registered and configured correctly.
