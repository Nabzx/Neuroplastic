"""Training orchestration (architecture only; the optimisation loop is deferred).

* :mod:`training.trainer`    -- wires env + agents + communication + plasticity
  + algorithm into a runnable object.
* :mod:`training.algorithms` -- the MARL algorithm interface (IPPO baseline).
* :mod:`training.rollout`    -- experience collection / storage.
* :mod:`training.callbacks`  -- logging / checkpointing / topology snapshots.
* :mod:`training.cli`        -- the ``nci`` command-line entrypoint.
"""

from training.trainer import Trainer

__all__ = ["Trainer"]
