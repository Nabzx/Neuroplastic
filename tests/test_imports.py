"""Smoke tests: packages import cleanly and the dry-run wiring resolves.

Torch/PettingZoo-dependent modules are skipped when those optional deps are
absent, so this suite passes with only the core install.
"""

import importlib

import pytest

# Modules that must import with only the core dependencies installed.
LIGHT_MODULES = [
    "core",
    "core.registry",
    "core.seeding",
    "configs",
    "configs.schema",
    "configs.loader",
    "communication",
    "communication.graph",
    "communication.message",
    "communication.topology",
    "communication.protocol",
    "communication.channel",
    "plasticity",
    "plasticity.base",
    "plasticity.hebbian",
    "plasticity.homeostasis",
    "plasticity.modulation",
    "evaluation",
    "evaluation.graph_metrics",
    "evaluation.information_metrics",
    "evaluation.coordination",
    "evaluation.stability",
    "evaluation.report",
    "evaluation.evaluator",
    "analysis.topology_analysis",
    "analysis.specialisation_analysis",
    "analysis.statistics",
    "environments",
    "environments.registry",
    "environments.benchmarks",
    "agents",
    "agents.base",
    "training",
    "training.trainer",
    "training.cli",
    "training.algorithms",
    "training.utils",
]


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_light_module_imports(module):
    importlib.import_module(module)


def test_torch_dependent_modules():
    pytest.importorskip("torch")
    importlib.import_module("agents.policy")
    importlib.import_module("agents.recurrent_policy")
    importlib.import_module("training.learner")
    importlib.import_module("training.rollout")
    importlib.import_module("communication.adaptive")
    importlib.import_module("plasticity.plastic_edges")
    importlib.import_module("training.compare")


def test_dry_run_describe_resolves_wiring():
    from configs import load_config
    from training.trainer import Trainer

    cfg = load_config("configs/default.yaml")
    described = Trainer(cfg).describe()
    assert described["experiment"] == "nci_default"
    assert described["communication"]["topology"]["name"] == "adaptive"
    assert described["plasticity"]["rule"]["impl"] == "HebbianRule"
    assert described["algorithm"]["name"] == "ippo"


def test_cli_dry_run(capsys):
    from training.cli import main

    exit_code = main(["--config", "configs/default.yaml", "--dry-run"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "nci_default" in out
