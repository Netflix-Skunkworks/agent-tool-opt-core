"""Optimizers — ``agent_tool_opt_core.api.Optimizer`` implementations.

Construct one by id with ``catalog.build_optimizer`` (``llm`` / ``draft`` / ``pi``
/ ``gepa`` / ``toolobserver``). Reward shaping is an Optimizer config (a composable
``method``), not a separate class. Search optimizers (``gepa``) set
``wants_train_eval`` and receive a train-only ``TrainEvaluator``.
"""

from agent_tool_opt_core.optimizers.catalog import build_optimizer, list_optimizers
from agent_tool_opt_core.optimizers._common import (
    OptimizerInfrastructureFailure,
    OptimizerLLMFailure,
)

__all__ = [
    "OptimizerInfrastructureFailure",
    "OptimizerLLMFailure",
    "build_optimizer",
    "list_optimizers",
]
