"""Canonical method registry — the single taxonomy shared by the experiments
(experiments/utils/runner.py) and the dashboard (dashboard/backend/schema.py).

See registry.py for the method definitions.
"""
from wildfire_governance.methods.registry import (  # noqa: F401
    METHODS,
    CALIBRATION_ENV,
    MethodSpec,
    get_method,
    run_episode_kwargs,
    all_method_ids,
    table1_method_ids,
)
