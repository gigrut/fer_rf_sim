"""Application-facing simulation service built on the existing physics engine."""

from pathlib import Path

from fer_rf_sim.engine import simulate
from fer_rf_sim.config import SimulationConfig


class SimulationRunner:
    """Small application boundary around the existing simulation implementation."""

    def run(self, config: SimulationConfig, *, progress_callback=None, cancel_event=None) -> Path:
        config.validate()
        hooks = {}
        if progress_callback is not None:
            hooks["progress_callback"] = progress_callback
        if cancel_event is not None:
            hooks["cancel_event"] = cancel_event
        result = simulate(config.as_namespace(), **hooks)
        return Path(result)


def run_simulation(config: SimulationConfig) -> Path:
    """Run one simulation and return its generated HDF5 artifact path."""
    return SimulationRunner().run(config)
