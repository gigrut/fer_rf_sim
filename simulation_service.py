"""Application-facing simulation service built on the existing physics engine."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from FER_constant_current_simulation import simulate


@dataclass
class SimulationConfig:
    """Validated simulation inputs shared by CLI, workflow, and GUI callers."""

    n_E: int = 100
    n_V: int = 100
    n_Z: int = 100
    n_A: int = 30
    E_min: float = 0.01
    E_max: float = 5.5
    e_extra: float = 0.0
    v_min: float = 0.0
    v_max: float = 10.0
    z_min: float = 0.3
    z_max: float = 5.0
    A_min: float = 0.0
    A_max: float = 5.0
    phi_tip: float = 4.0
    phi_samp: float = 5.0
    n_cheb: int = 32
    threads: int = -1
    out: str = "fer_output"
    force_rebuild_lut: bool = False
    list_luts: bool = False
    use_lut: bool = False
    upscale: int = 1
    upscale_z: int | None = None
    upscale_v: int | None = None
    upscale_e: int | None = None

    def validate(self) -> None:
        """Reject invalid grid and barrier settings before expensive work starts."""
        integer_fields = ("n_E", "n_V", "n_Z", "n_A", "n_cheb")
        for name in integer_fields:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.n_V < 2 or self.n_Z < 2:
            raise ValueError("n_V and n_Z must be at least 2")
        if self.n_E < 2:
            raise ValueError("n_E must be at least 2")
        if self.E_min < 0 or self.E_min >= self.E_max:
            raise ValueError("E_min must be non-negative and less than E_max")
        if self.v_min > self.v_max or self.z_min <= 0 or self.z_min >= self.z_max:
            raise ValueError("simulation voltage and distance ranges are invalid")
        if self.A_min < 0 or self.A_min > self.A_max:
            raise ValueError("RF amplitude range is invalid")
        if self.phi_tip <= 0 or self.phi_samp <= 0:
            raise ValueError("work functions must be positive")
        if self.upscale < 1:
            raise ValueError("upscale must be positive")

    def as_namespace(self) -> SimpleNamespace:
        """Return the attribute-style object expected by the existing simulator."""
        self.validate()
        return SimpleNamespace(**asdict(self))

    @classmethod
    def from_namespace(cls, namespace: Any) -> "SimulationConfig":
        """Copy recognized settings from an argparse namespace or similar object."""
        values = {
            field.name: getattr(namespace, field.name)
            for field in fields(cls)
            if hasattr(namespace, field.name)
        }
        return cls(**values)


class SimulationRunner:
    """Small application boundary around the existing simulation implementation."""

    def run(self, config: SimulationConfig) -> Path:
        config.validate()
        result = simulate(config.as_namespace())
        return Path(result)


def run_simulation(config: SimulationConfig) -> Path:
    """Run one simulation and return its generated HDF5 artifact path."""
    return SimulationRunner().run(config)
