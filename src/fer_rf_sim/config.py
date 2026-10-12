"""Shared simulation defaults, validation, and explicit configuration precedence."""

import argparse
from dataclasses import asdict, dataclass, fields
import math
from numbers import Integral, Real
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


@dataclass
class SimulationConfig:
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
    # Fermi energies above each electrode's own band bottom (low temperature, free-electron bands).
    E_F_tip: float = 5.5
    E_F_samp: float = 5.5
    n_cheb: int = 32
    threads: int = -1
    out: str = "fer_output"
    force_rebuild_lut: bool = False
    list_luts: bool = False
    use_lut: bool = False
    amplitude_chunk: int = 8
    dc_table: bool = False
    dc_table_points: int = 129
    dc_table_max_points: int = 16385
    dc_table_rtol: float = 1e-3
    # Tunneling current spans many decades; any fixed absolute floor skips refinement at large heights.
    dc_table_atol: float = 0.0
    upscale: int = 1
    upscale_z: int | None = None
    upscale_v: int | None = None
    upscale_e: int | None = None

    def validate(self) -> None:
        for name in ("n_E", "n_V", "n_Z", "n_A", "n_cheb", "upscale",
                     "upscale_z", "upscale_v", "upscale_e", "threads", "amplitude_chunk",
                     "dc_table_points", "dc_table_max_points"):
            value = getattr(self, name)
            if value is None and name in ("upscale_z", "upscale_v", "upscale_e"):
                continue
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            minimum = 2 if name in ("n_E", "n_V", "n_Z", "dc_table_points", "dc_table_max_points") else 1
            if name == "threads" and value == -1:
                continue
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}" +
                                 (" or -1" if name == "threads" else ""))
        for name in ("E_min", "E_max", "e_extra", "v_min", "v_max", "z_min", "z_max",
                     "A_min", "A_max", "phi_tip", "phi_samp", "E_F_tip", "E_F_samp",
                     "dc_table_rtol", "dc_table_atol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        effective_energy = self.E_max + self.e_extra
        if (self.E_min < 0 or self.E_min >= self.E_max or
                not math.isfinite(effective_energy) or effective_energy <= self.E_min):
            raise ValueError("energy range and effective E_max + e_extra must exceed E_min >= 0")
        if self.v_min >= self.v_max or not math.isfinite(self.v_max - self.v_min):
            raise ValueError("voltage range must be strictly increasing and finite")
        if self.z_min <= 0 or self.z_min >= self.z_max:
            raise ValueError("distance range must satisfy 0 < z_min < z_max")
        if self.A_min < 0 or self.A_min > self.A_max or (self.n_A > 1 and self.A_min == self.A_max):
            raise ValueError("RF amplitude range must increase for multiple samples")
        if not all(math.isfinite(value) for value in (self.v_min - self.A_max, self.v_max + self.A_max)):
            raise ValueError("effective RF voltage range must be finite")
        if self.phi_tip <= 0 or self.phi_samp <= 0:
            raise ValueError("work functions must be positive")
        for name in ("E_F_tip", "E_F_samp"):
            if not self.E_min < getattr(self, name) <= effective_energy:
                raise ValueError(f"{name} must exceed E_min and not exceed E_max + e_extra, "
                                 "so the energy grid spans the occupied states")
        if self.dc_table_max_points < self.dc_table_points:
            raise ValueError("dc_table_max_points must be at least dc_table_points")
        if (self.dc_table_rtol < 0 or self.dc_table_atol < 0
                or (self.dc_table_rtol == 0 and self.dc_table_atol == 0)):
            raise ValueError("DC table tolerances must be non-negative and not both zero")
        for name in ("use_lut", "force_rebuild_lut", "list_luts", "dc_table"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.out, (str, Path)) or not str(self.out).strip():
            raise ValueError("out must be a nonempty path")

    def as_namespace(self) -> SimpleNamespace:
        self.validate()
        return SimpleNamespace(**asdict(self))

    @classmethod
    def from_namespace(cls, namespace: Any) -> "SimulationConfig":
        return cls(**{field.name: getattr(namespace, field.name)
                      for field in fields(cls) if hasattr(namespace, field.name)})


def normalize_settings(values: Mapping) -> dict:
    if not isinstance(values, Mapping):
        raise ValueError("simulation settings must be a mapping")
    names = {field.name.lower(): field.name for field in fields(SimulationConfig)}
    normalized = {}
    for key, value in values.items():
        name = names.get(str(key).replace("-", "_").lower())
        if name is None:
            raise ValueError(f"Unknown simulation setting: {key}")
        if name in normalized:
            raise ValueError(f"Duplicate simulation setting: {key}")
        normalized[name] = value
    return normalized


def load_config(config_path=None, overrides=None) -> SimulationConfig:
    """Merge built-ins < YAML < explicit edits; YAML paths are config-relative."""
    path = Path(config_path) if config_path is not None else Path("fer_config.yaml")
    values = {}
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        if document is None:
            document = {}
        if not isinstance(document, dict):
            raise ValueError("configuration must be a mapping")
        if "simulation" in document:
            unknown = set(document) - {"simulation", "real_data"}
            if unknown:
                raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
        values = normalize_settings(document.get("simulation", document))
        if "out" in values:
            output = Path(values["out"])
            values["out"] = str((path.resolve().parent / output).resolve())
    elif config_path is not None:
        raise FileNotFoundError(f"config file not found: {path}")
    values.update(normalize_settings(overrides or {}))
    config = SimulationConfig(**values)
    config.validate()
    return config


def add_simulation_arguments(parser):
    parser.add_argument("--config", default=argparse.SUPPRESS)
    for field in fields(SimulationConfig):
        name = field.name
        if name == "use_lut":
            group = parser.add_mutually_exclusive_group()
            group.add_argument("--use-lut", dest=name, action="store_true", default=argparse.SUPPRESS)
            group.add_argument("--no-lut", "--no-use-lut", dest=name, action="store_false",
                               default=argparse.SUPPRESS)
            continue
        options = ["--" + name]
        if "_" in name:
            options.append("--" + name.replace("_", "-"))
        if isinstance(field.default, bool):
            parser.add_argument(*options, dest=name, action=argparse.BooleanOptionalAction,
                                default=argparse.SUPPRESS)
        else:
            value_type = int if name.startswith("upscale") else type(field.default)
            parser.add_argument(*options, dest=name, type=value_type, default=argparse.SUPPRESS)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="FER simulation (normalized current units)")
    add_simulation_arguments(parser)
    values = vars(parser.parse_args(argv))
    path = values.pop("config", None)
    try:
        return load_config(path, values)
    except (OSError, ValueError, TypeError, yaml.YAMLError) as error:
        parser.error(str(error))