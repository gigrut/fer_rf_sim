from pathlib import Path

import pytest

from fer_rf_sim.config import SimulationConfig, load_config, parse_config


def test_explicit_default_and_false_override_yaml(tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("simulation:\n  n_V: 200\n  use_lut: true\n  Z_min: 0.5\n  out: results\n")
    config = parse_config(["--config", str(config_path), "--n_V", "100", "--no-lut"])
    assert config.n_V == 100
    assert config.use_lut is False
    assert config.z_min == 0.5
    assert Path(config.out) == tmp_path / "results"


@pytest.mark.parametrize("values", [
    {"n_E": 2.5}, {"n_A": True}, {"v_min": 1, "v_max": 1}, {"E_max": float("inf")},
    {"phi_tip": float("nan")}, {"threads": 0}, {"upscale_z": 0}, {"upscale_v": 1.1},
    {"e_extra": -6}, {"use_lut": "false"}, {"A_min": 1, "A_max": 1},
])
def test_invalid_config_rejected(values):
    with pytest.raises(ValueError):
        SimulationConfig(**values).validate()


def test_unknown_config_keys_and_missing_files(tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("simulation:\n  n_typo: 20\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(config_path)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_cli_path_override_is_cwd_relative(tmp_path):
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("simulation:\n  out: results\n")
    assert load_config(config_path, {"out": "explicit"}).out == "explicit"


@pytest.mark.parametrize("document", ["[]", "false", "simulation: []", "simulation: null"])
def test_malformed_yaml_sections_rejected(tmp_path, document):
    path = tmp_path / "invalid.yaml"
    path.write_text(document)
    with pytest.raises(ValueError):
        load_config(path)