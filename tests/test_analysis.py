from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from fer_rf_sim import constant_current as analysis
from fer_rf_sim.analysis import build_current_array, constant_current_slice


@pytest.mark.parametrize("trace", [[1, 2, 3], [3, 2, 1], [1, 3, 2], [1, np.nan, 3], [1, 1, 2]])
def test_cli_and_shared_inversion_agree(trace):
    heights = np.array([1., 2., 3.])
    voltage = np.array([0., 1.])
    current = np.broadcast_to(np.array(trace)[:, None, None], (3, 2, 1)).copy()
    expected = constant_current_slice(1.5, heights, build_current_array(current, voltage))
    actual = analysis.find_constant_current_slice(1.5, current, heights, voltage, [0])
    np.testing.assert_allclose(actual[0], expected[..., 0], equal_nan=True)
    np.testing.assert_allclose(actual[1], expected[..., 1], equal_nan=True)
    outside = analysis.find_constant_current_slice(4, current, heights, voltage, [0])
    assert np.isnan(outside[0]).all()


@pytest.mark.parametrize("axis", [[0, 0], [1, 0], [0, np.nan], [0]])
def test_invalid_voltage_axis(axis):
    with pytest.raises(ValueError, match="V_values"):
        build_current_array(np.ones((2, len(axis), 1)), axis)


def test_multiple_currents_selects_rf_and_builds_gradient_once(monkeypatch):
    voltage = np.array([0., 1., 2.])
    heights = np.array([1., 2., 3.])
    current = heights[:, None, None] * (voltage[None, :, None] + 1)
    monkeypatch.setattr(analysis, "load_simulation_data", lambda path: (current, heights, voltage, [0.]))
    monkeypatch.setattr(analysis.go.Figure, "show", lambda self: None)
    calls = []
    original = analysis.build_current_array
    monkeypatch.setattr(analysis, "build_current_array",
                        lambda *args: (calls.append(1), original(*args))[1])
    figure = analysis.analyze_multiple_currents([2., 3.])
    assert len(figure.data) == 2
    assert len(calls) == 1


@pytest.mark.parametrize("script, arguments", [
    ("quick_check.py", []), ("constant_current_analysis.py", ["--multiple", "1", "2"]),
])
def test_real_analysis_commands_fail_for_missing_artifact(tmp_path, script, arguments):
    result = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / script),
                             "--h5-path", str(tmp_path / "missing.h5"), *arguments],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1