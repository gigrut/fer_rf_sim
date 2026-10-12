from pathlib import Path
import subprocess
import sys

import pytest


def test_headless_imports_do_not_require_optional_interfaces(tmp_path):
    code = '''
import importlib.abc
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'PySide6', 'plotly', 'matplotlib', 'pandas'}:
            raise AssertionError('Optional dependency imported: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import fer_rf_sim.service
import fer_rf_sim.analysis
import fer_rf_sim.gui
print('headless imports passed')
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "headless imports passed" in result.stdout


@pytest.mark.parametrize("command", ["simulate", "check", "workflow", "constant-current", "transmission", "lut"])
def test_installed_commands_outside_repo(tmp_path, command):
    result = subprocess.run([sys.executable, "-m", "fer_rf_sim", command, "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_example_imports_do_not_plot_or_write(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = (f"import sys; sys.path.insert(0, {str(root)!r}); "
            "import figure_helper; "
            "assert 'matplotlib' not in sys.modules; assert 'plotly' not in sys.modules")
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.iterdir())


def test_legacy_plot_helpers_remain_importable():
    from FER_plotter import plot_T_slice_3d, plot_logI_vs_VZ_slider, scatter3_slider
    assert all(callable(function) for function in (plot_T_slice_3d, plot_logI_vs_VZ_slider, scatter3_slider))