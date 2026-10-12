from pathlib import Path
import subprocess
import sys

from fer_rf_sim import workflow as run_workflow


def test_workflow_reuses_runner_path(tmp_path, monkeypatch):
    artifact = tmp_path / "selected.h5"
    artifact.touch()
    monkeypatch.setattr(run_workflow.SimulationRunner, "run", lambda self, config: artifact)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda command, check: calls.append(command))
    monkeypatch.setattr(run_workflow, "find_latest_simulation_file",
                        lambda *args: (_ for _ in ()).throw(AssertionError("rediscovered")))
    assert run_workflow.main(["--I-target", "1e-9"]) == 0
    assert len(calls) == 2
    for command in calls:
        assert command[0] == sys.executable
        assert command[1] == "-m"
        assert command[2] in ("fer_rf_sim.check", "fer_rf_sim.constant_current")
        assert command[3:5] == ["--h5-path", str(artifact)]


def test_analysis_failure_is_workflow_failure(tmp_path, monkeypatch):
    artifact = tmp_path / "selected.h5"
    artifact.touch()

    def fail(command, check):
        raise subprocess.CalledProcessError(2, command)

    monkeypatch.setattr(subprocess, "run", fail)
    assert run_workflow.main([
        "--skip-simulation", "--h5-path", str(artifact), "--I-target", "1e-9",
    ]) == 1


def test_missing_artifact_is_workflow_failure(tmp_path):
    assert run_workflow.main([
        "--skip-simulation", "--h5-path", str(tmp_path / "missing.h5"),
    ]) == 1


def test_workflow_list_luts_does_not_simulate(tmp_path, monkeypatch):
    def unexpected_run(self, config):
        raise AssertionError("Listing caches must not start a simulation")

    monkeypatch.setattr(run_workflow.SimulationRunner, "run", unexpected_run)
    assert run_workflow.main(["--list-luts", "--out", str(tmp_path)]) == 0


def test_explicit_counts_override_quick_preset(tmp_path, monkeypatch):
    captured = []

    def run(self, config):
        captured.append(config)
        return tmp_path / "selected.h5"

    monkeypatch.setattr(run_workflow.SimulationRunner, "run", run)
    assert run_workflow.main(["--steps", "simulation", "--quick-test", "--n_V", "100"]) == 0
    assert captured[0].n_E == 50
    assert captured[0].n_V == 100


def test_real_workflow_reuses_new_artifact_outside_repo(tmp_path):
    import os
    import h5py
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "current.h5").write_bytes(b"stale file must not be selected")
    environment = dict(os.environ)
    result = subprocess.run([
        sys.executable, "-m", "fer_rf_sim", "workflow", "--steps", "simulation", "quick-check", "constant-current",
        "--n_E", "8", "--n_V", "3", "--n_Z", "2", "--n_A", "1", "--n_cheb", "2", "--threads", "1",
        "--out", str(output), "--I-target", "1e-9", "--no-show",
    ], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    artifacts = list(output.glob("current_phit*.h5"))
    assert len(artifacts) == 1
    assert f"Key simulation parameters for {artifacts[0]}" in result.stdout
    with h5py.File(artifacts[0]) as file:
        assert file["I"].shape == (2, 3, 1)
    assert (output / "current.h5").read_bytes() == b"stale file must not be selected"