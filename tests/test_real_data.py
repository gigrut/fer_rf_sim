import numpy as np
import pytest

from fer_rf_sim.real_data import RealDataRepository
from fer_rf_sim.peaks import detect_peaks
from fer_rf_sim.units import amplitude_to_peak, dbm_to_peak_volts


def test_shuffled_filenames_do_not_assign_rf(tmp_path):
    for name in ("zeta.csv", "alpha.csv"):
        (tmp_path / name).write_text("bias (mV),lockin-x (mV),curr (nA)\n0,1,2\n100,2,3\n")
    spectra = RealDataRepository().load(tmp_path)
    assert all(spectrum.rf_amplitude is None for spectrum in spectra)
    with pytest.raises(ValueError, match="RF metadata"):
        spectra[0].require_rf()
    mapped = RealDataRepository().load(tmp_path, rf_mapping={"zeta.csv": 0.56, "alpha.csv": 0.})
    assert {spectrum.path.name: spectrum.rf_amplitude for spectrum in mapped} == {"zeta.csv": 0.56, "alpha.csv": 0.}
    np.testing.assert_allclose(mapped[0].data["bias (V)"], [0, 0.1])
    np.testing.assert_allclose(mapped[0].data["current (A)"], [2e-9, 3e-9], atol=0)


def test_mapping_precedes_actual_metadata(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_text("bias (mV),lockin-x (mV),RF RMS (V)\n0,1,1\n100,2,1\n")
    assert RealDataRepository().load(path)[0].rf_amplitude == pytest.approx(np.sqrt(2))
    spectrum = RealDataRepository().load(path, rf_mapping={path.name: 0.})[0]
    assert spectrum.rf_amplitude == 0 and spectrum.rf_source == "mapping"


def test_explicit_rf_units():
    assert dbm_to_peak_volts(0, 50) == pytest.approx(np.sqrt(0.1))
    assert amplitude_to_peak(2000, unit="mV", kind="peak-to-peak") == 1
    assert amplitude_to_peak(1, kind="rms") == pytest.approx(np.sqrt(2))
    with pytest.raises(ValueError):
        amplitude_to_peak(-1)


def test_peak_prominences_align_without_filter():
    result = detect_peaks([0, 1, 2, 3, 4], [0, 2, 0, 1, 0])
    np.testing.assert_array_equal(result.prominence, [2, 1])
    assert result.prominence.shape == result.position.shape