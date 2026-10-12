import numpy as np
import pytest

from fer_rf_sim.peaks import (
    PeakSpectrum, export_peak_csv, extract_spectra, real_peak_spectra, simulated_peak_spectra, track_rf_peaks,
)


def test_batch_extraction_preserves_rf_and_peak_measurements():
    voltage = np.linspace(0, 4, 401)
    spectra = [
        PeakSpectrum(f"RF {amplitude}", voltage,
                     np.exp(-((voltage - (1 + amplitude * 0.2)) / 0.1) ** 2), amplitude)
        for amplitude in (0., 0.5, 1.)
    ]
    results = extract_spectra(spectra, prominence=0.1)
    assert len(results) == 3
    for result, amplitude in zip(results, (0., 0.5, 1.)):
        assert result.spectrum.rf_amplitude == amplitude
        np.testing.assert_allclose(result.peaks.position, [1 + amplitude * 0.2])
        assert result.peaks.width[0] > 0
        assert result.peaks.prominence[0] == pytest.approx(1)


def test_extraction_does_not_bridge_invalid_regions():
    spectrum = PeakSpectrum("partial", np.arange(11.),
                            np.array([0, 2, 0, np.nan, 0, 3, 0, np.nan, 0, 4, 0]))
    result, = extract_spectra([spectrum])
    np.testing.assert_array_equal(result.peaks.index, [1, 5, 9])
    np.testing.assert_array_equal(result.peaks.position, [1, 5, 9])
    assert result.spectrum.rf_amplitude is None


def test_empty_and_invalid_spectra():
    result, = extract_spectra([PeakSpectrum("empty", np.arange(4.), np.full(4, np.nan))])
    assert result.peaks.position.size == 0
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_spectra([PeakSpectrum("bad axis", np.array([0., 2., 1.]), np.ones(3))])
    with pytest.raises(ValueError, match="RF amplitude"):
        extract_spectra([PeakSpectrum("bad RF", np.arange(3.), np.ones(3), -1)])


def test_real_and_simulated_rf_dependence_agree(tmp_path):
    from fer_rf_sim.real_data import load_real_data
    from fer_rf_sim.repository import SimulationData

    voltage = np.linspace(0, 4, 401)
    amplitudes = np.array([0., 0.5, 1.])
    currents = []
    for index, amplitude in enumerate(amplitudes):
        signal = np.exp(-((voltage - (1 + amplitude * 0.2)) / 0.1) ** 2)
        current = np.zeros_like(voltage)
        current[1:] = np.cumsum((signal[1:] + signal[:-1]) * np.diff(voltage) / 2)
        currents.append(current)
        import pandas as pd

        pd.DataFrame({"bias (mV)": voltage[::-1] * 1000,
                      "lockin-x (mV)": signal[::-1] * 1000,
                      "RF (V)": amplitude}).to_csv(tmp_path / f"{2 - index}.csv", index=False)
    cube = np.stack(currents, axis=-1)[None, ...]
    data = SimulationData(tmp_path / "sim.h5", cube, np.array([1.]), voltage, amplitudes, {})
    measured = extract_spectra(real_peak_spectra(load_real_data(tmp_path)), prominence=0.1)
    simulated = extract_spectra(simulated_peak_spectra(data, mode="fixed-height dI/dV"), prominence=0.1)
    real_tracks = track_rf_peaks(measured)
    sim_tracks = track_rf_peaks(simulated)
    np.testing.assert_allclose([peak.position for peak in real_tracks], [peak.position for peak in sim_tracks])
    np.testing.assert_allclose([peak.shift for peak in real_tracks], [0., 0.1, 0.2], atol=1e-14)
    assert {peak.track for peak in real_tracks + sim_tracks} == {1}
    assert all(result.spectrum.signal_unit == "V" for result in measured)


def test_tracking_does_not_renumber_when_a_peak_disappears():
    voltage = np.linspace(0, 4, 401)
    results = extract_spectra([
        PeakSpectrum("baseline", voltage,
                     np.exp(-((voltage - 1) / .1) ** 2) + np.exp(-((voltage - 3) / .1) ** 2), 0),
        PeakSpectrum("RF", voltage, np.exp(-((voltage - 3.1) / .1) ** 2), 1),
        PeakSpectrum("unknown", voltage, np.exp(-((voltage - 2) / .1) ** 2)),
    ], prominence=.1)
    tracks = track_rf_peaks(results)
    assert [peak.track for peak in tracks] == [1, 2, 2]
    assert tracks[-1].shift == pytest.approx(.1)
    assert len(track_rf_peaks(results, max_shift=.05)) == 3
    assert track_rf_peaks(results, max_shift=.05)[-1].track == 3
    with pytest.raises(ValueError, match="two distinct"):
        track_rf_peaks(results[:1] + results[2:])


def test_rf_replicates_share_tracks():
    voltage = np.linspace(0, 2, 201)
    results = extract_spectra([
        PeakSpectrum(str(index), voltage, np.exp(-((voltage - position) / .1) ** 2), amplitude)
        for index, (amplitude, position) in enumerate([(0., 1.), (1., 1.1), (1., 1.12)])
    ])
    tracks = track_rf_peaks(results)
    assert [peak.track for peak in tracks] == [1, 1, 1]
    assert [peak.rf_amplitude for peak in tracks] == [0., 1., 1.]


@pytest.mark.parametrize("mode", ["constant-current dI/dV", "constant-current dZ/dV"])
def test_simulated_constant_current_modes_preserve_invalid_regions(tmp_path, mode):
    from fer_rf_sim.repository import SimulationData

    voltage = np.linspace(1, 3, 21)
    heights = np.array([1., 2., 3.])
    amplitudes = np.array([0., 1.])
    cube = np.exp(-heights[:, None, None]) * voltage[None, :, None] * np.ones((1, 1, 2))
    data = SimulationData(tmp_path / "sim.h5", cube, heights, voltage, amplitudes, {})
    spectra = simulated_peak_spectra(data, mode=mode, current_target=.4)
    assert len(spectra) == 2
    assert all(np.isnan(spectrum.signal[0]) for spectrum in spectra)
    assert all(np.isfinite(spectrum.signal[-1]) for spectrum in spectra)
    assert len(extract_spectra(spectra)) == 2


def test_peak_csv_preserves_units_unknown_rf_and_tracks(tmp_path):
    import csv

    results = extract_spectra([
        PeakSpectrum("measured.csv", np.arange(3.), np.array([0., 2., 0.]), None, "V"),
        PeakSpectrum("RF-off", np.arange(3.), np.array([0., 2., 0.]), 0., "V"),
        PeakSpectrum("RF-on", np.arange(3.), np.array([0., 3., 0.]), 1., "V"),
    ])
    path = tmp_path / "peaks.csv"
    export_peak_csv(results, path, rf_peaks=track_rf_peaks(results))
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["rf_peak_v"] == rows[0]["track"] == rows[0]["shift_v"] == ""
    assert rows[0]["signal_unit"] == "V"
    assert rows[1]["track"] == rows[2]["track"] == "1"
    assert float(rows[1]["rf_peak_v"]) == 0
    assert float(rows[2]["shift_v"]) == 0