import numpy as np
import pytest

from fer_rf_sim.broadening import broaden_signal, broaden_index_spectrum
from fer_rf_sim.quadrature import CGQ
from fer_rf_sim.csv_data import iter_filtered_spectra, load_fitted_peaks, explicit_baseline_mask


def test_zero_rf_scalar_and_uneven_broadening():
    assert CGQ(np.square, 2., 0.) == 4
    assert CGQ(np.square, 2., 1.) == pytest.approx(4.5)
    bias = np.array([0., .1, .4, 1., 2.])
    signal = 3 * bias + 2
    np.testing.assert_allclose(broaden_signal(bias, signal, 0), signal, atol=0)
    result = broaden_index_spectrum(signal, bias, 0)
    np.testing.assert_allclose(result["broadened channel"], 3 * result["bias"] + 2, atol=0)


def test_broadening_edge_policy_and_units():
    bias = np.linspace(-2, 2, 101)
    result = broaden_signal(bias, bias ** 2, .2)
    assert np.isnan(result[0]) and np.isnan(result[-1])
    assert np.isfinite(result[50])
    millivolts = broaden_signal(bias * 1000, bias ** 2, 200)
    np.testing.assert_allclose(result, millivolts, rtol=1e-12, atol=1e-14, equal_nan=True)


def test_stream_aggregate_schema_and_filters(tmp_path):
    path = tmp_path / "aggregate.csv"
    path.write_text('Voltage,Setpoint,RF Freq,RF Power,dZ/dV,RelativeZ,"FileName"\n'
                    '100,1000,100000000,-30,1,2,"with,comma"\n'
                    '200,1000,100000001,0,3,4,other\n'
                    '300,500,100000000,-30,5,6,ignored\n')
    chunks = list(iter_filtered_spectra(path, setpoint=1000, rf_frequency=1e8,
                                       rf_powers=[-30, 0], voltage_unit="mV", chunksize=1))
    assert len(chunks) == 2
    assert chunks[0].iloc[0]["FileName"] == "with,comma"
    assert chunks[0].attrs["units"]["V"] == "mV"
    with pytest.raises(ValueError, match="Fitted-peak"):
        load_fitted_peaks(path)


def test_no_filename_order_baseline_inference():
    import pandas as pd
    frame = pd.DataFrame({"FileName": ["z.csv", "a.csv"]})
    with pytest.raises(ValueError, match="baselines require"):
        explicit_baseline_mask(frame)
    assert explicit_baseline_mask(frame, {"z.csv": True, "a.csv": False}).tolist() == [True, False]