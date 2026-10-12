"""Standalone native Qt peak extraction and RF dependence pages."""

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout,
    QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QStyle, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from .peaks import (
    export_peak_csv, extract_spectra, real_peak_spectra, simulated_peak_spectra,
    track_rf_peaks,
)
from .plotly_host import PlotlyHost


class PeakAnalysisPanel:
    """Own two pages; the parent window owns data, loading, and tab placement."""

    def __init__(self, window, web_engine_view):
        self.window = window
        self.peak_results = []
        self.rf_peaks = []
        self.peak_host = None
        self.rf_host = None
        self._simulation = None
        self._closed = False
        self.peak_page, peak_layout = self._page("Peak extraction")
        self.rf_page, rf_layout = self._page("RF dependence")

        self.peak_source = self._combo(["Simulated data", "Real data"])
        self.rf_source = self._combo(["Simulated data", "Real data"])
        self._source_controls(peak_layout, self.peak_source, "peak")
        self._source_controls(rf_layout, self.rf_source, "rf")

        self.peak_mode = self._combo([
            "constant-current dI/dV", "constant-current dZ/dV", "fixed-height dI/dV",
        ])
        self.peak_target = QLineEdit()
        self.peak_target.setPlaceholderText("Current (a.u.)")
        self.peak_height = self._combo([])
        self.peak_polarity = self._combo(["positive", "negative"])
        self.peak_prominence = QDoubleSpinBox()
        self.peak_prominence.setDecimals(8)
        self.peak_prominence.setRange(0.0, 1e9)
        self.peak_prominence.setValue(0.0)
        self.peak_distance = QSpinBox()
        self.peak_distance.setRange(1, 2147483647)
        self.peak_distance.setValue(1)
        settings = QGridLayout()
        settings.setColumnStretch(0, 1)
        settings.setColumnStretch(1, 1)
        for row, fields in enumerate((
            (("Mode", self.peak_mode), ("Polarity", self.peak_polarity)),
            (("Current (a.u.)", self.peak_target), ("Tip height (nm)", self.peak_height)),
            (("Min prominence", self.peak_prominence), ("Distance (samples)", self.peak_distance)),
        )):
            for column, (label, field) in enumerate(fields):
                settings.addWidget(QLabel(label), row * 2, column)
                settings.addWidget(field, row * 2 + 1, column)
        peak_layout.addLayout(settings)

        self.real_spectrum_selector = self._combo([])
        self.real_rf_input = QLineEdit()
        self.real_rf_input.setPlaceholderText("unknown")
        self.real_rf_button = self._button("Set RF", QStyle.SP_DialogApplyButton)
        self.real_rf_button.clicked.connect(self._set_real_rf)
        self._real_rf_widgets = QWidget()
        metadata = QGridLayout(self._real_rf_widgets)
        metadata.setContentsMargins(0, 0, 0, 0)
        metadata.setColumnStretch(1, 1)
        metadata.addWidget(QLabel("Measured spectrum"), 0, 0)
        metadata.addWidget(self.real_spectrum_selector, 0, 1, 1, 3)
        metadata.addWidget(QLabel("RF (V peak)"), 1, 0)
        metadata.addWidget(self.real_rf_input, 1, 1)
        metadata.addWidget(self.real_rf_button, 1, 2)
        peak_layout.addWidget(self._real_rf_widgets)

        self.extract_button = self._button("Extract peaks", QStyle.SP_MediaPlay)
        self.extract_button.setProperty("role", "primary")
        self.extract_button.clicked.connect(self._run_peak_extraction)
        self.peak_export_button = self._button("Export CSV", QStyle.SP_DialogSaveButton)
        self.peak_export_button.clicked.connect(self._export_peaks)
        actions = QGridLayout()
        actions.addWidget(self.extract_button, 0, 0)
        actions.setColumnStretch(1, 1)
        actions.addWidget(self.peak_export_button, 0, 2)
        peak_layout.addLayout(actions)
        self.peak_spectrum = self._combo([])
        peak_layout.addWidget(self.peak_spectrum)
        self.peak_status = self._status(peak_layout)
        self.peak_plot_view, self.peak_host = self._plot(peak_layout, web_engine_view)
        self.peak_table = self._table(peak_layout, [
            "Position (V)", "Height", "Prominence", "Width (V)",
        ])

        self.rf_metric = self._combo(["position", "shift", "height", "prominence", "width"])
        self.rf_track = self._combo(["All tracks"])
        self.rf_max_shift = QDoubleSpinBox()
        self.rf_max_shift.setDecimals(6)
        self.rf_max_shift.setRange(0.000001, 1e6)
        self.rf_max_shift.setValue(0.25)
        self.rf_max_shift.setSuffix(" V")
        rf_settings = QGridLayout()
        rf_settings.setColumnStretch(0, 1)
        rf_settings.setColumnStretch(1, 1)
        rf_settings.addWidget(QLabel("Metric"), 0, 0)
        rf_settings.addWidget(QLabel("Track"), 0, 1)
        rf_settings.addWidget(self.rf_metric, 1, 0)
        rf_settings.addWidget(self.rf_track, 1, 1)
        rf_settings.addWidget(QLabel("Max shift"), 2, 0)
        rf_settings.addWidget(self.rf_max_shift, 3, 0)
        self.rf_export_button = self._button("Export CSV", QStyle.SP_DialogSaveButton)
        self.rf_export_button.clicked.connect(self._export_peaks)
        rf_settings.addWidget(self.rf_export_button, 3, 1)
        rf_layout.addLayout(rf_settings)
        self.rf_status = self._status(rf_layout)
        self.rf_plot_view, self.rf_host = self._plot(rf_layout, web_engine_view)
        self.rf_table = self._table(rf_layout, [
            "Spectrum", "Track", "RF (V peak)", "Position (V)", "Shift (V)",
            "Height", "Prominence", "Width (V)", "Signal unit",
        ])
        peak_layout.addStretch()
        rf_layout.addStretch()

        self.peak_source.currentIndexChanged.connect(self._source_changed)
        self.rf_source.currentIndexChanged.connect(self._rf_source_changed)
        self.peak_mode.currentIndexChanged.connect(self._settings_changed)
        self.peak_target.textChanged.connect(self.invalidate)
        self.peak_height.currentIndexChanged.connect(self.invalidate)
        self.peak_polarity.currentIndexChanged.connect(self.invalidate)
        self.peak_prominence.valueChanged.connect(self.invalidate)
        self.peak_distance.valueChanged.connect(self.invalidate)
        self.peak_spectrum.currentIndexChanged.connect(self._render_peak_plot)
        self.real_spectrum_selector.currentIndexChanged.connect(self._render_real_rf_input)
        self.rf_max_shift.valueChanged.connect(self._refresh_rf_analysis)
        self.rf_metric.currentIndexChanged.connect(self._render_rf_plot)
        self.rf_track.currentIndexChanged.connect(self._render_rf_plot)
        self.refresh_inputs()

    def _page(self, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        heading = QLabel(title)
        heading.setProperty("role", "title")
        layout.addWidget(heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        return scroll, layout

    @staticmethod
    def _combo(items):
        combo = QComboBox()
        combo.addItems(items)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(12)
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        combo.currentTextChanged.connect(combo.setToolTip)
        combo.setToolTip(combo.currentText())
        return combo

    def _button(self, text, icon):
        button = QPushButton(text)
        button.setIcon(self.window.style().standardIcon(icon))
        return button

    def _source_controls(self, layout, selector, prefix):
        controls = QGridLayout()
        controls.setColumnStretch(1, 1)
        controls.addWidget(QLabel("Source"), 0, 0)
        controls.addWidget(selector, 0, 1)
        file_button = self._button("Open file", QStyle.SP_DialogOpenButton)
        folder_button = self._button("Open folder", QStyle.SP_DirOpenIcon)
        file_button.clicked.connect(self._load_file)
        folder_button.clicked.connect(self._load_folder)
        controls.addWidget(file_button, 0, 2)
        controls.addWidget(folder_button, 0, 3)
        setattr(self, f"{prefix}_file_button", file_button)
        setattr(self, f"{prefix}_folder_button", folder_button)
        layout.addLayout(controls)

    @staticmethod
    def _status(layout):
        status = QLabel()
        status.setProperty("role", "muted")
        status.setWordWrap(True)
        layout.addWidget(status)
        return status

    def _plot(self, layout, web_engine_view):
        host = None
        if web_engine_view is None or self.window.plot_document is None:
            view = QLabel("Plot unavailable")
            view.setProperty("role", "muted")
        else:
            view = web_engine_view()
            view.setMinimumHeight(220)
            view.setMaximumHeight(340)
            host = PlotlyHost(view, self.window.plot_document, lazy=True)
        view.hide()
        layout.addWidget(view)
        return view, host

    @staticmethod
    def _table(layout, headers):
        table = QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        table.setMinimumHeight(140)
        table.setMaximumHeight(260)
        table.setMinimumWidth(0)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(table)
        return table

    def _source_changed(self, index):
        self.rf_source.blockSignals(True)
        self.rf_source.setCurrentIndex(index)
        self.rf_source.blockSignals(False)
        self._update_controls()
        self.invalidate()

    def _rf_source_changed(self, index):
        self.peak_source.setCurrentIndex(index)

    def _settings_changed(self, *_):
        self._update_controls()
        self.invalidate()

    def _update_controls(self):
        simulated = self.peak_source.currentText() == "Simulated data"
        fixed_height = self.peak_mode.currentText() == "fixed-height dI/dV"
        self.peak_mode.setEnabled(simulated)
        self.peak_target.setEnabled(simulated and not fixed_height)
        self.peak_height.setEnabled(simulated and fixed_height)
        self.peak_folder_button.setEnabled(not simulated)
        self.rf_folder_button.setEnabled(not simulated)
        self._real_rf_widgets.setVisible(not simulated)
        available = self.window.simulated_data is not None if simulated else bool(self.window.real_spectra)
        self.extract_button.setEnabled(available)

    def _load_file(self, *_):
        if self.peak_source.currentText() == "Simulated data":
            self.window._select_simulation_for_plot()
        else:
            self.window._select_real_data_file()
        self.refresh_inputs()

    def _load_folder(self, *_):
        if self.peak_source.currentText() == "Real data":
            self.window._select_real_data_directory()
            self.refresh_inputs()

    def refresh_inputs(self):
        """Reflect parent data after loading, including preloaded measured spectra."""
        data = self.window.simulated_data
        previous_height = self.peak_height.currentData()
        self.peak_height.blockSignals(True)
        self.peak_height.clear()
        if data is not None:
            for index, height in enumerate(data.z):
                self.peak_height.addItem(f"{height:.8g} nm", index)
            if data is self._simulation and previous_height is not None:
                self.peak_height.setCurrentIndex(self.peak_height.findData(previous_height))
        self.peak_height.blockSignals(False)
        if data is not self._simulation or not self.peak_target.text().strip():
            targets = self.window.constant_current_targets
            self.peak_target.blockSignals(True)
            if data is not None and len(targets):
                index = min(max(self.window.simulated_current_slider.value(), 0), len(targets) - 1)
                self.peak_target.setText(f"{targets[index]:.12g}")
            else:
                self.peak_target.clear()
            self.peak_target.blockSignals(False)
        self._simulation = data
        previous_label = self.real_spectrum_selector.currentData()
        self.real_spectrum_selector.blockSignals(True)
        self.real_spectrum_selector.clear()
        for spectrum in self.window.real_spectra:
            self.real_spectrum_selector.addItem(str(spectrum.path), str(spectrum.path))
        index = self.real_spectrum_selector.findData(previous_label)
        if index >= 0:
            self.real_spectrum_selector.setCurrentIndex(index)
        self.real_spectrum_selector.blockSignals(False)
        self.real_spectrum_selector.setToolTip(self.real_spectrum_selector.currentText())
        self._render_real_rf_input()
        self._update_controls()
        self.invalidate()

    def _render_real_rf_input(self, *_):
        index = self.real_spectrum_selector.currentIndex()
        valid = 0 <= index < len(self.window.real_spectra)
        self.real_rf_input.setEnabled(valid)
        self.real_rf_button.setEnabled(valid)
        amplitude = self.window.real_spectra[index].rf_amplitude if valid else None
        self.real_rf_input.setText("" if amplitude is None else f"{amplitude:.12g}")

    def _set_real_rf(self, *_):
        index = self.real_spectrum_selector.currentIndex()
        if not 0 <= index < len(self.window.real_spectra):
            self.peak_status.setText("No measured spectrum selected.")
            return
        text = self.real_rf_input.text().strip()
        try:
            amplitude = None if text.lower() in {"", "unknown", "none"} else float(text)
            if amplitude is not None and (not np.isfinite(amplitude) or amplitude < 0):
                raise ValueError("RF must be finite and nonnegative (V peak).")
        except ValueError as error:
            self.peak_status.setText(f"Invalid RF amplitude: {error}")
            return
        self.window.real_spectra[index] = replace(
            self.window.real_spectra[index], rf_amplitude=amplitude,
            rf_source="manual" if amplitude is not None else "unknown",
        )
        self.invalidate()
        self._render_real_rf_input()
        self.window._render_real_plot()
        value = "unknown" if amplitude is None else f"{amplitude:.8g} V peak"
        self.peak_status.setText(f"Measured RF set to {value} (session only).")

    @staticmethod
    def _clear_plot(view, host):
        view.hide()
        if host is not None:
            host.pending = None
            if host.ready and not host.closed:
                import plotly.graph_objects as go

                host.update(go.Figure())

    def invalidate(self, *_):
        """Discard measurements and export state after source or extraction edits."""
        self.peak_results = []
        self.peak_spectrum.blockSignals(True)
        self.peak_spectrum.clear()
        self.peak_spectrum.blockSignals(False)
        self.peak_spectrum.setToolTip("")
        self.peak_spectrum.setEnabled(False)
        self.peak_table.setRowCount(0)
        self.peak_export_button.setEnabled(False)
        self._clear_plot(self.peak_plot_view, self.peak_host)
        self._clear_rf()
        self.peak_status.setText("No extracted peaks.")
        self.rf_status.setText("No extracted peaks.")

    def _clear_rf(self):
        self.rf_peaks = []
        self.rf_table.setRowCount(0)
        self.rf_track.blockSignals(True)
        self.rf_track.clear()
        self.rf_track.addItem("All tracks", None)
        self.rf_track.blockSignals(False)
        self.rf_track.setEnabled(False)
        self.rf_export_button.setEnabled(False)
        self._clear_plot(self.rf_plot_view, self.rf_host)

    def _run_peak_extraction(self, *_):
        self.invalidate()
        try:
            if self.peak_source.currentText() == "Real data":
                spectra = real_peak_spectra(self.window.real_spectra)
            else:
                if self.window.simulated_data is None:
                    raise ValueError("No simulated data loaded.")
                mode = self.peak_mode.currentText()
                target = None
                if mode != "fixed-height dI/dV":
                    target = float(self.peak_target.text())
                    if not np.isfinite(target) or target <= 0:
                        raise ValueError("Current target must be finite and positive.")
                spectra = simulated_peak_spectra(
                    self.window.simulated_data, mode=mode, current_target=target,
                    height_index=self.peak_height.currentData(),
                    current_array=self.window.current_array,
                )
            if not spectra:
                raise ValueError("No spectra loaded.")
            results = extract_spectra(
                spectra, prominence=self.peak_prominence.value(),
                distance=self.peak_distance.value(), polarity=self.peak_polarity.currentText(),
            )
        except Exception as error:
            self.peak_status.setText(f"Peak extraction failed: {error}")
            return
        self.peak_results = results
        self.peak_spectrum.blockSignals(True)
        for index, result in enumerate(results):
            self.peak_spectrum.addItem(result.spectrum.label, index)
            self.peak_spectrum.setItemData(index, result.spectrum.label, Qt.ToolTipRole)
        self.peak_spectrum.blockSignals(False)
        self.peak_spectrum.setEnabled(True)
        count = sum(result.peaks.position.size for result in results)
        self.peak_export_button.setEnabled(count > 0)
        self.peak_status.setText(f"{count} peaks in {len(results)} spectra.")
        self._render_peak_plot()
        self._refresh_rf_analysis()

    @staticmethod
    def _fill_table(table, rows):
        table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for column, value in enumerate(values):
                text = f"{value:.10g}" if isinstance(value, (float, np.floating)) else str(value)
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                table.setItem(row_index, column, item)

    @staticmethod
    def _style_figure(figure):
        figure.update_layout(
            template="plotly_white", margin=dict(l=60, r=20, t=30, b=50),
            font=dict(family="Segoe UI", size=12), legend=dict(orientation="h"),
        )

    def _render_peak_plot(self, *_):
        self.peak_table.setRowCount(0)
        self._clear_plot(self.peak_plot_view, self.peak_host)
        index = self.peak_spectrum.currentIndex()
        if not 0 <= index < len(self.peak_results):
            return
        result = self.peak_results[index]
        spectrum, peaks = result.spectrum, result.peaks
        self.peak_spectrum.setToolTip(spectrum.label)
        self.peak_table.setHorizontalHeaderLabels([
            "Position (V)", f"Height ({spectrum.signal_unit})",
            f"Prominence ({spectrum.signal_unit})", "Width (V)",
        ])
        self._fill_table(self.peak_table, list(zip(
            peaks.position, peaks.height, peaks.prominence, peaks.width,
        )))
        if self.peak_host is None:
            return
        import plotly.graph_objects as go

        figure = go.Figure()
        figure.add_trace(go.Scatter(x=spectrum.voltage, y=spectrum.signal, mode="lines", name="Spectrum"))
        figure.add_trace(go.Scatter(
            x=peaks.position, y=peaks.height, mode="markers", name="Peaks",
            marker=dict(size=8, color="#c65046"),
        ))
        figure.update_layout(xaxis_title="Bias (V)", yaxis_title=f"Signal ({spectrum.signal_unit})")
        self._style_figure(figure)
        self.peak_plot_view.show()
        self.peak_host.update(figure)

    def _refresh_rf_analysis(self, *_):
        selected_track = self.rf_track.currentData()
        self._clear_rf()
        if not self.peak_results:
            self.rf_status.setText("No extracted peaks.")
            return
        unknown = sum(result.spectrum.rf_amplitude is None for result in self.peak_results)
        amplitudes = {
            result.spectrum.rf_amplitude for result in self.peak_results
            if result.spectrum.rf_amplitude is not None
        }
        if len(amplitudes) < 2:
            self.rf_status.setText(
                f"RF analysis requires two distinct known amplitudes; {len(amplitudes)} available. "
                f"Excluded {unknown} unknown-RF spectra."
            )
            return
        try:
            self.rf_peaks = track_rf_peaks(self.peak_results, max_shift=self.rf_max_shift.value())
        except Exception as error:
            self.rf_status.setText(f"RF tracking failed: {error}. Excluded {unknown} unknown-RF spectra.")
            return
        tracks = sorted({peak.track for peak in self.rf_peaks})
        self.rf_export_button.setEnabled(bool(self.rf_peaks))
        self.rf_track.blockSignals(True)
        for track in tracks:
            self.rf_track.addItem(f"Track {track}", track)
        selected_index = self.rf_track.findData(selected_track)
        if selected_index >= 0:
            self.rf_track.setCurrentIndex(selected_index)
        self.rf_track.blockSignals(False)
        self.rf_track.setEnabled(bool(tracks))
        self.rf_status.setText(
            f"{len(self.rf_peaks)} measurements, {len(tracks)} tracks, {len(amplitudes)} RF amplitudes. "
            f"Excluded {unknown} unknown-RF spectra. Shift reference: lowest observed RF per track."
        )
        self._render_rf_plot()

    def _render_rf_plot(self, *_):
        self.rf_table.setRowCount(0)
        self._clear_plot(self.rf_plot_view, self.rf_host)
        selected = self.rf_track.currentData()
        observations = [peak for peak in self.rf_peaks if selected is None or peak.track == selected]
        rows = []
        for peak in observations:
            spectrum = self.peak_results[peak.result_index].spectrum
            rows.append((
                spectrum.label, peak.track, peak.rf_amplitude, peak.position, peak.shift,
                peak.height, peak.prominence, peak.width, spectrum.signal_unit,
            ))
        self._fill_table(self.rf_table, rows)
        if not observations or self.rf_host is None:
            return
        import plotly.graph_objects as go

        metric = self.rf_metric.currentText()
        groups = {}
        for peak in observations:
            groups.setdefault(peak.track, {}).setdefault(peak.rf_amplitude, []).append(getattr(peak, metric))
        unit = "V"
        if metric in {"height", "prominence"}:
            units = {self.peak_results[peak.result_index].spectrum.signal_unit for peak in observations}
            unit = next(iter(units)) if len(units) == 1 else "mixed units"
        figure = go.Figure()
        for track, values in sorted(groups.items()):
            amplitudes = sorted(values)
            figure.add_trace(go.Scatter(
                x=amplitudes, y=[float(np.median(values[amplitude])) for amplitude in amplitudes],
                mode="lines+markers", name=f"Track {track}",
            ))
        figure.update_layout(
            xaxis_title="RF amplitude (V peak)", yaxis_title=f"Median {metric} ({unit})",
        )
        self._style_figure(figure)
        self.rf_plot_view.show()
        self.rf_host.update(figure)

    def _export_peaks(self, *_):
        if not self.peak_results or not any(result.peaks.position.size for result in self.peak_results):
            return
        path, _ = QFileDialog.getSaveFileName(
            self.window, "Export peak measurements", "peaks.csv", "CSV files (*.csv)",
        )
        if not path:
            return
        try:
            export_peak_csv(self.peak_results, path, rf_peaks=self.rf_peaks)
        except Exception as error:
            message = f"Peak export failed: {error}"
        else:
            message = f"Exported peak measurements to {path}"
        self.peak_status.setText(message)
        self.rf_status.setText(message)

    def close(self):
        """Release web views without closing the parent's shared document."""
        if self._closed:
            return
        self._closed = True
        for host in (self.peak_host, self.rf_host):
            if host is not None:
                host.close()