"""One local Plotly document shared by desktop views, updated through Plotly.react."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from PySide6.QtCore import QEventLoop, QTimer, QUrl


class PlotlyDocument:
    def __init__(self):
        from plotly.offline import get_plotlyjs
        self.directory = TemporaryDirectory(prefix="fer_plotly_")
        self.path = Path(self.directory.name) / "plot.html"
        self.path.write_text(
            '<!doctype html><html><head><meta charset="utf-8"><style>'
            'html,body,#fer-plot{height:100%;width:100%;margin:0;}</style></head>'
            '<body><div id="fer-plot"></div><script>' + get_plotlyjs() + '</script><script>'
            'new ResizeObserver(() => {const plot = document.getElementById("fer-plot");'
            'if (plot.data) Plotly.Plots.resize(plot);}).observe(document.body);'
            '</script></body></html>',
            encoding="utf-8",
        )

    def close(self):
        for attempt in range(100):
            try:
                self.path.unlink(missing_ok=True)
                break
            except PermissionError:
                if attempt == 99:
                    raise
                loop = QEventLoop()
                QTimer.singleShot(50, loop.quit)
                loop.exec()
        self.directory.cleanup()


class PlotlyHost:
    def __init__(self, view, document, *, lazy=False):
        self.view = view
        self.document = document
        self.started = False
        self.ready = False
        self.closed = False
        self.pending = None
        view.loadFinished.connect(self._loaded)
        if not lazy:
            self._start()

    def _start(self):
        self.started = True
        self.view.load(QUrl.fromLocalFile(str(self.document.path)))

    def _loaded(self, success):
        self.ready = bool(success) and not self.closed
        if self.ready and self.pending is not None:
            self._flush()

    def update(self, figure):
        if self.closed:
            return
        if not self.started and not figure.data:
            return
        self.pending = figure.to_json()
        if not self.started:
            self._start()
        if self.ready and self.pending is not None:
            self._flush()

    def _flush(self):
        payload = json.dumps(self.pending)
        script = (f"(() => {{const figure = JSON.parse({payload});"
                  "Plotly.react('fer-plot', figure.data, figure.layout, {responsive: true});})()")
        self.view.page().runJavaScript(script)
        self.pending = None

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.pending = None
        self.view.stop()
        if not hasattr(self.view.page(), "setHtml"):
            self.view.setUrl(QUrl("about:blank"))
            return
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        completed = []

        def blank_loaded(success):
            if success and self.view.url() == QUrl("about:blank"):
                completed.append(True)
                loop.quit()

        self.view.loadFinished.connect(blank_loaded)
        self.view.setUrl(QUrl("about:blank"))
        if not completed:
            timer.start(5000)
            loop.exec()
        timer.stop()
        self.view.loadFinished.disconnect(blank_loaded)