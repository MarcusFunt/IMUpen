"""Optional PySide6 prompted handwriting recorder.

Install: pip install -e ".[gui]"
Run:     smartpen-gui --port COM6 --writer p01 --session day01

The GUI never sends the displayed prompt to the ESP32-S3; it is saved only
as a supervised training label on the host.
"""
from __future__ import annotations

import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QVBoxLayout, QWidget,
)

from smartpen.collector.cli import load_prompts
from smartpen.collector.recording import CaptureBuffer
from smartpen.collector.serial_reader import SerialFrameReader
from smartpen.collector.storage import RecordingStore
from smartpen.protocol import Config, FrameParser


class SerialPump(QThread):
    received = Signal(bytes)
    failed = Signal(str)

    def __init__(self, port: str, baud: int):
        super().__init__()
        self.port = port
        self.baud = baud
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            with SerialFrameReader(self.port, self.baud) as reader:
                while not self._stop.is_set():
                    chunk = reader.read_chunk()
                    if chunk:
                        self.received.emit(chunk)
        except Exception as exc:
            self.failed.emit(str(exc))

    def stop(self) -> None:
        self._stop.set()


class CollectorWindow(QMainWindow):
    def __init__(
        self, port: str, baud: int, prompts: list[str],
        store: RecordingStore, writer_id: str, session_id: str
    ):
        super().__init__()
        self.prompts = prompts
        self.store = store
        self.writer_id = writer_id
        self.session_id = session_id
        self.index = 0
        self.monitor = FrameParser()
        self.latest_config: Config | None = None
        self.active: CaptureBuffer | None = None
        self.pending: CaptureBuffer | None = None

        self.setWindowTitle("SmartPen handwriting collector")
        central = QWidget()
        layout = QVBoxLayout(central)
        self.progress = QLabel()
        self.prompt_label = QLabel()
        self.prompt_label.setWordWrap(True)
        font = QFont()
        font.setPointSize(20)
        self.prompt_label.setFont(font)
        self.status = QLabel("Connecting to USB serial...")
        self.sample_label = QLabel("Samples: 0")
        layout.addWidget(self.progress)
        layout.addWidget(self.prompt_label)
        layout.addWidget(self.status)
        layout.addWidget(self.sample_label)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start recording")
        self.stop_button = QPushButton("Stop recording")
        self.accept_button = QPushButton("Accept")
        self.reject_button = QPushButton("Reject / retry")
        self.skip_button = QPushButton("Skip prompt")
        for button in (self.start_button, self.stop_button,
                       self.accept_button, self.reject_button, self.skip_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.setCentralWidget(central)
        self.start_button.clicked.connect(self.start_recording)
        self.stop_button.clicked.connect(self.stop_recording)
        self.accept_button.clicked.connect(lambda: self.save(True))
        self.reject_button.clicked.connect(lambda: self.save(False))
        self.skip_button.clicked.connect(self.advance)
        self._update_prompt()

        self.pump = SerialPump(port, baud)
        self.pump.received.connect(self._on_bytes)
        self.pump.failed.connect(self._on_error)
        self.pump.start()
        self.resize(820, 250)

    def _update_prompt(self) -> None:
        if self.index >= len(self.prompts):
            self.progress.setText("Session complete")
            self.prompt_label.setText("All prompts recorded.")
            self.start_button.setEnabled(False)
            self.skip_button.setEnabled(False)
        else:
            self.progress.setText(f"Prompt {self.index + 1} of {len(self.prompts)}")
            self.prompt_label.setText(self.prompts[self.index])
            self.start_button.setEnabled(self.active is None and self.pending is None)
            self.skip_button.setEnabled(self.active is None and self.pending is None)
        self.stop_button.setEnabled(self.active is not None)
        self.accept_button.setEnabled(self.pending is not None)
        self.reject_button.setEnabled(self.pending is not None)

    def _on_bytes(self, chunk: bytes) -> None:
        for frame in self.monitor.feed(chunk):
            if isinstance(frame.message, Config):
                self.latest_config = frame.message
        if self.active is not None:
            self.active.feed(chunk)
            self.sample_label.setText(
                f"Samples: {self.active.sample_count}, "
                f"CRC errors: {self.active.parser.crc_errors}"
            )
        elif self.latest_config is not None and self.pending is None:
            self.status.setText(
                f"Connected: {self.latest_config.sample_rate_hz} Hz, "
                f"I2C 0x{self.latest_config.i2c_address:02X}"
            )

    def _on_error(self, message: str) -> None:
        self.status.setText(f"Serial error: {message}")
        QMessageBox.critical(self, "Serial connection failed", message)
        self.start_button.setEnabled(False)

    def start_recording(self) -> None:
        if self.index >= len(self.prompts) or self.active is not None:
            return
        self.active = CaptureBuffer()
        self.status.setText("Recording: write on paper, then press Stop")
        self.sample_label.setText("Samples: 0")
        self._update_prompt()

    def stop_recording(self) -> None:
        if self.active is None:
            return
        self.pending = self.active
        self.active = None
        self.status.setText(
            f"Stopped: {self.pending.sample_count} valid samples. Accept or retry."
        )
        self._update_prompt()

    def save(self, accepted: bool) -> None:
        if self.pending is None:
            return
        if not self.pending.sample_count:
            QMessageBox.warning(self, "No data", "No valid samples were received.")
            self.pending = None
            self._update_prompt()
            return
        try:
            meta = self.store.save(
                writer_id=self.writer_id,
                session_id=self.session_id,
                prompt=self.prompts[self.index],
                raw_bytes=bytes(self.pending.raw),
                frames=self.pending.frames,
                accepted=accepted,
                fallback_config=self.latest_config,
                parser_crc_errors=self.pending.parser.crc_errors,
                parser_header_errors=self.pending.parser.header_errors,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.pending = None
        self.status.setText(
            f"Saved {meta['sample_id']} ({'accepted' if accepted else 'rejected'})"
        )
        if accepted:
            self.index += 1
        self._update_prompt()

    def advance(self) -> None:
        if self.active is not None or self.pending is not None:
            return
        self.index += 1
        self._update_prompt()

    def closeEvent(self, event) -> None:
        if self.active is not None or self.pending is not None:
            choice = QMessageBox.question(
                self, "Unsaved recording",
                "A recording is not saved. Close anyway?"
            )
            if choice != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.pump.stop()
        self.pump.wait(2000)
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description="SmartPen prompted GUI recorder")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--writer", default="p01")
    parser.add_argument("--session",
                        default=datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M"))
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    app = QApplication([])
    window = CollectorWindow(args.port, args.baud, load_prompts(args.prompts),
                             RecordingStore(args.output), args.writer, args.session)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
