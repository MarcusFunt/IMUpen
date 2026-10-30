"""Prompted single-writer terminal collector.

A background thread continuously drains serial bytes while the foreground
prompts for start/stop and accept/reject. This avoids losing samples while
waiting for keyboard input. A GUI can reuse ReaderWorker and RecordingStore.
"""
from __future__ import annotations

import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path

from smartpen.collector.recording import CaptureBuffer
from smartpen.collector.serial_reader import SerialFrameReader
from smartpen.collector.storage import RecordingStore
from smartpen.protocol import Config, FrameParser

DEFAULT_PROMPTS = [
    "hello world",
    "the quick brown fox",
    "writing with an inertial pen",
    "a small experiment",
    "the pen moves across paper",
]


class ReaderWorker:
    """Own the serial-reading thread and guard active capture transitions."""

    def __init__(self, reader: SerialFrameReader):
        self.reader = reader
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.active: CaptureBuffer | None = None
        self.latest_config: Config | None = None
        self.error: Exception | None = None
        self.monitor = FrameParser()

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                chunk = self.reader.read_chunk()
                if not chunk:
                    continue
                with self.lock:
                    for frame in self.monitor.feed(chunk):
                        if isinstance(frame.message, Config):
                            self.latest_config = frame.message
                    if self.active is not None:
                        self.active.feed(chunk)
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def start_capture(self) -> None:
        with self.lock:
            if self.active is not None:
                raise RuntimeError("A recording is already in progress")
            self.active = CaptureBuffer()

    def stop_capture(self) -> tuple[CaptureBuffer, Config | None]:
        with self.lock:
            if self.active is None:
                raise RuntimeError("No recording in progress")
            capture = self.active
            self.active = None
            return capture, self.latest_config

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2)


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [line for line in prompts if line and not line.startswith("#")]
    if not prompts:
        raise ValueError("Prompt file is empty")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled handwriting from an ESP32-S3")
    parser.add_argument("--port", required=True, help="USB CDC serial port, e.g. COM6")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--writer", default="p01")
    parser.add_argument("--session",
                        default=datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M"))
    parser.add_argument("--prompts", type=Path, help="UTF-8 text file, one prompt per line")
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    store = RecordingStore(args.output)
    with SerialFrameReader(args.port, args.baud) as reader:
        worker = ReaderWorker(reader)
        worker.start()
        try:
            print("Connected. USB CDC may reset the board; wait for config frames.")
            print("Use the same pen grip and mounting for this session.")
            for index, prompt in enumerate(prompts, start=1):
                if worker.error:
                    raise RuntimeError("Serial reader failed") from worker.error
                print(f"\n[{index}/{len(prompts)}] WRITE: {prompt}")
                input("Press ENTER to START recording...")
                worker.start_capture()
                input("Write on paper, then press ENTER to STOP...")
                capture, config = worker.stop_capture()
                print(f"Valid samples: {capture.sample_count}; "
                      f"CRC errors: {capture.parser.crc_errors}")
                if not capture.sample_count:
                    print("No valid samples: retry after checking wiring/serial.")
                    continue
                answer = input("Accept [Y], reject [n], quit [q]? ").strip().lower()
                if answer == "q":
                    break
                accepted = answer not in ("n", "no")
                metadata = store.save(
                    writer_id=args.writer, session_id=args.session,
                    prompt=prompt, raw_bytes=bytes(capture.raw),
                    frames=capture.frames, accepted=accepted,
                    fallback_config=config,
                    parser_crc_errors=capture.parser.crc_errors,
                    parser_header_errors=capture.parser.header_errors,
                )
                print(f"Saved {metadata['sample_id']} "
                      f"({'accepted' if accepted else 'rejected'})")
        finally:
            worker.close()
    print("Session complete.")


if __name__ == "__main__":
    main()
