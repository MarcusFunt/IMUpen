"""
Dear PyGui demo showcasing NVIDIA's Parakeet-TDT automatic speech recognition model.

Steps:
1. Install dependencies: pip install -r requirements.txt
2. Run this script: python parakeet_tdt_demo.py
3. Use the "Open Audio" button to select a WAV/FLAC audio file.
"""
import threading
from pathlib import Path
from typing import Optional

import dearpygui.dearpygui as dpg
from transformers import pipeline

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


class ParakeetDemo:
    def __init__(self) -> None:
        self._pipe = None
        self._running_thread: Optional[threading.Thread] = None

    def _set_status(self, message: str) -> None:
        dpg.set_value("status_text", message)

    def _set_transcript(self, text: str) -> None:
        dpg.set_value("transcript_output", text)

    def _load_pipeline(self):
        if self._pipe is None:
            self._set_status("Loading model (this can take a bit)...")
            self._pipe = pipeline(
                task="automatic-speech-recognition",
                model=MODEL_ID,
                device_map="auto",
            )

    def _run_transcription(self, audio_path: Path) -> None:
        try:
            self._load_pipeline()
            self._set_status("Transcribing audio...")
            result = self._pipe(str(audio_path))
            transcript = result.get("text", "")
            self._set_transcript(transcript)
            self._set_status("Done")
        except Exception as exc:  # noqa: BLE001 - user-facing demo needs broad handling
            self._set_status(f"Error: {exc}")

    def request_transcription(self, sender, app_data) -> None:  # noqa: ANN001
        if self._running_thread and self._running_thread.is_alive():
            self._set_status("Already transcribing. Please wait...")
            return

        selection = app_data.get("selections", {}) if app_data else {}
        if not selection:
            self._set_status("No file selected.")
            return

        file_path = Path(next(iter(selection.values())))
        self._set_status(f"Selected: {file_path.name}")

        def worker():
            self._run_transcription(file_path)

        self._running_thread = threading.Thread(target=worker, daemon=True)
        self._running_thread.start()

    def setup_ui(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Parakeet-TDT ASR Demo", width=700, height=500)

        with dpg.window(label="Parakeet-TDT 0.6B v3", width=-1, height=-1):
            dpg.add_text("Select a short WAV/FLAC file to transcribe with NVIDIA's Parakeet-TDT model.")
            dpg.add_text(f"Model: {MODEL_ID}")
            dpg.add_spacer(height=5)
            dpg.add_button(label="Open Audio", callback=lambda: dpg.show_item("file_dialog"))
            dpg.add_same_line()
            dpg.add_button(label="Clear", callback=lambda: self._set_transcript(""))
            dpg.add_spacer(height=10)
            dpg.add_text("Status: ", bullet=True)
            dpg.add_text("Waiting for audio file...", tag="status_text")
            dpg.add_spacer(height=10)
            dpg.add_text("Transcript:")
            dpg.add_input_text(tag="transcript_output", multiline=True, readonly=True, width=-1, height=300)

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.request_transcription,
            file_count=1,
            tag="file_dialog",
        ):
            dpg.add_file_extension(".wav", color=(0, 255, 0, 255))
            dpg.add_file_extension(".flac", color=(0, 150, 250, 255))
            dpg.add_file_extension(".*")

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window(dpg.last_item(), True)
        dpg.start_dearpygui()
        dpg.destroy_context()


def main() -> None:
    demo = ParakeetDemo()
    demo.setup_ui()


if __name__ == "__main__":
    main()
