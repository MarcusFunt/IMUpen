"""Child-process-only pyserial-compatible virtual endpoint for end-to-end tests."""
import json
import os
from pathlib import Path
import time
import types


def install():
    script_path = os.environ.get("IMUPEN_VIRTUAL_SERIAL_SCRIPT")
    if not script_path:
        return
    config = json.loads(Path(script_path).read_text(encoding="utf-8"))
    shared = {"index": 0}

    class ScriptedSerial:
        def __init__(self, **_kwargs):
            self.dtr = False
            self.rts = False
            self.port = None

        def open(self):
            if config.get("open_error"):
                raise OSError(str(config["open_error"]))

        def close(self):
            pass

        def read_until(self, expected=b"\n", size=None):
            index = shared["index"]
            if index >= len(config["events"]):
                return b""
            action = config["events"][index]
            shared["index"] += 1
            delay = float(action.get("delay_s", 0.0)) * float(config.get("delay_scale", 1.0))
            if delay > 0:
                time.sleep(delay)
            if action.get("raise"):
                raise OSError(action["raise"])
            payload = action.get("data", "").encode("ascii", errors="replace")
            return payload[:size] if size is not None else payload

    module = types.ModuleType("serial")
    module.Serial = ScriptedSerial
    module.SerialException = OSError
    import sys
    sys.modules["serial"] = module


install()
