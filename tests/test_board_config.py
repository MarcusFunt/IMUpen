"""Static hardware target guard: prevent accidental reversion to the wrong GPIOs."""
from configparser import ConfigParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_target_board_and_exposed_i2c_pads():
    pio = ConfigParser(interpolation=None)
    assert pio.read(ROOT / "firmware" / "platformio.ini")
    env = pio["env:seeed_xiao_esp32s3"]
    assert env["board"].strip() == "seeed_xiao_esp32s3"
    flags = env["build_flags"].split()
    assert "-DIMU_SDA_PIN=5" in flags   # D4
    assert "-DIMU_SCL_PIN=6" in flags   # D5
    assert "-DARDUINO_USB_CDC_ON_BOOT=1" in flags
