"""Load the scripted serial shim only in explicitly configured child tests."""
import os

if os.environ.get("IMUPEN_VIRTUAL_SERIAL_SCRIPT"):
    from virtual_serial_support import install
    install()
