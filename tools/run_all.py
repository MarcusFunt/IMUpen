#!/usr/bin/env python3
"""Cross-platform Python, native firmware-harness, and decoder test runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "hardware-free"


def run(label, command, *, env=None, cwd=ROOT):
    print(f"\n== {label} ==", flush=True)
    start = time.perf_counter()
    process = subprocess.run(command, cwd=cwd, env=env, check=False)
    result = {"name": label, "status": "PASS" if process.returncode == 0 else "FAIL",
              "return_code": process.returncode,
              "elapsed_seconds": round(time.perf_counter() - start, 3)}
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return result


def compiler_path():
    for name in ("g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError("A C++17 compiler (g++ or clang++) is required for native tests.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitizers", action="store_true",
                        help="also build and run AddressSanitizer/UndefinedBehaviorSanitizer (Windows requires LLVM-MinGW clang++)")
    args = parser.parse_args(argv)
    BUILD.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    compiler = compiler_path()
    sanitizer_compiler = None
    compiler_dir = str(Path(compiler).resolve().parent)
    env = os.environ.copy()
    env["PATH"] = compiler_dir + os.pathsep + env.get("PATH", "")
    env["PYTHONHASHSEED"] = "0"
    try:
        results.append(run("Python unit tests", [sys.executable, "-m", "unittest",
                                                   "discover", "-s", "tests", "-p", "test_*.py", "-v"], env=env))
        results.append(run("Virtual integration tests", [sys.executable, "-m", "unittest",
                                                           "discover", "-s", "tests/hardware_free",
                                                           "-p", "test_*.py", "-v"], env=env))
        decoder = BUILD / ("fifo_decoder_test.exe" if os.name == "nt" else "fifo_decoder_test")
        results.append(run("Build FIFO decoder", [compiler, "-std=c++17", "-Wall", "-Wextra", "-O2",
                                                    "-Ifirmware/IMUpen", "tests/fifo_decoder_test.cpp",
                                                    "-o", str(decoder)]))
        results.append(run("FIFO decoder native test", [str(decoder)], env=env))
        decoder_property = BUILD / ("fifo_decoder_property_test.exe" if os.name == "nt" else "fifo_decoder_property_test")
        results.append(run("Build FIFO decoder property test", [compiler, "-std=c++17", "-Wall", "-Wextra", "-O2",
                                                                  "-Ifirmware/IMUpen", "tests/fifo_decoder_property_test.cpp",
                                                                  "-o", str(decoder_property)]))
        results.append(run("FIFO decoder seeded property test", [str(decoder_property)], env=env))
        harness = BUILD / ("virtual_imu_test.exe" if os.name == "nt" else "virtual_imu_test")
        results.append(run("Build native firmware harness", [compiler, "-std=c++17", "-Wall", "-Wextra", "-O2",
                                                               "-Itests/hardware_free/stubs",
                                                               "tests/hardware_free/native_harness.cpp",
                                                               "-o", str(harness)]))
        results.append(run("Native firmware harness", [str(harness)], env=env))
        if args.sanitizers:
            sanitizer_compiler = compiler
            if os.name == "nt":
                sanitizer_compiler = shutil.which("clang++")
                if not sanitizer_compiler:
                    raise RuntimeError(
                        "Windows sanitizer tests require an LLVM-MinGW clang++ toolchain. "
                        "Install it with: winget install --id MartinStorsjo.LLVM-MinGW.UCRT -e"
                    )
            sanitizer_dir = str(Path(sanitizer_compiler).resolve().parent)
            sanitizer_env = env.copy()
            sanitizer_env["PATH"] = sanitizer_dir + os.pathsep + sanitizer_env.get("PATH", "")
            sanitizer_cases = (
                ("native firmware harness", "tests/hardware_free/native_harness.cpp",
                 "-Itests/hardware_free/stubs", "virtual_imu_sanitized"),
                ("FIFO decoder property test", "tests/fifo_decoder_property_test.cpp",
                 "-Ifirmware/IMUpen", "fifo_decoder_property_sanitized"),
            )
            for label, source, include, executable in sanitizer_cases:
                suffix = ".exe" if os.name == "nt" else ""
                sanitizer_binary = BUILD / (executable + suffix)
                results.append(run(f"Build ASan/UBSan {label}",
                                   [sanitizer_compiler, "-std=c++17", "-Wall", "-Wextra", "-O1",
                                    "-g", "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                                    include, source, "-o", str(sanitizer_binary)], env=sanitizer_env))
                results.append(run(f"ASan/UBSan {label}", [str(sanitizer_binary)], env=sanitizer_env))
    except Exception as exc:
        report = {"schema": "imupen_verification_v1", "status": "FAIL",
                  "error": str(exc), "completed_stages": results}
        (BUILD / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    report = {"schema": "imupen_verification_v1", "status": "PASS",
              "python": sys.version, "platform": platform.platform(),
              "compiler": compiler, "compiler_version": subprocess.run(
                  [compiler, "--version"], capture_output=True, text=True, check=False).stdout.splitlines()[0],
              "sanitizer_compiler": sanitizer_compiler,
              "sanitizer_compiler_version": (subprocess.run(
                  [sanitizer_compiler, "--version"], capture_output=True, text=True,
                  check=False).stdout.splitlines()[0] if sanitizer_compiler else None),
              "elapsed_seconds": round(time.perf_counter() - started, 3),
              "hardware_verified": False, "stages": results}
    (BUILD / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
