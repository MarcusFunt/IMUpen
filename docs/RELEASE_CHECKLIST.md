# IMUpen release evidence checklist

Before tagging or handing off a build, record:

- [ ] Git commit SHA and working-tree status.
- [ ] Arduino CLI, exact FQBN, ESP32 core, USB options, and compile log.
- [ ] Python version and `requirements.txt` hash.
- [ ] `tools/run_all.py` verification summary, including Python, virtual serial, native harness, 1,000 seeded scheduler cases, and sanitizer result where available.
- [ ] Representative clean and fault-injection `report.json` files with `report_schema_version` and `quality_policy_version`.
- [ ] Synthetic data manifests, seeds, configuration hashes, and simulator version where simulations were used.
- [ ] Open compiler warnings, software defects, and unresolved assumptions.
- [ ] Explicit M6 electrical/sensor evidence and M7 labeled handwriting evidence. If not performed, mark both pending and do not describe the prototype as physically verified.

Raw recordings remain immutable and should not be placed in source control. Generated files under `build/`, `captures/`, `measurements/`, `recordings/`, and `artifacts/` are ignored by Git.
