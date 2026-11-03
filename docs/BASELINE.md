# IMUpen workspace baseline

## Source and repository state

- Imported from `IMUpen-Complete-Project-Testing-and-Hardware-Plan.zip` on 2026-11-03.
- Source archive SHA-256: `78C45DACEE679C0F4109F75BD1004F2BDEC4F1D8E908C939119CE188C6903B0F`.
- Existing overlapping project files were byte-identical to the ZIP; missing plan documents and the additive `tests/hardware_free/` pack were copied in.
- The local Git directory started without a commit. IMU-001 records the imported project and reviewed setup changes in a dedicated work branch; the archive hash above identifies the supplied source bundle.

## Toolchain observed on this machine

| Tool | Observed version/configuration |
| --- | --- |
| OS | Windows 10.0.26200 |
| Python | 3.13.5 |
| pyserial | 3.5 installed globally; project setup uses an isolated `.venv` |
| C++ compiler | MinGW-w64 GCC 15.2.0 (UCRT, POSIX, x86_64) |
| Sanitizer compiler | LLVM-MinGW Clang 22.1.8 (UCRT, x86_64); ASan and UBSan runtime verified |
| Arduino CLI | 1.4.1 |
| Board FQBN | `esp32:esp32:XIAO_ESP32S3` |
| Espressif Arduino core | 3.3.6 installed |
| Default USB configuration | Hardware CDC/JTAG, CDC on boot enabled |

The intended hardware-free configuration is recorded in `configs/xiao_lsm6ds3_spi_208hz.yaml`. This is documentation only; firmware constants are not generated from it.

## Validation status in this workspace

| Check | Status |
| --- | --- |
| Archive files imported and overlapping files compared | Complete |
| Python dependencies isolated in `.venv` | Complete; pyserial 3.5 installed |
| Base Python tests | Passed: 5/5 |
| Virtual Python tests | Passed: 8/8; three cases intentionally preserve known-defect behavior |
| Native C++ FIFO decoder | Passed |
| Native firmware harness | Passed: all 6 scenarios |
| XIAO firmware compile | Not run in this setup task |
| Physical sensor/electrical measurements | Not performed; hardware remains unverified |
| Handwriting feasibility | Not evaluated |

The plan archive reports earlier test results from its preparation environment. Those results are provenance notes, not a local run record. Update this table only with results actually produced from this workspace.

## Known defects retained for follow-up

The virtual suite now names these cases explicitly as known defects. Their assertions document current behavior and are expected to be rewritten as positive regression checks when the corresponding fixes land:

- Nonzero FIFO patterns do not receive reconstructed timestamp models.
- Clipped/saturated raw values are not reported as a quality issue.
- An empty CSV report has no explicit failed-verdict field.

No changes were made to `firmware/IMUpen/IMUpen.ino` to accommodate the mocks.
