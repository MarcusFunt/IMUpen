# IMUpen — complete project and validation bundle

This ZIP combines the original IMUpen acquisition project, the added hardware-free validation tools, and the complete 869-line implementation plan.

## Where to start

1. `COMPLETE_IMPLEMENTATION_PLAN.md` — detailed development roadmap and acceptance criteria.
2. `HARDWARE_FREE_TEST_PLAN.md` — existing tests, commands, and known limitations.
3. `HARDWARE_SPECIFICATIONS.md` — pin mapping, sensor variants, SPI, electrical and FIFO budgets.
4. `README.md` — original project usage documentation.
5. `tests/hardware_free/` — additional simulation harness and fault-injection tooling.

**Scope:** the native simulation harness and Python tests cannot establish physical hardware correctness, and some tests intentionally document pre-existing report defects.
