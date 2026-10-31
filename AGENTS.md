## Project Context

A Python project with 54 files across 9 directories.

## Stack

**Languages:**
- Python (83%)
- C++ (8%)
- C (6%)
- TOML (3%)

**Frameworks & Tools:**
- pytest (testing)

## Commands

```bash
pytest  # test
smartpen-collect  # smartpen-collect
smartpen-replay  # smartpen-replay
smartpen-gui  # smartpen-gui
smartpen-hwr  # smartpen-hwr
```

## Conventions

- **Naming**: mixed
- **File organization**: flat
- **Config files**: pyproject.toml
- **CI/CD**: .github/workflows/smartpen-ci.yml

## Architecture

**Key directories:**
- `firmware/`
- `prompts/`
- `reports/`
- `smartpen/`
- `tests/` - Test files

## Boundaries

**Always:**
- Run `pytest` before committing changes
- Follow mixed naming convention
- Follow flat file organization

**Ask first:**
- Adding new dependencies
- Changing project configuration files
- Modifying CI/CD pipelines

**Never:**
- Commit secrets, API keys, or .env files
- Delete or overwrite test files without understanding them
- Force push to main/master branch

<!-- agentseed:meta {"sha":"9ca15a36c904b971a7b87afd9fbaee6a0358af33","timestamp":"2026-10-31T14:23:01.743Z","format":"agentseed-v1"} -->
