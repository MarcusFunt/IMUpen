## Project Context

A Python project with 25 files across 7 directories.

## Stack

**Languages:**
- Python (68%)
- C++ (16%)
- C (11%)
- TOML (5%)

**Frameworks & Tools:**
- pytest (testing)

## Commands

```bash
pytest  # test
smartpen-collect  # smartpen-collect
smartpen-replay  # smartpen-replay
smartpen-gui  # smartpen-gui
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

<!-- agentseed:meta {"sha":"9b2234de2122da20937181f44bda5b48e704124e","timestamp":"2026-10-30T09:45:47.033Z","format":"agentseed-v1"} -->
