"""Lightweight code and dataset provenance helpers for research runs."""
from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def code_provenance() -> dict[str, str | bool | None]:
    """Fingerprint the installed HWR source and, when available, its Git state."""
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for source_file in sorted(source_root.glob("*.py")):
        digest.update(source_file.name.encode("utf-8"))
        digest.update(b"\0")
        source_bytes = source_file.read_bytes()
        digest.update(len(source_bytes).to_bytes(8, "big"))
        digest.update(source_bytes)
    repository_root = source_root.parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "smartpen/hwr"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=3,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit = None
        dirty = None
    return {
        "source_tree_sha256": digest.hexdigest(),
        "git_commit": commit,
        "hwr_worktree_dirty": dirty,
    }
