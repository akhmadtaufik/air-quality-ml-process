#!/usr/bin/env python3
"""Helper script to run the test suite with optional coverage."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]

    if importlib.util.find_spec("pytest_cov") is not None:
        cmd.extend(["--cov=src", "--cov-report=html"])
    else:
        print("⚠ pytest-cov not installed; running tests without coverage.")

    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
