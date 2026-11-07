"""Smoke test: the package installs and imports."""

import frugalgpt


def test_package_imports() -> None:
    assert frugalgpt.__version__ == "0.1.0"
    assert frugalgpt.DEFAULT_SEED == 2024
