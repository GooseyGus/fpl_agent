"""Smoke test: every module in the package imports, so moved files can't silently break."""

import importlib
import pkgutil

import pytest

import fpl_hq

MODULES = [module.name for module in pkgutil.walk_packages(fpl_hq.__path__, "fpl_hq.")]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
