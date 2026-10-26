"""Shared pytest setup: repo on the path, and opt-in flags for the expensive tests.

Everything runs offline unless a test is marked `slow` (downloads a model and trains it) or `live`
(needs benchmark datasets from the HF cache or the network). Those are skipped before their bodies
run, so plain `pytest` is safe on a login node.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OPT_IN = {
    "slow": ("--slow", "downloads a model and trains it; opt in with --slow"),
    "live": ("--live", "needs benchmark datasets (HF cache or network); opt in with --live"),
}


def pytest_addoption(parser):
    for mark, (flag, help_text) in OPT_IN.items():
        parser.addoption(flag, action="store_true", default=False, help=help_text)


def pytest_collection_modifyitems(config, items):
    for mark, (flag, help_text) in OPT_IN.items():
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=help_text)
        for item in items:
            if mark in item.keywords:
                item.add_marker(skip)
