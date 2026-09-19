"""Shared test fixtures."""

from __future__ import annotations

import tempfile

import pytest


@pytest.fixture
def workdir():
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp
